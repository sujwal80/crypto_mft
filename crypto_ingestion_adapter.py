import asyncio
import logging
import time
from typing import Dict, List, Optional
import aiohttp
import websockets
from pydantic import BaseModel, ValidationError, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# Pydantic Schema Validation (Fault Isolation)
# ======================================================================
class InternalTick(BaseModel):
    """Unified internal tick structure representing normalized market data across all exchanges."""
    symbol: str
    exchange: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp_ns: int = Field(..., description="Epoch timestamp in nanoseconds")

class BinanceDepthEntry(BaseModel):
    price: str
    quantity: str

class BinanceDepthPayload(BaseModel):
    """Validates incoming Binance Level 2 order book snapshots/updates."""
    e: str # Event type
    E: int # Event time (ms)
    s: str # Symbol
    b: List[BinanceDepthEntry] # Bids
    a: List[BinanceDepthEntry] # Asks

# ======================================================================
# Ingestion Watchdog (Monitors Health & Stalls)
# ======================================================================
class IngestionWatchdog:
    """Monitors WebSocket health, detects silent data stalls, and manages REST backfill reconciliation."""
    def __init__(self, symbol: str, exchange: str, rest_url: str, timeout_seconds: float = 5.0):
        self.symbol = symbol
        self.exchange = exchange
        self.rest_url = rest_url
        self.timeout_seconds = timeout_seconds
        self.last_tick_time: float = time.time()
        self.is_connected: bool = False

    def update_tick_time(self):
        self.last_tick_time = time.time()

    async def monitor_health(self, queue: asyncio.Queue, reconnect_callback):
        """Background task that checks for silent data stalls."""
        while True:
            await asyncio.sleep(1)
            if self.is_connected:
                time_gap = time.time() - self.last_tick_time
                if time_gap > self.timeout_seconds:
                    logger.warning(f"SILENT DATA STALL DETECTED for {self.symbol} ({time_gap:.1f}s). Force restarting socket...")
                    self.is_connected = False
                    await reconnect_callback()
                    
    async def reconcile_missing_data(self, queue: asyncio.Queue, session: aiohttp.ClientSession):
        """Queries REST API to backfill missing ticks during a socket disconnection."""
        logger.info(f"Executing REST Backfill Reconciliation for {self.symbol}...")
        try:
            async with session.get(self.rest_url) as response:
                if response.status == 429:
                    logger.warning(f"Rate limited on REST backfill for {self.symbol}. Backing off...")
                    return
                if response.status == 200:
                    data = await response.json()
                    # Parse Binance REST order book snapshot
                    bids = data.get("bids", [["0", "0"]])
                    asks = data.get("asks", [["0", "0"]])
                    
                    recovery_tick = InternalTick(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        bid=float(bids[0][0]),
                        ask=float(asks[0][0]),
                        bid_size=float(bids[0][1]),
                        ask_size=float(asks[0][1]),
                        timestamp_ns=int(time.time() * 1e9)
                    )
                    await queue.put(recovery_tick)
                    logger.info(f"Successfully patched internal queue with REST snapshot for {self.symbol}.")
        except Exception as e:
            logger.error(f"REST Backfill failed for {self.symbol}: {e}")

# ======================================================================
# Multi-Exchange WebSocket Adapter Pool
# ======================================================================
class BinanceCryptoAdapter:
    def __init__(self, symbol: str, wss_url: str, rest_url: str):
        self.symbol = symbol
        self.wss_url = wss_url
        self.rest_url = rest_url
        self.watchdog = IngestionWatchdog(symbol=symbol, exchange="BINANCE", rest_url=rest_url)
        self.session: Optional[aiohttp.ClientSession] = None
        self.queue: Optional[asyncio.Queue] = None
        self.reconnect_event = asyncio.Event()

    async def force_reconnect(self):
        """Callback triggered by Watchdog when a stall is detected."""
        self.reconnect_event.set()

    async def connect_and_stream(self, queue: asyncio.Queue):
        self.queue = queue
        self.session = aiohttp.ClientSession()
        
        # Start Watchdog Monitoring
        asyncio.create_task(self.watchdog.monitor_health(queue, self.force_reconnect))
        
        backoff = 1.0
        while True:
            try:
                logger.info(f"Connecting to Binance WSS for {self.symbol}...")
                async with websockets.connect(self.wss_url) as ws:
                    self.watchdog.is_connected = True
                    backoff = 1.0 # Reset backoff on successful connection
                    
                    # Execute REST reconciliation backfill upon reconnection
                    await self.watchdog.reconcile_missing_data(queue, self.session)
                    
                    while not self.reconnect_event.is_set():
                        raw_message = await ws.recv()
                        self.watchdog.update_tick_time()
                        
                        try:
                            # Pydantic Schema Validation
                            payload = BinanceDepthPayload.model_validate_json(raw_message)
                            
                            # Extract Best Bid / Best Ask
                            bid = float(payload.b[0].price) if payload.b else 0.0
                            bid_size = float(payload.b[0].quantity) if payload.b else 0.0
                            ask = float(payload.a[0].price) if payload.a else 0.0
                            ask_size = float(payload.a[0].quantity) if payload.a else 0.0
                            
                            # Normalize Timestamp to Epoch Nanoseconds
                            timestamp_ns = payload.E * 1_000_000 # ms to ns
                            
                            internal_tick = InternalTick(
                                symbol=self.symbol,
                                exchange="BINANCE",
                                bid=bid,
                                ask=ask,
                                bid_size=bid_size,
                                ask_size=ask_size,
                                timestamp_ns=timestamp_ns
                            )
                            
                            await queue.put(internal_tick)
                        except ValidationError as ve:
                            logger.error(f"Schema Validation Error on Binance stream: {ve}")
                        except Exception as e:
                            logger.error(f"Error parsing tick: {e}")
                            
            except (websockets.WebSocketException, ConnectionError) as we:
                logger.warning(f"Binance WebSocket Disconnected: {we}. Reconnecting in {backoff}s...")
                self.watchdog.is_connected = False
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0) # Exponential backoff up to 30s
            finally:
                self.reconnect_event.clear()

# ======================================================================
# Execution Pipeline (Consumer Demonstration)
# ======================================================================
async def feature_engine_consumer(queue: asyncio.Queue):
    """Demonstrates Phase 2 consuming normalized ticks from the queue."""
    logger.info("Feature Engine Consumer started.")
    while True:
        tick: InternalTick = await queue.get()
        logger.info(f"INGESTED TICK -> {tick.model_dump()}")
        queue.task_done()

async def main():
    # Configuration for BTC/USDT on Binance
    SYMBOL = "BTCUSDT"
    BINANCE_WSS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth5@100ms"
    BINANCE_REST_URL = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit=5"
    
    # Create central asyncio Queue for inter-process decoupling
    message_queue = asyncio.Queue()
    
    # Initialize Adapter
    binance_adapter = BinanceCryptoAdapter(symbol=SYMBOL, wss_url=BINANCE_WSS_URL, rest_url=BINANCE_REST_URL)
    
    # Concurrently run Ingestion Adapter and Feature Consumer
    await asyncio.gather(
        binance_adapter.connect_and_stream(message_queue),
        feature_engine_consumer(message_queue)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading system shutdown cleanly.")
