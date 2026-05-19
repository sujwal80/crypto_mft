import asyncio
import logging
import time
from typing import Optional
import aiohttp
import websockets
from pydantic import ValidationError

from core.schemas import InternalTick, BinancePartialDepthPayload
from core.exceptions import SchemaValidationException
from ingestion.base_adapter import DataFeedAdapter
from ingestion.watchdog import IngestionWatchdog

logger = logging.getLogger(__name__)

class BinanceCryptoAdapter(DataFeedAdapter):
    """Production implementation of DataFeedAdapter for Binance WebSocket streaming."""
    def __init__(self, symbol: str, wss_url: str, rest_url: str):
        self.symbol = symbol
        self.wss_url = wss_url
        self.rest_url = rest_url
        self.watchdog = IngestionWatchdog(symbol=symbol, exchange="BINANCE", rest_url=rest_url)
        self.session: Optional[aiohttp.ClientSession] = None
        self.queue: Optional[asyncio.Queue] = None
        self.reconnect_event = asyncio.Event()
        self.is_running = False
        
    async def force_reconnect(self):
        """Callback triggered by Watchdog when a stall is detected."""
        self.reconnect_event.set()
        
    async def connect_and_stream(self, queue: asyncio.Queue):
        self.queue = queue
        self.session = aiohttp.ClientSession()
        self.is_running = True
        
        # Start Watchdog Monitoring
        self.watchdog.monitoring_task = asyncio.create_task(self.watchdog.monitor_health(self.force_reconnect))
        
        backoff = 1.0
        while self.is_running:
            try:
                logger.info(f"Connecting to Binance WSS for {self.symbol}...")
                async with websockets.connect(self.wss_url) as ws:
                    self.watchdog.is_connected = True
                    backoff = 1.0 # Reset backoff on successful connection
                    
                    # Execute REST reconciliation backfill upon reconnection
                    await self.watchdog.reconcile_missing_data(queue, self.session)
                    
                    while not self.reconnect_event.is_set() and self.is_running:
                        raw_message = await ws.recv()
                        self.watchdog.update_tick_time()
                        
                        try:
                            # Pydantic Schema Validation
                            payload = BinancePartialDepthPayload.model_validate_json(raw_message)
                            
                            # Extract Best Bid / Best Ask
                            bid = float(payload.bids[0][0])
                            bid_size = float(payload.bids[0][1])
                            ask = float(payload.asks[0][0])
                            ask_size = float(payload.asks[0][1])
                            
                            # Normalize Timestamp to Epoch Nanoseconds
                            timestamp_ns = time.time_ns()
                            
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

    async def close(self):
        """Cleanly shuts down background tasks and client sessions."""
        self.is_running = False
        if self.watchdog.monitoring_task:
            self.watchdog.monitoring_task.cancel()
        if self.session:
            await self.session.close()
        logger.info(f"Binance adapter for {self.symbol} closed cleanly.")
