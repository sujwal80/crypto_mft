import asyncio
import json
import logging
import websockets
from typing import Optional

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    """
    Production-grade, asynchronous WebSocket client for Binance Futures L2 Depth & Trades.
    Implements robust auto-reconnect backoff, connection state tracking, and high-speed event queue feeding.
    """
    def __init__(self, symbol: str = "btcusdt", depth_queue: Optional[asyncio.Queue] = None, trade_queue: Optional[asyncio.Queue] = None):
        self.symbol = symbol.lower()
        self.depth_queue = depth_queue or asyncio.Queue()
        self.trade_queue = trade_queue or asyncio.Queue()
        self.running = False
        self.depth_url = f"wss://fstream.binance.com/ws/{self.symbol}@depth20@100ms"
        self.trade_url = f"wss://fstream.binance.com/ws/{self.symbol}@aggTrade"
        self.tasks = []

    async def start(self):
        """Starts the asynchronous ingestion loops."""
        self.running = True
        self.tasks = [
            asyncio.create_task(self._connect_loop(self.depth_url, self.depth_queue, "DepthL2")),
            asyncio.create_task(self._connect_loop(self.trade_url, self.trade_queue, "AggTrade"))
        ]
        logger.info(f"Binance WS client started for {self.symbol}")

    async def stop(self):
        """Stops the ingestion loops gracefully."""
        self.running = False
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("Binance WS client stopped")

    async def _connect_loop(self, url: str, queue: asyncio.Queue, stream_name: str):
        """WebSocket connection loop with exponential backoff reconnect logic."""
        backoff = 1.0
        max_backoff = 60.0
        
        while self.running:
            try:
                logger.info(f"Connecting to Binance {stream_name} WebSocket: {url}")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info(f"Binance {stream_name} WebSocket connected successfully")
                    backoff = 1.0  # Reset backoff on success
                    
                    while self.running:
                        message = await ws.recv()
                        data = json.loads(message)
                        
                        # Push event to the in-memory high-speed queue
                        # Use non-blocking put_nowait to prevent event loop choking
                        try:
                            queue.put_nowait(data)
                        except asyncio.QueueFull:
                            # Drop oldest item if queue is full to prevent memory leaks / latency build-up
                            queue.get_nowait()
                            queue.put_nowait(data)
                            logger.warning(f"Binance {stream_name} Queue full, oldest event dropped")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Binance {stream_name} WS connection error: {e}. Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, max_backoff)
