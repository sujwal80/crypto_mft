import asyncio
import logging
import time
from typing import Callable, Optional
import aiohttp

from core.schemas import InternalTick
from core.exceptions import DataStallException, RateLimitException

logger = logging.getLogger(__name__)

class IngestionWatchdog:
    """Monitors WebSocket health, detects silent data stalls, and manages REST backfill reconciliation."""
    def __init__(self, symbol: str, exchange: str, rest_url: str, timeout_seconds: float = 5.0):
        self.symbol = symbol
        self.exchange = exchange
        self.rest_url = rest_url
        self.timeout_seconds = timeout_seconds
        self.last_tick_time: float = time.time()
        self.is_connected: bool = False
        self.monitoring_task: Optional[asyncio.Task] = None

    def update_tick_time(self):
        self.last_tick_time = time.time()

    async def monitor_health(self, reconnect_callback: Callable):
        """Background task that checks for silent data stalls."""
        logger.info(f"Watchdog monitoring started for {self.symbol} (Stall threshold: {self.timeout_seconds}s).")
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
                    raise RateLimitException(f"Rate limited on REST backfill for {self.symbol}.")
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
        except RateLimitException as rle:
            logger.warning(str(rle))
        except Exception as e:
            logger.error(f"REST Backfill failed for {self.symbol}: {e}")
