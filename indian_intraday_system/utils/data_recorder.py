"""Asynchronous non-blocking live market data recorder.

Populates local data lakes while paper trading.
"""

import asyncio
import json
import os
import time
from typing import List
import pandas as pd


class LiveDataRecorder:
    """Buffers incoming ticks in memory and flushes them asynchronously to disk."""

    def __init__(self, output_dir: str = "./datasets/raw_ticks"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.buffer: List[dict] = []
        self.lock = asyncio.Lock()
        self.active = False
        self._flush_task = None
        self.current_date_str = time.strftime("%Y-%m-%d")

    def start(self):
        """Starts the background flushing loop."""
        self.active = True
        self.current_date_str = time.strftime("%Y-%m-%d")
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())
        print(f"[DataRecorder] Background tick recorder armed. Output: {self.output_dir}")

    async def record_tick(self, tick: dict):
        """Inserts a tick payload into the async buffer."""
        if not self.active:
            return
        
        # Clean metadata for storage
        tick_copy = tick.copy()
        tick_copy["local_timestamp"] = time.time()
        
        async with self.lock:
            self.buffer.append(tick_copy)

    async def _periodic_flush_loop(self):
        """Asynchronously flushes buffers to disk every 30 seconds."""
        while self.active:
            await asyncio.sleep(30.0)
            await self.flush_to_disk()

    async def flush_to_disk(self):
        """Writes buffered ticks in memory to compressed local Parquet or CSV files."""
        if not self.buffer:
            return

        # Swap buffer
        async with self.lock:
            flush_data = self.buffer
            self.buffer = []

        try:
            # Create pandas DataFrame
            df = pd.DataFrame(flush_data)

            # Target path (daily file)
            file_path = os.path.join(self.output_dir, f"ticks_{self.current_date_str}.csv")
            
            # If file doesn't exist, write with header; else append without header
            # Using standard CSV for ease of readability, or Parquet if pandas support is active.
            # Since we installed pandas in our venv, we can append seamlessly
            header = not os.path.exists(file_path)
            
            # Execute blocking I/O in a thread pool to ensure zero impact on trade execution latency
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: df.to_csv(file_path, mode="a", index=False, header=header)
            )
            
            print(f"[DataRecorder] Flushed {len(flush_data)} live ticks to local data lake: {file_path}")
        except Exception as e:
            print(f"[DataRecorder] ERROR flushing ticks to disk: {e}")

    async def stop(self):
        """Shuts down the recorder and performs final flush."""
        print("[DataRecorder] Shuts down tick recorder. Flushing remaining buffers...")
        self.active = False
        if self._flush_task:
            self._flush_task.cancel()
        await self.flush_to_disk()
        print("[DataRecorder] Tick recorder shut down successfully.")
