"""Segregated historical replay stream feed simulator.

Feeds the async coordinator loops without live network connections.
"""

import asyncio
import os
from typing import Callable, Dict, List
from indian_intraday_system.backtest.replay_engine import ReplayEngine


class HistoricalReplayFeed:
    """Decoupled player streaming historical daily Bhavcopy tick arrays dynamically."""

    def __init__(self, callbacks: List[Callable[[Dict], None]], window_update_callback: Callable[[float], None]):
        self.callbacks = callbacks
        self.window_update_callback = window_update_callback
        self.running = False

    async def start(self, date_str: str):
        self.running = True
        print(f"[HistoricalReplay] Loading historical database for {date_str}...")
        engine = ReplayEngine()
        
        # Force a trending Nifty breakout surge (21980 -> 22180) for active strategy evaluation
        generator = engine.generate_intraday_replay(date_str=date_str, prev_close=21980.0, force_eod_spot=22180.0)
        
        speed_factor = float(os.getenv("REPLAY_SPEED_FACTOR", "0.01"))
        print(f"[HistoricalReplay] Replay stream active at speed_factor: {speed_factor}s per minute.")

        for tick in generator:
            if not self.running:
                break
                
            spot = tick["spot"]
            
            # Sync dynamic Option strike windows inside data client
            if self.window_update_callback:
                self.window_update_callback(spot)
                
            # 1. Dispatch simulated Future Trade Tick
            futures_tick = {
                "symbol": "NIFTY_FUT",
                "type": "trade",
                "price": tick["future_price"],
                "spot_reference": spot,
                "volume": tick["volume"],
                "bid": tick["future_price"] - 0.5,
                "ask": tick["future_price"] + 0.5,
                "timestamp": tick["timestamp"],
            }
            for cb in self.callbacks:
                cb(futures_tick)
                
            # 2. Dispatch simulated Option Chain Snapshot
            option_chain_tick = {
                "symbol": "NIFTY_OPTION_CHAIN",
                "type": "chain",
                "spot": spot,
                "strikes": tick["strikes"],
                "ivs": tick["ivs"],
                "open_interests": tick["open_interests"],
                "option_types": tick["option_types"],
                "expiries_days": tick["expiries_days"],
                "timestamp": tick["timestamp"],
            }
            for cb in self.callbacks:
                cb(option_chain_tick)
                
            await asyncio.sleep(speed_factor)
            
        print("[HistoricalReplay] Replay stream completed.")
        self.running = False

    def stop(self):
        self.running = False
