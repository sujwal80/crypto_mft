import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class BarResampler:
    """
    Microstructure Perception Layer.
    Converts raw sub-second Level 2 depth ticks and AggTrades into uniform resampled bars.
    Supports both time-based intervals (seconds) and tick-count-based intervals.
    Allows configuring prioritization for production (time-based) vs backtesting (tick-based).
    """
    def __init__(self, bar_duration_seconds: int = 60, bar_duration_ticks: Optional[int] = None, prioritize_time: bool = True):
        self.duration_seconds = bar_duration_seconds
        self.duration_ticks = bar_duration_ticks
        self.prioritize_time = prioritize_time
        
        self.current_bar: Dict = {}
        self.last_bar_close_time = 0.0
        self.cumulative_volume_delta = 0.0
        self.tick_counter = 0

    def process_tick(self, price: float, bid_qty: float, ask_qty: float, 
                     is_trade: bool, trade_price: float = 0.0, trade_qty: float = 0.0, 
                     is_buyer_maker: bool = False, timestamp_ns: int = None) -> Optional[Dict]:
        """
        Aggregates raw sub-second depth tick or trade.
        Returns completed resampled bar Dict when interval has elapsed, otherwise returns None.
        """
        self.tick_counter += 1
        
        # Retrieve current clock time
        now = float(timestamp_ns / 1e9) if timestamp_ns is not None else time.time()
        
        if not self.current_bar:
            # Initialize new resampled bar
            self.last_bar_close_time = now
            self.tick_counter = 1
            self.current_bar = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "volume": 0.0,
                "cvd": 0.0,
                "timestamp": now
            }
            
        # Update prices
        self.current_bar["high"] = max(self.current_bar["high"], price)
        self.current_bar["low"] = min(self.current_bar["low"], price)
        self.current_bar["close"] = price
        self.current_bar["bid_qty"] = bid_qty
        self.current_bar["ask_qty"] = ask_qty
        
        if is_trade:
            self.current_bar["volume"] += trade_qty
            # Buy/Sell CVD mapping: is_buyer_maker=True represents seller taker (aggressive sell)
            trade_delta = -trade_qty if is_buyer_maker else trade_qty
            self.current_bar["cvd"] += trade_delta
            self.cumulative_volume_delta += trade_delta
            
        # Check if resampled interval is completed
        is_completed = False
        if self.duration_ticks is not None and not self.prioritize_time:
            # Tick-based resampling takes absolute priority when prioritize_time is disabled
            if self.tick_counter >= self.duration_ticks:
                is_completed = True
        elif timestamp_ns is not None and self.prioritize_time:
            # Time-based resampling takes absolute priority when timestamps are available
            if now - self.last_bar_close_time >= self.duration_seconds:
                is_completed = True
        elif self.duration_ticks is not None:
            # Fallback to tick-based resampling ONLY if time is blind
            if self.tick_counter >= self.duration_ticks:
                is_completed = True
                
        if is_completed:
            completed_bar = self.current_bar.copy()
            completed_bar["cumulative_cvd"] = self.cumulative_volume_delta
            
            # Reset for next interval bar
            self.current_bar = {}
            self.tick_counter = 0
            return completed_bar
            
        return None
