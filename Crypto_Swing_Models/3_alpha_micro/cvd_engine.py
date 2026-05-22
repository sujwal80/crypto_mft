import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class CumulativeVolumeDeltaEngine:
    """
    Cumulative Volume Delta (CVD) Engine.
    Tracks live buy/sell aggressor sweeps (market order volume delta) and detects 
    divergences (e.g. price not moving despite aggressive taker sweeps, proving passive absorption).
    """
    def __init__(self, rolling_window_ticks: int = 5000):
        self.window_size = rolling_window_ticks
        self.trade_history = deque()
        
        # Rolling aggregations
        self.rolling_cvd = 0.0
        self.total_buy_volume = 0.0
        self.total_sell_volume = 0.0
        
    def process_trade(self, trade_price: float, amount: float, is_buyer_maker: bool) -> float:
        """
        Processes a single trade event and updates the rolling CVD.
        
        Args:
            trade_price: Execution price of the trade
            amount: Trade size in contracts/units
            is_buyer_maker: True if buyer was passive maker (taker was seller -> aggressive sell),
                            False if buyer was aggressive taker (taker was buyer -> aggressive buy).
                            
        Returns:
            float: Current rolling CVD value
        """
        amount = float(amount)
        trade_price = float(trade_price)
        
        # Determine taker action:
        # If buyer is maker -> Taker is Seller (Aggressive Sell -> delta -amount)
        # If buyer is taker -> Taker is Buyer (Aggressive Buy -> delta +amount)
        delta = -amount if is_buyer_maker else amount
        
        # Append to history
        self.trade_history.append((delta, amount))
        
        # Update aggregates
        self.rolling_cvd += delta
        if delta > 0:
            self.total_buy_volume += amount
        else:
            self.total_sell_volume += amount
            
        # Evict oldest trade if we exceed the rolling window size to prevent memory leaks
        if len(self.trade_history) > self.window_size:
            old_delta, old_amount = self.trade_history.popleft()
            self.rolling_cvd -= old_delta
            if old_delta > 0:
                self.total_buy_volume -= old_amount
            else:
                self.total_sell_volume -= old_amount
                
        return self.rolling_cvd

    def get_aggression_ratio(self) -> float:
        """
        Calculates the taker buy-to-sell aggression ratio over the rolling window.
        """
        if self.total_sell_volume == 0.0:
            return 1.0 if self.total_buy_volume == 0.0 else 100.0
        return self.total_buy_volume / self.total_sell_volume

    def detect_absorption_divergence(self, price_change_pct: float, threshold_cvd_zscore: float = 2.0) -> bool:
        """
        Detects passive absorption divergence:
        e.g. CVD spikes intensely (> 2.0 Z-score or massive imbalance), 
        but spot price refuses to advance (extremely flat or negative price change).
        This is the absolute Sniper confirmation trigger for reversals!
        """
        # If buy volume is vastly dominant but price did not rise, or vice versa
        if abs(self.rolling_cvd) < 10.0:
            return False
            
        agg_ratio = self.get_aggression_ratio()
        
        # Scenario A: Extreme Taker Selling (ratio < 0.3) but spot price remains flat or goes up (price_change_pct >= -0.05%)
        if agg_ratio < 0.3 and price_change_pct >= -0.0005:
            return True
            
        # Scenario B: Extreme Taker Buying (ratio > 3.0) but spot price remains flat or goes down (price_change_pct <= 0.05%)
        if agg_ratio > 3.0 and price_change_pct <= 0.0005:
            return True
            
        return False
