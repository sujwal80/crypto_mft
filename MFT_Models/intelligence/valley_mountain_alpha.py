import numpy as np
import logging
from collections import deque
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class ValleyMountainAlpha(BaseAlphaStrategy):
    """
    Valley-Mountain (Buy the Dip, Sell the High) Strategy.
    Buys when the price is at or near a rolling minimum (valley).
    Sells (or Shorts) when the price is at or near a rolling maximum (mountain).
    """
    def __init__(
        self,
        lookback: int = 500,
        entry_buffer: float = 0.0005,  # 5 bps buffer near extremes
        ofi_threshold: float = 0.20,   # Minimum buyer imbalance to confirm bottom
        forecast_value: float = 0.002,  # Calibrated forecast size
        **kwargs
    ):
        self.lookback = lookback
        self.entry_buffer = entry_buffer
        self.ofi_threshold = ofi_threshold
        self.forecast_value = forecast_value
        
        self.prices = deque(maxlen=self.lookback)
        logger.info(f"ValleyMountainAlpha initialized: lookback={self.lookback}, buffer={self.entry_buffer}, ofi_threshold={self.ofi_threshold}")

    def predict(self, features: np.ndarray) -> float:
        """
        Args:
            features: [z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price]
        """
        rolling_imbalance = features[2]
        mid_price = features[5]
        
        self.prices.append(mid_price)
        
        if len(self.prices) < self.lookback:
            # Warm-up period
            return 0.0
            
        prices_arr = np.array(self.prices)
        rolling_min = np.min(prices_arr)
        rolling_max = np.max(prices_arr)
        
        # Check if we are in a "valley" (near rolling min)
        is_in_valley = mid_price <= rolling_min * (1.0 + self.entry_buffer)
        
        # Check if we are on a "mountain" (near rolling max)
        is_on_mountain = mid_price >= rolling_max * (1.0 - self.entry_buffer)
        
        if is_in_valley:
            # Price is at a valley -> BUY ONLY if order flow shows buyer aggression (OFI > threshold)
            if rolling_imbalance > self.ofi_threshold:
                return self.forecast_value
            
        if is_on_mountain:
            # Price is at a mountain -> SELL/SHORT ONLY if order flow shows seller aggression (OFI < -threshold)
            if rolling_imbalance < -self.ofi_threshold:
                return -self.forecast_value
            
        # In the middle of the range -> neutral
        return 0.0
