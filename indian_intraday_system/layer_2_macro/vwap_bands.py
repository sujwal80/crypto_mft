import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger("VwapBands")

class IntradayVwapBands:
    """
    High-performance, recursive Intraday VWAP and Volatility Bands Calculator.
    Maintains O(1) running accumulators since the daily session open.
    """
    def __init__(self):
        self.reset_session()

    def reset_session(self):
        """Resets all session accumulators at the open of a new trading day."""
        self.sum_pv = 0.0      # Running sum of Price * Volume
        self.sum_v = 0.0       # Running sum of Volume
        
        self.sum_v_p2 = 0.0    # Running sum of Volume * Price^2 (for O(1) variance)
        self.vwap = 0.0
        self.sigma = 1.0
        self.ticks_count = 0

    def update(self, price: float, volume: float) -> Optional[Dict]:
        """
        Ingests a new price bar, updates the running accumulators recursively, 
        and returns the VWAP, volatility bands, and spread Z-score.
        """
        if volume <= 0.0:
            # Zero volume bar (pricing check only)
            if self.sum_v > 0.0:
                z_score = (price - self.vwap) / (self.sigma + 1e-8)
                return self._generate_output(price, z_score)
            return None
            
        self.ticks_count += 1
        
        # 1. Update O(1) VWAP accumulators
        self.sum_pv += price * volume
        self.sum_v += volume
        self.vwap = self.sum_pv / self.sum_v
        
        # 2. Update O(1) Volatility (Variance) accumulators
        # Variance = E[X^2] - (E[X])^2
        self.sum_v_p2 += volume * (price ** 2)
        mean_p2 = self.sum_v_p2 / self.sum_v
        variance = mean_p2 - (self.vwap ** 2)
        
        # Dynamic standard deviation
        self.sigma = np.sqrt(max(variance, 1e-8))
        
        # 3. Compute current Z-score
        z_score = (price - self.vwap) / (self.sigma + 1e-8)
        
        return self._generate_output(price, z_score)

    def _generate_output(self, price: float, z_score: float) -> Dict:
        """Helper to package standard envelope coordinates."""
        return {
            "vwap": self.vwap,
            "sigma": self.sigma,
            "z_score": z_score,
            "upper_band_1": self.vwap + 1.0 * self.sigma,
            "lower_band_1": self.vwap - 1.0 * self.sigma,
            "upper_band_2": self.vwap + 2.0 * self.sigma,
            "lower_band_2": self.vwap - 2.0 * self.sigma,
            "upper_band_3": self.vwap + 3.0 * self.sigma,
            "lower_band_3": self.vwap - 3.0 * self.sigma
        }
