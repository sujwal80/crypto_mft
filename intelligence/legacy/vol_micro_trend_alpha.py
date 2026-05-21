import numpy as np
import logging
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class VolMicroTrendStrategy(BaseAlphaStrategy):
    """
    Calibrated Regime-Switching Volatility-Adaptive Micro-Trend strategy.
    Only enters BUY trades during verified upward momentum breakouts (volatility + trend).
    """
    def __init__(self, threshold: float = 0.35, sensitivity: float = 0.01, w: float = 0.85):
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.w = w

    def predict(self, features: np.ndarray) -> float:
        z_score, spread, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features
        
        # Only enter during active, moving regimes (volatility >= 0.8 bps)
        vol_rel = rolling_vol / (mid_price + 1e-8)
        is_moving = vol_rel >= 0.00008
        
        normalized_drift = micro_price_drift / spread if spread > 0.0 else 0.0
        momentum_score = (self.w * rolling_imbalance) + ((1.0 - self.w) * normalized_drift)
        
        # Entry momentum check: score must exceed threshold, and price z_score must be positive
        is_uptrend = (momentum_score > self.threshold) and (z_score >= 0.5)
        
        if is_moving and is_uptrend:
            return float(min(momentum_score * self.sensitivity, 0.005))
            
        return 0.0
