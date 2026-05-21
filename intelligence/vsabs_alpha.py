import numpy as np
import logging
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class VSABSAlpha(BaseAlphaStrategy):
    """
    Volatility-Scale Adaptive Breakout Strategy (VSABS) Alpha Model.
    Combines spread-normalized micro-momentum and rolling spot Z-score confirmations,
    safeguarded by dynamic Fee-Gating and high-volatility panic circuit breakers.
    Adapted to run on the production 6-element FeatureStore vector.
    """
    def __init__(
        self,
        fee_rate: float = 0.0010,
        fvr_multiplier: float = 2.5,
        vol_ceiling: float = 0.0150,
        momentum_threshold: float = 0.35,
        zscore_threshold: float = 0.5,
        forecast_cap: float = 0.005,
        forecast_multiplier: float = 0.01,
        w_ofi: float = 0.85,
        threshold: float = None,
        **kwargs
    ):
        if threshold is not None:
            momentum_threshold = threshold
        self.fee_rate = fee_rate
        self.fvr_multiplier = fvr_multiplier
        self.vol_ceiling = vol_ceiling
        
        self.momentum_threshold = momentum_threshold
        self.zscore_threshold = zscore_threshold
        self.forecast_cap = forecast_cap
        self.forecast_multiplier = forecast_multiplier
        self.w_ofi = w_ofi
        
        # Calculated dynamic hibernation floor (e.g., 50 bps for 0.1% fee_rate with 2.5x multiplier)
        self.vol_floor = self.fvr_multiplier * (2.0 * self.fee_rate)
        
        logger.info(
            f"VSABSAlpha initialized: VFR floor = {self.vol_floor * 10000:.1f} bps | "
            f"Vol Ceiling = {self.vol_ceiling * 10000:.1f} bps | "
            f"Momentum Threshold = {self.momentum_threshold:.2f}"
        )

    def predict(self, features: np.ndarray) -> float:
        """
        Calculates the expected return or directional trading prediction.
        
        Args:
            features: 6-dimension float array computed by FeatureStore containing:
                      [z_score, spread, rolling_imbalance, micro_price_drift, rolling_vol, mid_price]
                      
        Returns:
            float: Expected directional return forecast, or 0.0 if hibernating/circuit-broken.
        """
        z_score = features[0]
        spread = features[1]
        rolling_imbalance = features[2]
        micro_price_drift = features[3]
        rolling_vol = features[4]
        mid_price = features[5]

        # 1. Scale-invariant relative local volatility
        relative_vol = rolling_vol / (mid_price + 1e-8)

        # 2. Volatility Floor (VFR Gating) Hibernation circuit breaker
        if relative_vol < self.vol_floor:
            # Relative volatility too low to cover round-trip fees; hibernate
            return 0.0

        # 3. Volatility Ceiling (Panic Circuit Breaker)
        if relative_vol > self.vol_ceiling:
            # Relative volatility too high (toxic/slippage regime); suspend
            return 0.0

        # 4. Spread-normalized micro-momentum score
        normalized_drift = micro_price_drift / (spread + 1e-8)
        momentum_score = (self.w_ofi * rolling_imbalance) + ((1.0 - self.w_ofi) * normalized_drift)

        # 5. Breakout entry heuristics
        is_bullish_breakout = (momentum_score > self.momentum_threshold) and (z_score >= self.zscore_threshold)
        is_bearish_breakout = (momentum_score < -self.momentum_threshold) and (z_score <= -self.zscore_threshold)

        if is_bullish_breakout:
            # Return calibrated positive forecast
            return float(min(momentum_score * self.forecast_multiplier, self.forecast_cap))
            
        if is_bearish_breakout:
            # Return calibrated negative forecast
            return float(max(momentum_score * self.forecast_multiplier, -self.forecast_cap))

        # Ranging regime
        return 0.0
