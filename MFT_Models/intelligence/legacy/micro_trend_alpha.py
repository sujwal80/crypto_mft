import numpy as np
import logging
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class MicroTrendMomentumAlpha(BaseAlphaStrategy):
    """
    Micro-Price Order Flow Momentum (MP-OFM) quantitative alpha strategy.
    Refined with scale-invariant signals and a Fee-to-Volatility Ratio (FVR) filter.
    """
    def __init__(
        self, 
        threshold: float = 0.35, 
        sensitivity: float = 0.01, 
        w: float = 0.85,
        f_total: float = 0.0006,       # 6 bps round trip fee (0.02% maker, 0.04% taker)
        fvr_limit: float = 2.0,        # Hibernation threshold (fees are > 2x local volatility)
        max_vol_ratio: float = 0.015, 
        z_score_cap: float = 1.5
    ):
        """
        Initializes MicroTrendMomentumAlpha.

        Args:
            threshold: Combined momentum score trigger limit.
            sensitivity: Scaling factor for forecasting return.
            w: Imbalance smoothing weight (relative weight on rolling imbalance vs drift).
            f_total: Total estimated round trip fee percentage.
            fvr_limit: Fee-to-Volatility Ratio limit (stops trading when volatility is too low to cover fees).
            max_vol_ratio: Volatility circuit breaker threshold (rolling_vol / mid_price).
            z_score_cap: Maximum z-score limit to avoid buying at local overbought peaks.
        """
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.w = w
        self.f_total = f_total
        self.fvr_limit = fvr_limit
        self.max_vol_ratio = max_vol_ratio
        self.z_score_cap = z_score_cap

    def predict(self, features: np.ndarray) -> float:
        """
        Predicts directional micro-trend return expectation.

        Args:
            features: 6-dimension features array.

        Returns:
            float: Positive forecast (BUY), negative forecast (SELL), or 0.0 (FLAT).
        """
        z_score, spread, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        # 1. Volatility circuit breaker to isolate extreme market spikes
        vol_rel = rolling_vol / (mid_price + 1e-8)
        if vol_rel > self.max_vol_ratio:
            return 0.0

        # 2. Refinement 3: Fee-to-Volatility Ratio (FVR) Filter
        # If fees are too high relative to local volatility, hibernate to prevent fee-extermination
        if vol_rel > 0.0 and self.fvr_limit is not None:
            fvr = self.f_total / vol_rel
            if fvr > self.fvr_limit:
                # Hibernation Mode active
                return 0.0

        # 3. Refinement 1: Scale-Invariant Momentum Score
        # We normalize micro-price drift by the raw spread instead of the price scale
        if spread > 0.0:
            normalized_drift = micro_price_drift / spread  # Mathematically equivalent to Inst_Imbalance / 2
        else:
            normalized_drift = 0.0
        
        # Compute weighted scale-invariant momentum score
        momentum_score = (self.w * rolling_imbalance) + ((1.0 - self.w) * normalized_drift)

        # 4. Bounded Entry Triggering
        if momentum_score > self.threshold and z_score <= self.z_score_cap:
            # Predict micro-uptrend (Long)
            return min(momentum_score * self.sensitivity, 0.005)
        elif momentum_score < -self.threshold:
            # Predict micro-downtrend (Short)
            return max(momentum_score * self.sensitivity, -0.005)

        return 0.0
