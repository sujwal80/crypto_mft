import numpy as np
from intelligence.base_strategy import BaseAlphaStrategy

class StatisticalHeuristicAlpha(BaseAlphaStrategy):
    """
    Default rule-based statistical heuristic fallback strategy.
    Uses simple thresholds on Z-Score, OBI, and micro-price drift to trigger signals.
    """
    def __init__(self):
        pass

    def predict(self, features: np.ndarray) -> float:
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        # Volatility circuit breaker for extreme environments
        if rolling_vol / mid_price > 0.02:
            return 0.0

        is_trending = abs(z_score) > 2.0
        if is_trending:
            if z_score > 2.0 and micro_price_drift > 0.05 and rolling_imbalance > 0.3:
                return 0.004
            elif z_score < -2.0 and micro_price_drift < -0.05 and rolling_imbalance < -0.3:
                return -0.004
        else:
            if z_score < -1.0 and rolling_imbalance > 0.4 and micro_price_drift > 0.02:
                return 0.002
            elif z_score > 1.0 and rolling_imbalance < -0.4 and micro_price_drift < -0.02:
                return -0.002

        return 0.0
