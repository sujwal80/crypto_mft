import numpy as np
import logging
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class OrnsteinUhlenbeckAlpha(BaseAlphaStrategy):
    """
    Simulates mean-reversion paths using Ornstein-Uhlenbeck continuous stochastic differential equation physics.

    The OU process assumes that the asset price behaves like a spring: the further it is stretched
    (high Z-Score), the stronger the force pulling it back to its historical mean.

    Formula:
        dx_t = theta * (mu - x_t) * dt + sigma * dW_t
    """
    def __init__(self, theta: float = 0.15, entry_multiplier: float = 1.5):
        """
        Initializes OrnsteinUhlenbeckAlpha.

        Args:
            theta: Reversion speed coefficient (pull strength).
            entry_multiplier: Scaling parameter used to define volatility boundaries.
        """
        self.theta = theta
        self.entry_multiplier = entry_multiplier

    def predict(self, features: np.ndarray) -> float:
        """
        Predicts expected directional return using Ornstein-Uhlenbeck snapback boundaries.

        Args:
            features: 6-dimension float array computed by FeatureStore.

        Returns:
            float: Estimated expected return forecast (alpha forecast).
        """
        # Unpack unified features vector
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        threshold = self.entry_multiplier * rolling_vol / mid_price

        # Volatility circuit breaker for extreme outliers
        if abs(z_score) > 2.5:
            return 0.0

        # If price has drifted lower, expect upward pulling force
        if z_score < -1.2:
            expected_pull = self.theta * (-z_score) * threshold
            return min(expected_pull, 0.006)
        # If price has drifted higher, expect downward pulling force
        elif z_score > 1.2:
            expected_pull = self.theta * (-z_score) * threshold
            return max(expected_pull, -0.006)

        return 0.0
