import numpy as np
import logging

logger = logging.getLogger(__name__)

class OrnsteinUhlenbeckAlpha:
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

    def predict(self, z_score: float, rolling_vol: float, mid_price: float) -> float:
        """
        Predicts expected directional return using Ornstein-Uhlenbeck snapback boundaries.

        Args:
            z_score: Deviation of current price relative to rolling lookback window.
            rolling_vol: Standard deviation of prices over lookback window.
            mid_price: Current baseline mid-market price.

        Returns:
            float: Estimated expected return forecast (alpha forecast).
        """
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
