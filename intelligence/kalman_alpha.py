import logging
import numpy as np
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class KalmanFilterAlpha(BaseAlphaStrategy):
    """
    State-Space estimator that calculates clean 'fair price' states under noisy L2 order book observations.

    Operates recursively inside high-frequency loops. It predicts the next state, computes the error
    covariance, and updates its estimate using a dynamically calculated Kalman Gain when a new
    volume-weighted micro-price drift measurement arrives.
    """
    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        """
        Initializes KalmanFilterAlpha.

        Args:
            process_noise: Q matrix diagonal parameter representing variance of fair price state.
            measurement_noise: R matrix diagonal parameter representing volatility of order book noise.
        """
        self.x_est = 0.0  # Clean fair price state estimate
        self.p_est = 1.0  # Estimation error covariance
        self.q = process_noise
        self.r = measurement_noise
        self.initialized = False

    def predict(self, features: np.ndarray) -> float:
        """
        Updates state spaces and estimates convergence opportunities.

        Args:
            features: 6-dimension float array computed by FeatureStore.

        Returns:
            float: Alpha forecast return.
        """
        # Unpack unified features vector
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        if not self.initialized:
            self.x_est = mid_price
            self.initialized = True
            return 0.0

        # 1. Time Update (Prediction)
        x_pred = self.x_est
        p_pred = self.p_est + self.q

        # 2. Measurement Update (Correction)
        # Clean fair price observation uses microstructural volume drifts
        observation = mid_price + micro_price_drift
        kalman_gain = p_pred / (p_pred + self.r + 1e-8)
        self.x_est = x_pred + kalman_gain * (observation - x_pred)
        self.p_est = (1 - kalman_gain) * p_pred

        # Calculate deviation percentage of market price from estimated fair price
        fair_deviation = (self.x_est - mid_price) / mid_price

        if fair_deviation > 0.0015:
            return 0.004  # Underpriced -> Buy signal
        elif fair_deviation < -0.0015:
            return -0.004  # Overpriced -> Sell signal

        return 0.0
