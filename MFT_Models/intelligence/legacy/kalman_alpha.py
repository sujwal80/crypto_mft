import numpy as np
import logging
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class KalmanFilterAlpha(BaseAlphaStrategy):
    """
    Recursive State-Space Kalman Filter quantitative alpha strategy.
    Filters L2 microstructural noise to track the "true fair price" hidden state
    and generates mean-reversion signals when observations deviate from the filter.
    """
    def __init__(
        self, 
        process_noise: float = 0.01,      # Q: variance of true price movements
        measure_noise: float = 25.0,      # R: variance of bid-ask queue noise
        threshold: float = 0.0008,        # 8 bps divergence trigger
        sensitivity: float = 0.05
    ):
        self.Q = process_noise
        self.R = measure_noise
        self.threshold = threshold
        self.sensitivity = sensitivity
        
        # State variables
        self.x = None                     # Filtered fair price state estimate
        self.P = 1.0                      # Filter error covariance
        
    def predict(self, features: np.ndarray) -> float:
        """
        Predicts expected return forecast using Kalman Filter state estimation.
        
        Args:
            features: [z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price]
        """
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features
        
        # 1. Initialize filter state on first seen price tick
        if self.x is None:
            self.x = mid_price
            self.P = 1.0
            return 0.0
            
        # 2. Kalman Predict Phase
        # State transition is identity (random walk with no systemic drift)
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # 3. Kalman Update Phase
        # Measurement model is identity (observed mid price is fair price + noise)
        measurement_residual = mid_price - x_pred
        S = P_pred + self.R               # Innovation covariance
        K = P_pred / S                    # Optimal Kalman Gain
        
        self.x = x_pred + K * measurement_residual
        self.P = (1.0 - K) * P_pred
        
        # 4. Signal Generation (Regime-Switching Kalman Filter)
        # Calculate percentage divergence from the clean filtered fair price estimate
        divergence = (mid_price - self.x) / self.x
        
        # Dynamic trend momentum detection (using micro-price drift)
        # If micro-price drift is strong, we are in a momentum breakout regime.
        # Otherwise, we are in a mean-reversion/ranging regime.
        is_trending = abs(micro_price_drift) >= 0.15
        
        alpha = 0.0
        if is_trending:
            # MOMENTUM REGIME: Trade in the direction of the breakout
            if divergence > self.threshold:
                alpha = self.sensitivity * (divergence / self.threshold)
                logger.debug(f"⚡ [KALMAN-MOM-BUY] Trend Breakout Upwards! Mid: ${mid_price:.2f} | Fair: ${self.x:.2f} | Signal: {alpha:.5f}")
            elif divergence < -self.threshold:
                alpha = -self.sensitivity * (abs(divergence) / self.threshold)
                logger.debug(f"⚡ [KALMAN-MOM-SELL] Trend Breakdown Downwards! Mid: ${mid_price:.2f} | Fair: ${self.x:.2f} | Signal: {alpha:.5f}")
        else:
            # MEAN REVERSION REGIME: Trade the snapback against the divergence
            if divergence > self.threshold:
                alpha = -self.sensitivity * (divergence / self.threshold)
                logger.debug(f"🛡️ [KALMAN-MR-SHORT] Mean Reversion Rejection Overhead! Mid: ${mid_price:.2f} | Fair: ${self.x:.2f} | Signal: {alpha:.5f}")
            elif divergence < -self.threshold:
                alpha = self.sensitivity * (abs(divergence) / self.threshold)
                logger.debug(f"🛡️ [KALMAN-MR-LONG] Mean Reversion Support Below! Mid: ${mid_price:.2f} | Fair: ${self.x:.2f} | Signal: {alpha:.5f}")
            
        # Cap signal to preserve risk sizing bounds
        return float(np.clip(alpha, -0.005, 0.005))
