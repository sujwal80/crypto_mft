import numpy as np
import logging

logger = logging.getLogger(__name__)

class MicroPriceKalmanFilter:
    """
    State-Space Noise Filter for High-Frequency Micro-Price estimation.
    Filters out microstructure noise (bid-ask bounces, order book spoofing) 
    to track the true latent fair value of the asset.
    """
    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        """
        Args:
            Q: Process noise variance (how fast the true price changes)
            R: Measurement noise variance (how noisy the observations are)
        """
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        
        # Kalman Filter State variables
        self.x = None # Estimated true price (latent state)
        self.p = 1.0  # Estimation error covariance

    def reset(self, initial_price: float):
        """Resets the Kalman Filter state to the initial price."""
        self.x = float(initial_price)
        self.p = 1.0
        
    def filter_tick(self, observed_price: float) -> float:
        """
        Runs a single predict-update cycle of the Kalman Filter on a new price tick.
        
        Args:
            observed_price: The noisy mid-price or micro-price from the exchange.
            
        Returns:
            float: The filtered fair value estimate.
        """
        observed_price = float(observed_price)
        
        if self.x is None:
            self.reset(observed_price)
            return self.x
            
        # 1. Predict Phase
        # x_pred = x_prev (random walk state model)
        # P_pred = P_prev + Q
        p_pred = self.p + self.q
        
        # 2. Update Phase
        # Innovation: y - x_pred
        innovation = observed_price - self.x
        # Innovation covariance: S = P_pred + R
        s = p_pred + self.r
        
        # Kalman Gain: K = P_pred / S
        k = p_pred / s
        
        # Updated state estimate: x = x_pred + K * innovation
        self.x = self.x + k * innovation
        # Updated covariance: P = (1 - K) * P_pred
        self.p = (1.0 - k) * p_pred
        
        return self.x

    def update_noise_parameters(self, new_q: float, new_r: float):
        """Allows dynamically updating Q and R based on rolling volatility/spreads."""
        self.q = float(new_q)
        self.r = float(new_r)
