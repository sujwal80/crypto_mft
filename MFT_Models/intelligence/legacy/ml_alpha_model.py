import os
import logging
import numpy as np
from intelligence.legacy.ml_alpha import LightGBMAlpha

logger = logging.getLogger(__name__)

class MLAlphaModel(LightGBMAlpha):
    """
    Stationary Machine Learning Engine (ML).
    Subclasses LightGBMAlpha to guarantee 100% backwards compatibility with test cases.
    Ingests LOB parameters, constructs a 6-feature stationary vector
    (synthesizing momentum_score on-the-fly), and predicts forward returns.
    Employs an automated fallback to a vectorized Ridge Regression estimator
    if LightGBM booster weights or libraries are missing.
    """
    def __init__(self, model_path: str = "weights.lgb", numpy_path: str = "weights.npy"):
        super().__init__(model_path=model_path, numpy_path=numpy_path)

    def predict(self, features: np.ndarray) -> float:
        """
        Feeds the stationary 6-feature vector into the loaded prediction model.
        Synthesizes momentum_score = 0.85 * rolling_imbalance + 0.15 * normalized_drift.
        """
        z_score, spread, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features
        
        # Calculate normalized_drift and momentum_score dynamically
        normalized_drift = micro_price_drift / spread if spread > 0.0 else 0.0
        momentum_score = (0.85 * rolling_imbalance) + (0.15 * normalized_drift)
        
        # Construct features_stationary (6 elements)
        features_stationary = np.array([z_score, spread, rolling_imbalance, micro_price_drift, rolling_vol, momentum_score])
        
        forecast = 0.0
        
        # A. Try LightGBM prediction
        if self.booster is not None:
            try:
                forecast = self.booster.predict([features_stationary])[0]
            except Exception as e:
                logger.error(f"LightGBM prediction exception: {e}")

        # B. Try NumPy Ridge prediction
        elif self.numpy_weights is not None:
            try:
                # Prediction = w_1 * f_1 + ... + w_D * f_D + intercept (last element of weight vector)
                forecast = np.dot(features_stationary, self.numpy_weights[:-1]) + self.numpy_weights[-1]
            except Exception as e:
                logger.error(f"NumPy Ridge prediction exception: {e}")

        # Scale raw log-returns (typically ~1 bp) by 20.0 to align with sizing expectations
        return float(forecast) * 20.0
