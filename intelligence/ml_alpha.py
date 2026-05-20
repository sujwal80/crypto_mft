import os
import logging
import numpy as np
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class LightGBMAlpha(BaseAlphaStrategy):
    """
    Tabular Machine Learning strategy.
    Loads pre-trained LightGBM booster parameters or fallback NumPy Ridge coefficients
    from file and routes live feature vectors to output expected excess return forecasts.
    """
    def __init__(self, model_path: str = "weights.lgb", numpy_path: str = "weights.npy"):
        self.model_path = model_path
        self.numpy_path = numpy_path
        self.booster = None
        self.numpy_weights = None
        self.load_model()

    def load_model(self):
        # 1. Try loading LightGBM booster
        if os.path.exists(self.model_path):
            try:
                import lightgbm as lgb
                self.booster = lgb.Booster(model_file=self.model_path)
                logger.info(f"LightGBM booster successfully loaded from {self.model_path}")
                return
            except (ImportError, OSError) as e:
                logger.warning(f"LightGBM library failed to load: {e}. Falling back to NumPy weights...")
            except Exception as e:
                logger.error(f"Failed to load LightGBM booster: {e}")

        # 2. Fallback: Try loading NumPy Ridge coefficients
        if os.path.exists(self.numpy_path):
            try:
                self.numpy_weights = np.load(self.numpy_path)
                logger.info(f"NumPy Ridge model successfully loaded from {self.numpy_path}")
            except Exception as e:
                logger.error(f"Failed to load NumPy Ridge weights: {e}")
        else:
            logger.warning(f"Trained weights file ({self.model_path} or {self.numpy_path}) not found. ML forecasting deactivated.")

    def predict(self, features: np.ndarray) -> float:
        """
        Feeds tabular feature vector to loaded booster model or NumPy Ridge coefficients,
        and scales predictions to align with expected return sizing.
        """
        forecast = 0.0
        
        # A. Try LightGBM prediction
        if self.booster is not None:
            try:
                # LightGBM predicts over matrix rows. Wrap single vector in list.
                forecast = self.booster.predict([features])[0]
            except Exception as e:
                logger.error(f"LightGBM prediction exception: {e}")

        # B. Try NumPy Ridge prediction
        elif self.numpy_weights is not None:
            try:
                # Prediction = w_1 * f_1 + ... + w_D * f_D + intercept
                forecast = np.dot(features, self.numpy_weights[:-1]) + self.numpy_weights[-1]
            except Exception as e:
                logger.error(f"NumPy Ridge prediction exception: {e}")

        # Scale raw log-returns (usually in the scale of 1 bp) by 20.0 to align with sizing expectations (10-50 bps)
        return float(forecast) * 20.0
