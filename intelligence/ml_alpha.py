import os
import logging
import numpy as np
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class LightGBMAlpha(BaseAlphaStrategy):
    """
    Tabular Machine Learning strategy.
    Loads pre-trained LightGBM booster parameters from file and routes 
    live feature vectors to output expected excess return forecasts.
    """
    def __init__(self, model_path: str = "weights.lgb"):
        self.model_path = model_path
        self.booster = None
        self.load_model()

    def load_model(self):
        # Check if model weights exist in local folder
        if os.path.exists(self.model_path):
            try:
                import lightgbm as lgb
                self.booster = lgb.Booster(model_file=self.model_path)
                logger.info(f"LightGBM booster successfully loaded from {self.model_path}")
            except (ImportError, OSError) as e:
                logger.warning(f"lightgbm library failed to load: {e}. Model forecast will revert to neutral. (On macOS, you might need 'brew install libomp' to load LightGBM).")
            except Exception as e:
                logger.error(f"Failed to load LightGBM booster: {e}")
        else:
            logger.warning(f"Trained weights file {self.model_path} not found. ML forecasting deactivated.")

    def predict(self, features: np.ndarray) -> float:
        """
        Feeds tabular feature vector to loaded booster model.
        """
        if self.booster is None:
            return 0.0

        try:
            # LightGBM predicts over matrix rows. Wrap single vector in list.
            forecast = self.booster.predict([features])[0]
            return float(forecast)
        except Exception as e:
            logger.error(f"LightGBM prediction exception: {e}")
            return 0.0
