import os
import logging
import time
import numpy as np
from typing import Dict, Optional

# Import isolated math models
from intelligence.ou_alpha import OrnsteinUhlenbeckAlpha
from intelligence.kalman_alpha import KalmanFilterAlpha
from intelligence.ofi_alpha import OrderFlowImbalanceAlpha

logger = logging.getLogger(__name__)

class AlphaModel:
    """
    Dynamic model orchestrator routing feature vectors to specific LightGBM boosters
    or modular mathematical predictive algorithms (Kalman Filter, OU, OFI).
    """
    def __init__(self, model_path: str = "weights.lgb", alpha_type: str = "HEURISTIC"):
        """
        Initializes AlphaModel orchestrator.

        Args:
            model_path: Absolute path to LightGBM binary weights.
            alpha_type: Dynamic model selector string ("ML", "OU", "KALMAN", "OFI", "HEURISTIC").
        """
        self.model_path = model_path
        self.alpha_type = alpha_type.upper()
        self.booster = None

        # Instantiate modular mathematical engines
        self.ou_alpha = OrnsteinUhlenbeckAlpha()
        self.kalman_alpha = KalmanFilterAlpha()
        self.ofi_alpha = OrderFlowImbalanceAlpha()

        if self.alpha_type == "ML":
            self.load_model()
        else:
            logger.info(f"Alpha Engine loaded in mathematical/fallback mode: [{self.alpha_type}]")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                import lightgbm as lgb
                self.booster = lgb.Booster(model_file=self.model_path)
                logger.info(f"Successfully loaded LightGBM booster from {self.model_path}")
            except ImportError:
                logger.warning("LightGBM library not found. Falling back to statistical alpha.")
            except Exception as e:
                logger.error(f"Failed to load LightGBM model: {e}. Falling back to statistical alpha.")
        else:
            logger.warning(f"Model weights file {self.model_path} not found. Using default statistical heuristic.")

    def predict(self, features: np.ndarray) -> float:
        """
        Routes feature vector to selected prediction engine.

        Args:
            features: 6-dimension float array computed by FeatureStore.

        Returns:
            float: Forecast return prediction.
        """
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        if self.alpha_type == "ML" and self.booster:
            forecast = self.booster.predict([features])[0]
            return float(forecast)

        elif self.alpha_type == "OU":
            return self.ou_alpha.predict(z_score, rolling_vol, mid_price)

        elif self.alpha_type == "KALMAN":
            return self.kalman_alpha.predict(mid_price, micro_price_drift)

        elif self.alpha_type == "OFI":
            return self.ofi_alpha.predict(rolling_imbalance, micro_price_drift)

        else:
            # Default Heuristic Fallback
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
