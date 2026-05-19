import logging
import numpy as np
from typing import Dict, Optional

from intelligence.strategy_factory import AlphaStrategyFactory

logger = logging.getLogger(__name__)

class AlphaModel:
    """
    Dynamic model orchestrator routing feature vectors to modular 
    predictive strategy implementations using the Strategy Factory Pattern.
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

        # Instantiates the active strategy dynamically using the Factory Registry
        self.active_strategy = AlphaStrategyFactory.create_strategy(
            self.alpha_type, 
            model_path=self.model_path
        )

    def predict(self, features: np.ndarray) -> float:
        """
        Delegates feature vector directly to active strategy prediction module.

        Args:
            features: 6-dimension float array computed by FeatureStore.

        Returns:
            float: Expected return forecast.
        """
        return self.active_strategy.predict(features)
