import numpy as np
import logging
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class DirectionalMicroPriceAlpha(BaseAlphaStrategy):
    """
    Directional High-Frequency Strategy.
    Predicts price direction (UP/DOWN) based on immediate L2 micro-price drift.
    """
    def __init__(self):
        pass

    def predict(self, features: np.ndarray) -> float:
        """
        Predicts UP (+0.5%) or DOWN (-0.5%) based on micro-price drift.
        """
        # Unpack features
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        if micro_price_drift > 0.0:
            return 0.005  # Predict UP
        elif micro_price_drift < 0.0:
            return -0.005 # Predict DOWN

        return 0.0
