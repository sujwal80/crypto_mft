import logging
import numpy as np
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class OrderFlowImbalanceAlpha(BaseAlphaStrategy):
    """
    Order Flow Imbalance (OFI) alpha engine.

    Measures immediate buyer/seller momentum by tracking tick-by-tick accumulation changes
    in Limit Order Book (LOB) depth sizes.
    """
    def __init__(self, sensitivity: float = 0.5, threshold: float = 0.35):
        """
        Initializes OrderFlowImbalanceAlpha.

        Args:
            sensitivity: Scaling factor applied to volume momentum forecasts.
            threshold: Cutoff boundary filter to prevent trading in quiet ranges.
        """
        self.sensitivity = sensitivity
        self.threshold = threshold

    def predict(self, features: np.ndarray) -> float:
        """
        Evaluates microstructural volume flows to forecast directional breakouts.

        Args:
            features: 6-dimension float array computed by FeatureStore.

        Returns:
            float: Alpha return forecast.
        """
        # Unpack unified features vector
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        # Combine LOB imbalance and micro-price drift
        flow_momentum = (rolling_imbalance * 0.6) + ((micro_price_drift * 10.0) * 0.4)

        if flow_momentum > self.threshold:
            return min(flow_momentum * self.sensitivity * 0.01, 0.005)
        elif flow_momentum < -self.threshold:
            return max(flow_momentum * self.sensitivity * 0.01, -0.005)

        return 0.0
