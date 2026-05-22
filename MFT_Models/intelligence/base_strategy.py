from abc import ABC, abstractmethod
import numpy as np

class BaseAlphaStrategy(ABC):
    """
    Abstract Base Class defining the strategy contract for all quantitative alpha engines.
    
    Enforces a standardized input interface across ML and mathematical strategies to support 
    modular Strategy Factory instantiation.
    """
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """
        Calculates the expected return or directional trading prediction.
        
        Args:
            features: 6-dimension float array computed by FeatureStore containing:
                      [z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price]
                      
        Returns:
            float: Estimated alpha forecast (positive for buy, negative for sell, 0.0 for flat).
        """
        pass
