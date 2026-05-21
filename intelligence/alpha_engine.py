import logging
import numpy as np
from typing import Dict, Optional

from intelligence.strategy_factory import AlphaStrategyFactory
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class FVRAlphaWrapper(BaseAlphaStrategy):
    """
    Dynamic Decorator/Wrapper that adds Fee-to-Volatility Ratio (FVR) 
    and hibernation circuit breakers to any quantitative Alpha strategy.
    """
    def __init__(self, wrapped_strategy: BaseAlphaStrategy, f_total: float = 0.0006, fvr_limit: float = 2.0):
        self.wrapped = wrapped_strategy
        self.f_total = f_total
        self.fvr_limit = fvr_limit

    def predict(self, features: np.ndarray) -> float:
        rolling_vol = features[4]
        mid_price = features[5]
        
        # Relative local volatility
        vol_rel = rolling_vol / (mid_price + 1e-8)
        
        # FVR Hibernation circuit breaker
        if vol_rel > 0.0 and self.fvr_limit is not None:
            fvr = self.f_total / vol_rel
            if fvr > self.fvr_limit:
                # Hibernate to prevent fee-extermination
                return 0.0
                
        return self.wrapped.predict(features)

class AlphaModel:
    """
    Dynamic model orchestrator routing feature vectors to modular 
    predictive strategy implementations using the Strategy Factory Pattern.
    """
    def __init__(self, model_path: str = "weights.lgb", alpha_type: str = "KALMAN", enable_fvr: bool = True, **kwargs):
        """
        Initializes AlphaModel orchestrator.

        Args:
            model_path: Absolute path to LightGBM binary weights.
            alpha_type: Dynamic model selector string ("ML", "HYBRID", "KALMAN").
            enable_fvr: Toggle FVR fee protection wrapper.
            **kwargs: Additional strategy configurations passed to constructor.
        """
        self.model_path = model_path
        self.alpha_type = alpha_type.upper()
        
        # Set dynamic expected return threshold filter limit (default to 0.0)
        self.min_return_threshold = kwargs.get("min_return_threshold", 0.0)

        # Instantiates the active strategy dynamically using the Factory Registry
        self.active_strategy = AlphaStrategyFactory.create_strategy(
            self.alpha_type, 
            model_path=self.model_path,
            **kwargs
        )

        # Automatically wrap active strategy in FVR fee protection wrapper to prevent transaction fee-extermination
        if enable_fvr:
            f_total = kwargs.get("f_total", 0.0006)
            fvr_limit = kwargs.get("fvr_limit", 2.0)
            self.active_strategy = FVRAlphaWrapper(self.active_strategy, f_total=f_total, fvr_limit=fvr_limit)

    def predict(self, features: np.ndarray) -> float:
        """
        Delegates feature vector directly to active strategy prediction module,
        applying a unified expected return threshold filter.

        Args:
            features: 6-dimension float array computed by FeatureStore.

        Returns:
            float: Expected return forecast.
        """
        forecast = self.active_strategy.predict(features)
        
        # Clamp predicted forecast to 0.0 if it is below the minimum return threshold
        if abs(forecast) < self.min_return_threshold:
            return 0.0
            
        return forecast

