import logging
from typing import Dict, Type
from intelligence.base_strategy import BaseAlphaStrategy
from intelligence.ml_alpha import LightGBMAlpha
from intelligence.hybrid_ml_gex_alpha import HybridMLGEXAlpha
from intelligence.kalman_alpha import KalmanFilterAlpha
from intelligence.vol_micro_trend_alpha import VolMicroTrendStrategy

logger = logging.getLogger(__name__)

class AlphaStrategyFactory:
    """
    Strategy Factory pattern registry.
    Dynamically instantiates and registers alpha strategy implementations.
    Excludes underperforming legacy mathematical engines.
    """
    
    # Registry map of active profitable model types to their classes
    _REGISTRY: Dict[str, Type[BaseAlphaStrategy]] = {
        "ML": LightGBMAlpha,
        "HYBRID": HybridMLGEXAlpha,
        "KALMAN": KalmanFilterAlpha,
        "VOL_MICRO_TREND": VolMicroTrendStrategy
    }

    @classmethod
    def create_strategy(cls, alpha_type: str, **kwargs) -> BaseAlphaStrategy:
        """
        Factory instantiation method.
        
        Args:
            alpha_type: Upper case strategy string identifier ("ML", "HYBRID", "KALMAN").
            **kwargs: Configuration parameters passed to strategy constructor.
            
        Returns:
            BaseAlphaStrategy: Strategy strategy instance.
        """
        import inspect
        lookup_key = alpha_type.upper()
        if lookup_key not in cls._REGISTRY:
            logger.error(f"Strategy type '{alpha_type}' not found in factory registry! Defaulting to KALMAN.")
            strategy_class = KalmanFilterAlpha
        else:
            strategy_class = cls._REGISTRY[lookup_key]
            
        logger.info(f"Factory successfully instantiated Strategy: [{strategy_class.__name__}]")
        
        # Filter kwargs to only match parameters expected by the strategy class constructor
        sig = inspect.signature(strategy_class.__init__)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        return strategy_class(**valid_kwargs)
