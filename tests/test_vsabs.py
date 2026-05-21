import pytest
import numpy as np
from core.schemas import InternalTick
from perception.feature_store import FeatureStore
from intelligence.strategy_factory import AlphaStrategyFactory
from intelligence.vsabs_alpha import VSABSAlpha

def test_vsabs_factory_creation():
    """Verify VSABSAlpha can be instantiated by the strategy factory."""
    strategy = AlphaStrategyFactory.create_strategy(
        alpha_type="VSABS",
        fee_rate=0.0010,
        fvr_multiplier=2.5,
        vol_ceiling=0.0150
    )
    assert isinstance(strategy, VSABSAlpha)
    assert strategy.fee_rate == 0.0010
    assert strategy.vol_floor == 0.0050

def test_vsabs_vfr_hibernation():
    """Verify that the strategy hibernates (returns 0) when volatility is below the floor."""
    strategy = VSABSAlpha(
        fee_rate=0.0010,       # Vol floor is 2.5 * (2 * 0.0010) = 0.0050 (50 bps)
        fvr_multiplier=2.5,
        vol_ceiling=0.0150,
        momentum_threshold=0.35,
        zscore_threshold=0.5
    )
    
    # 6 features: [z_score, spread, rolling_imbalance, micro_price_drift, rolling_vol, mid_price]
    # Low volatility: rolling_vol = 0.1 on 100.0 price => 10 bps (below the 50 bps floor)
    # Strong signal setup that would otherwise trigger a buy
    features = np.array([1.0, 0.02, 0.90, 0.10, 0.10, 100.0])
    
    forecast = strategy.predict(features)
    assert forecast == 0.0  # Must hibernate

def test_vsabs_volatility_ceiling():
    """Verify that the strategy suspends trading when volatility is above the ceiling."""
    strategy = VSABSAlpha(
        fee_rate=0.0010,
        fvr_multiplier=2.5,
        vol_ceiling=0.0150,    # Vol ceiling is 150 bps
        momentum_threshold=0.35,
        zscore_threshold=0.5
    )
    
    # Volatility is too high: rolling_vol = 2.0 on 100.0 price => 200 bps (above the 150 bps ceiling)
    features = np.array([1.0, 0.02, 0.90, 0.10, 2.00, 100.0])
    
    forecast = strategy.predict(features)
    assert forecast == 0.0  # Must suspend/panic close

def test_vsabs_bullish_breakout():
    """Verify that the strategy triggers BUY signals during a bullish momentum breakout."""
    strategy = VSABSAlpha(
        fee_rate=0.0010,       # Vol floor is 50 bps
        fvr_multiplier=2.5,
        vol_ceiling=0.0150,
        momentum_threshold=0.35,
        zscore_threshold=0.5,
        forecast_cap=0.005,
        forecast_multiplier=0.01
    )
    
    # Volatility is in the corridor: rolling_vol = 0.80 on 100.0 price => 80 bps
    # High buy imbalance and z-score above threshold
    features = np.array([0.6, 0.02, 0.90, 0.05, 0.80, 100.0])
    
    forecast = strategy.predict(features)
    assert forecast > 0.0
    assert forecast == min(forecast, 0.005)

def test_vsabs_bearish_breakout():
    """Verify that the strategy triggers SELL signals during a bearish momentum breakdown."""
    strategy = VSABSAlpha(
        fee_rate=0.0010,       # Vol floor is 50 bps
        fvr_multiplier=2.5,
        vol_ceiling=0.0150,
        momentum_threshold=0.35,
        zscore_threshold=0.5,
        forecast_cap=0.005,
        forecast_multiplier=0.01
    )
    
    # Volatility is in the corridor: rolling_vol = 0.80 on 100.0 price => 80 bps
    # High sell imbalance and z-score below threshold
    features = np.array([-0.6, 0.02, -0.90, -0.05, 0.80, 100.0])
    
    forecast = strategy.predict(features)
    assert forecast < 0.0
    assert forecast == max(forecast, -0.005)
