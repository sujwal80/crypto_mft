import pytest
import numpy as np
from intelligence.legacy.gex_oi_alpha import GEXAlphaStrategy
from intelligence.legacy.gex_oi_engine import GexOiAlphaEngine

def test_gex_mathematical_equivalence():
    """
    Asserts that the newly vectorized GexOiAlphaEngine produces identical 
    Gamma and GEX values compared to the legacy loop-based GEXAlphaStrategy.
    """
    spot_price = 60000.0
    rolling_vol = 0.02 * spot_price # 2% volatility

    # 1. Instantiate both engines
    legacy_strategy = GEXAlphaStrategy(expiry_days=7.0, risk_free_rate=0.05)
    vector_engine = GexOiAlphaEngine(expiry_days=7.0, risk_free_rate=0.05)

    # 2. Initialize options chain
    legacy_strategy._initialize_options_chain(spot_price, rolling_vol)
    vector_engine._initialize_options_chain(spot_price, rolling_vol)

    # Ensure strikes are identical
    assert list(legacy_strategy.options_chain.keys()) == list(vector_engine.options_chain.keys())

    # 3. Calculate profiles
    legacy_profile = legacy_strategy.calculate_gex_profile(spot_price)
    vector_profile = vector_engine.calculate_gex_profile(spot_price)

    # 4. Assert identical output GEX values at all strike levels (with high precision)
    for strike in legacy_profile:
        val_legacy = legacy_profile[strike]
        val_vector = vector_profile[strike]
        np.testing.assert_allclose(val_legacy, val_vector, rtol=1e-9, atol=1e-9)

def test_gex_prediction_equivalence():
    """
    Asserts that the prediction output of the vectorized GexOiAlphaEngine 
    matches the legacy strategy under identical market features.
    """
    # Define features near Call Wall: base 60000, call wall at +0.4% (60240)
    # Proximity: 0.3%, so 60234 is near.
    mid_price_near_wall = 60234.0
    features_short = np.array([1.0, 1.0, -0.5, 0.0, 0.02 * 60000.0, mid_price_near_wall])

    legacy_strategy = GEXAlphaStrategy(expiry_days=7.0, risk_free_rate=0.05, ofi_threshold=0.3, sensitivity=0.0015)
    vector_engine = GexOiAlphaEngine(expiry_days=7.0, risk_free_rate=0.05, ofi_threshold=0.3, sensitivity=0.0015)

    # Warm up options chains
    legacy_strategy.predict(features_short)
    vector_engine.predict(features_short)

    # Reset cache to force immediate calculation for testing
    vector_engine.cached_gex_profile = None

    pred_legacy = legacy_strategy.predict(features_short)
    pred_vector = vector_engine.predict(features_short)

    assert pred_legacy == pred_vector
    assert pred_legacy < 0.0 # Verify it predicts short reversal
