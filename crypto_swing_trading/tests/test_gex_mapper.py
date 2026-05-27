import sys
import os
import pytest
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_alpha_macro")))
from gex_mapper import GexMapper

def test_gex_mapping_walls():
    """Verify that the GexMapper correctly identifies Support Put Walls, Squeeze Call Walls, and Zero Gamma."""
    mapper = GexMapper(model_type="COIN_MARGINED")
    
    spot_price = 60000.0
    strikes = np.linspace(57000.0, 63000.0, 7) # [57k, 58k, 59k, 60k, 61k, 62k, 63k]
    
    # Scenario Setup:
    # Dealers are net short on Call Walls (yielding negative GEX)
    # Dealers are net short on Put Walls (yielding positive/negative GEX depending on balance)
    # Let's set up explicit Open Interests:
    # Strike 58000: High Put OI, low Call OI (Put Wall downside support)
    # Strike 62000: High Call OI, low Put OI (Call Wall overhead resistance)
    # Strike 60000: Balanced or transitioning
    
    call_oi = np.array([100.0, 100.0, 200.0, 1000.0, 200.0, 5000.0, 100.0])  # Call OI peaks at index 5 (62000.0)
    put_oi = np.array([100.0, 5000.0, 200.0, 1000.0, 200.0, 100.0, 100.0])   # Put OI peaks at index 1 (58000.0)
    
    sigmas = np.full_like(strikes, 0.40)
    t = 7.0 / 365.0
    r = 0.05
    q = 0.00
    
    # Set Dealer positioning: dealer is Net Short Call at 62000 (-1.0 multiplier),
    # and Net Long Put at 58000 (+1.0 multiplier).
    # Let's pass simple multipliers:
    dealer_multipliers = np.array([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    
    gex_profile = mapper.calculate_gex_profile(
        spot_price=spot_price,
        strikes=strikes,
        call_oi=call_oi,
        put_oi=put_oi,
        sigmas=sigmas,
        t=t,
        r=r,
        q=q,
        dealer_multipliers=dealer_multipliers
    )
    
    # Run structural hedging mapping
    mapping = mapper.map_structural_hedging(gex_profile)
    
    # Verify mapping output structure
    assert "zero_gamma" in mapping
    assert "support_walls" in mapping
    assert "squeeze_walls" in mapping
    
    # Verify top support wall is strike 58000 (since dealer is long puts there, yielding high positive GEX)
    assert mapping["support_walls"][0][0] == 58000.0
    
    # Verify top squeeze wall is strike 62000 (since dealer is short calls there, yielding high negative GEX)
    assert mapping["squeeze_walls"][0][0] == 62000.0
    
    # Verify Zero Gamma exists
    assert mapping["zero_gamma"] is not None
