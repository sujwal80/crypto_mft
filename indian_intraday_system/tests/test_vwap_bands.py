import sys
import os
import pytest
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../layer_2_macro")))
from vwap_bands import IntradayVwapBands

def test_vwap_bands_basic_calculations():
    """Verify that running VWAP and Z-score update correctly under steady inputs."""
    calc = IntradayVwapBands()
    
    # Feed steady prices: Price = 100, Vol = 10
    res = calc.update(price=100.0, volume=10.0)
    
    assert res is not None
    assert res["vwap"] == 100.0
    assert res["sigma"] == 1e-4  # Dynamic floor bounds
    assert res["z_score"] == 0.0  # At the mean
    
    # Feed deviation: Price = 102, Vol = 10
    res = calc.update(price=102.0, volume=10.0)
    
    # VWAP = (100*10 + 102*10) / 20 = 101.0
    assert res["vwap"] == 101.0
    # Variance = E[X^2] - (E[X])^2 = (10000*10 + 10404*10)/20 - 10201 = 10202 - 10201 = 1.0
    # Sigma = sqrt(1.0) = 1.0
    assert abs(res["sigma"] - 1.0) < 1e-5
    # Z-score of 102 relative to Mean=101, Std=1 is exactly +1.0
    assert abs(res["z_score"] - 1.0) < 1e-5

def test_vwap_bands_session_reset():
    """Verify that daily session resets clear all memory back to zero."""
    calc = IntradayVwapBands()
    
    # Ingest some session price data
    calc.update(price=150.0, volume=100.0)
    calc.update(price=155.0, volume=200.0)
    
    # Verify memory is populated
    assert calc.ticks_count == 2
    assert calc.sum_v > 0.0
    
    # Trigger daily session reset (e.g., at 9:15 AM IST)
    calc.reset_session()
    
    # Verify all memory blocks are cleared cleanly
    assert calc.ticks_count == 0
    assert calc.sum_v == 0.0
    assert calc.sum_pv == 0.0
    assert calc.vwap == 0.0
