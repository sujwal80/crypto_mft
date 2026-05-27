import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../3_alpha_micro")))
from cvd_engine import CumulativeVolumeDeltaEngine

def test_cvd_engine_tracking():
    """Verify CVD engine correctly accumulates taker buy and sell volumes."""
    engine = CumulativeVolumeDeltaEngine(rolling_window_ticks=5)
    
    # Taker Buy: buyer is NOT maker (is_buyer_maker=False)
    engine.process_trade(60000.0, 10.0, False)
    # Taker Sell: buyer IS maker (is_buyer_maker=True)
    engine.process_trade(60000.0, 4.0, True)
    engine.process_trade(60000.0, 3.0, True)
    
    assert engine.rolling_cvd == 3.0 # 10 - 4 - 3 = 3
    assert engine.total_buy_volume == 10.0
    assert engine.total_sell_volume == 7.0
    assert engine.get_aggression_ratio() == 10.0 / 7.0
    
    # Eviction test
    engine.process_trade(60000.0, 2.0, False) # +2
    engine.process_trade(60000.0, 5.0, False) # +5
    # Window is now full (size 5) with: [10, -4, -3, 2, 5]
    # Add 6th trade to trigger eviction of oldest (10)
    engine.process_trade(60000.0, 1.0, True) # -1
    
    # The 10 should be popped, active queue is: [-4, -3, 2, 5, -1]
    # Sum = -4 - 3 + 2 + 5 - 1 = -1
    assert engine.rolling_cvd == -1.0
    assert engine.total_buy_volume == 7.0  # 2 + 5 = 7
    assert engine.total_sell_volume == 8.0 # 4 + 3 + 1 = 8

def test_absorption_detection():
    """Verify absorption divergence detection on massive taker sweeps with flat prices."""
    engine = CumulativeVolumeDeltaEngine(rolling_window_ticks=100)
    
    # Simulate heavy taker selling (buyer is maker)
    for _ in range(50):
        engine.process_trade(60000.0, 10.0, True)
        
    # CVD is extremely negative, but price change is 0 (price remained exactly 60000.0)
    assert engine.detect_absorption_divergence(price_change_pct=0.0) == True
    
    # With normal price drop, it should not flag as absorption (pro-trend move)
    assert engine.detect_absorption_divergence(price_change_pct=-0.02) == False
