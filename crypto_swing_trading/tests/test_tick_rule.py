import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_alpha_macro")))
from tick_rule import OptionsTickRule

def test_tick_rule_classification():
    """Verify trade classification under classical Tick Rule and zero-uptick/zero-downtick logic."""
    rule = OptionsTickRule()
    
    # First trade: assumption is 0 (unclassifiable)
    direction = rule.classify_trade(60000.0, 0.05, 10.0)
    assert direction == 0
    assert rule.dealer_positions[60000.0] == 0.0
    
    # Uptick: Retail Taker Buy (+1) -> Dealer short (-10.0)
    direction = rule.classify_trade(60000.0, 0.06, 10.0)
    assert direction == 1
    assert rule.dealer_positions[60000.0] == -10.0
    
    # Zero uptick: Unchanged price -> Same direction (+1) -> Dealer short (-10.0)
    direction = rule.classify_trade(60000.0, 0.06, 15.0)
    assert direction == 1
    assert rule.dealer_positions[60000.0] == -25.0
    
    # Downtick: Retail Taker Sell (-1) -> Dealer long (+20.0)
    direction = rule.classify_trade(60000.0, 0.04, 20.0)
    assert direction == -1
    assert rule.dealer_positions[60000.0] == -5.0
    
    # Zero downtick: Unchanged price -> Same direction (-1) -> Dealer long (+10.0)
    direction = rule.classify_trade(60000.0, 0.04, 10.0)
    assert direction == -1
    assert rule.dealer_positions[60000.0] == 5.0

def test_dealer_multipliers_and_decay():
    """Verify GEX integration multipliers mapping and exponential decay logic."""
    rule = OptionsTickRule()
    
    # Simulate trade history
    trades = [
        {"strike": 59000.0, "price": 0.02, "amount": 5.0, "direction": "buy"},   # Retail Buy -> Dealer Short
        {"strike": 60000.0, "price": 0.04, "amount": 10.0, "direction": "sell"}, # Retail Sell -> Dealer Long
    ]
    rule.process_trades_batch(trades)
    
    assert rule.dealer_positions[59000.0] == -5.0
    assert rule.dealer_positions[60000.0] == 10.0
    
    strikes = np.array([59000.0, 60000.0, 61000.0])
    multipliers = rule.get_dealer_multipliers(strikes)
    
    # Dealer is Short at 59000 (-1.0), Long at 60000 (+1.0), Long/Neutral at 61000 (+1.0)
    np.testing.assert_allclose(multipliers, np.array([-1.0, 1.0, 1.0]))
    
    # Test Decay
    rule.decay_positions(0.5)
    assert rule.dealer_positions[59000.0] == -2.5
    assert rule.dealer_positions[60000.0] == 5.0
