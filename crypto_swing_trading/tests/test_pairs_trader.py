import sys
import os
import pytest
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../5_arbitrage")))
from pairs_trader import CointegratedPairsTrader

def test_pairs_trader_warmup():
    """Verify the engine returns None until the lookback window is fully warmed up."""
    trader = CointegratedPairsTrader(lookback_window=10)
    
    # Ingest only 9 price ticks
    for i in range(9):
        z = trader.ingest_prices(btc_price=60000.0 + i, eth_price=3000.0 + i)
        assert z is None

def test_pairs_trader_ols_regression():
    """Verify that NumPy OLS regression correctly fits beta (slope) and alpha (constant)."""
    # Cointegration relationship: ln(BTC) = 2.0 * ln(ETH) + 0.5 (Spread residual is exactly 0.0)
    trader = CointegratedPairsTrader(lookback_window=10)
    
    # Feed 10 perfectly correlated candles
    for i in range(10):
        eth_price = 3000.0 + i * 10
        ln_eth = np.log(eth_price)
        ln_btc = 2.0 * ln_eth + 0.5
        btc_price = np.exp(ln_btc)
        
        z = trader.ingest_prices(btc_price=btc_price, eth_price=eth_price)
        
    # Model should warm up on 10th candle
    assert z is not None
    # Check slope (beta) is approximately 2.0 and constant (alpha) is approximately 0.5
    assert pytest.approx(trader.beta, abs=1e-3) == 2.0
    assert pytest.approx(trader.alpha, abs=1e-3) == 0.5
    # The Spread resid is 0, so Z-score should be close to 0
    assert pytest.approx(z, abs=1e-2) == 0.0

def test_pairs_trader_stat_arb_lifecycle():
    """Verify the complete Stat-Arb lifecycle: Entry on Divergence -> Exit on Mean Reversion."""
    trader = CointegratedPairsTrader(lookback_window=15, entry_z=2.0, exit_z=0.2)
    
    # 1. Warm up with cointegrated series (Beta = 1.5, Alpha = 1.0)
    # Vary eth_price to ensure X has statistical variance for OLS stability
    # Warm up for 35 ticks to ensure spread_history queue is populated (N=20 residuals)
    for i in range(35):
        eth_price = 3000.0 + i * 10
        ln_eth = np.log(eth_price)
        ln_btc = 1.5 * ln_eth + 1.0
        btc_price = np.exp(ln_btc)
        z = trader.ingest_prices(btc_price=btc_price, eth_price=eth_price)
        
    assert z is not None
    assert pytest.approx(trader.beta, abs=1e-3) == 1.5
    assert pytest.approx(z, abs=1e-2) == 0.0
    
    # 2. Spike BTC price upwards to create an overvalued Spread divergence (Z >= 2.0)
    # Spot spikes from normal equivalent to 4% upward premium spike
    ref_eth = 3000.0 + 35 * 10
    btc_spike = np.exp(1.5 * np.log(ref_eth) + 1.0) * 1.04
    z_spike = trader.ingest_prices(btc_price=btc_spike, eth_price=ref_eth)
    
    assert z_spike >= 2.0
    
    # Evaluate trade setup: should trigger paired SHORT SPREAD with capital scaled to force exactly 1.0 BTC size
    cmd = trader.evaluate_trade_setup(btc_price=btc_spike, eth_price=ref_eth, z_score=z_spike, capital=btc_spike * 20.0)
    assert cmd is not None
    assert cmd["action"] == "ENTRY"
    assert cmd["type"] == "SHORT_SPREAD"
    assert cmd["btc_order"]["side"] == "SELL"
    assert cmd["btc_order"]["qty"] == 1.0
    assert cmd["eth_order"]["side"] == "BUY"
    # Assert quantity matches the dynamic delta-neutral exchange-weighted ratio
    assert cmd["eth_order"]["qty"] == pytest.approx(trader.beta * (btc_spike / ref_eth), abs=1e-3)
    
    assert trader.in_position == True
    assert trader.position_type == "SHORT_SPREAD"
    
    # 3. Revert the spread back to 0.0 (Mean Reversion profit target hit)
    btc_revert = np.exp(1.5 * np.log(ref_eth) + 1.0)
    z_revert = trader.ingest_prices(btc_price=btc_revert, eth_price=ref_eth)
    
    cmd_exit = trader.evaluate_trade_setup(btc_price=btc_revert, eth_price=ref_eth, z_score=z_revert, capital=btc_spike * 20.0)
    assert cmd_exit is not None
    assert cmd_exit["action"] == "EXIT"
    assert cmd_exit["type"] == "TAKE_PROFIT"
    assert cmd_exit["btc_order"]["side"] == "BUY"   # Paired closing orders
    assert cmd_exit["eth_order"]["side"] == "SELL"
    
    assert trader.in_position == False

def test_pairs_trader_stop_loss():
    """Verify that statistical decoupling triggers the Stop-Loss immediate exit."""
    trader = CointegratedPairsTrader(lookback_window=15, entry_z=2.0, stop_loss_z=3.0)
    
    # Warm up for 35 ticks
    for i in range(35):
        eth_price = 3000.0 + i * 10
        ln_eth = np.log(eth_price)
        ln_btc = 1.5 * ln_eth + 1.0
        btc_price = np.exp(ln_btc)
        z = trader.ingest_prices(btc_price=btc_price, eth_price=eth_price)
        
    # Spike to trigger Entry
    ref_eth = 3000.0 + 35 * 10
    btc_spike = np.exp(1.5 * np.log(ref_eth) + 1.0) * 1.045
    z_spike = trader.ingest_prices(btc_price=btc_spike, eth_price=ref_eth)
    cmd = trader.evaluate_trade_setup(btc_price=btc_spike, eth_price=ref_eth, z_score=z_spike, capital=btc_spike * 20.0)
    assert trader.in_position == True
    
    # Decouple further to trigger Stop Loss (Z >= 4.0)
    # Simulate a structural 25% decoupling event to overcome rolling variance dilation
    btc_crash = np.exp(1.5 * np.log(ref_eth) + 1.0) * 1.25
    z_crash = trader.ingest_prices(btc_price=btc_crash, eth_price=ref_eth)
    assert z_crash >= 3.0
    
    cmd_sl = trader.evaluate_trade_setup(btc_price=btc_crash, eth_price=3000.0, z_score=z_crash)
    assert cmd_sl is not None
    assert cmd_sl["action"] == "EXIT"
    assert cmd_sl["type"] == "STOP_LOSS"
    
    assert trader.in_position == False
