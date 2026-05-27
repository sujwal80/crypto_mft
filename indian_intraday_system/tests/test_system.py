"""Comprehensive tests for the indian_intraday_system components and integration flow."""

import asyncio
import pytest
from indian_intraday_system import config
from indian_intraday_system.layer_1_data.truedata_ws import TrueDataWSClient
from indian_intraday_system.layer_2_macro.gex_mapper import calculate_gex, map_gex_levels
from indian_intraday_system.layer_2_macro.vanilla_bs import black_scholes_greeks, implied_volatility
from indian_intraday_system.layer_3_micro.basis_tracker import BasisTracker
from indian_intraday_system.main import GexMicroSystem


def test_nse_tick_formatting():
    # Test standard round_to_nse_tick formatting to nearest 0.05 paise
    assert config.round_to_nse_tick(22050.12) == 22050.10
    assert config.round_to_nse_tick(22050.13) == 22050.15
    assert config.round_to_nse_tick(22050.17) == 22050.15
    assert config.round_to_nse_tick(22050.18) == 22050.20


def test_vanilla_bs_with_bisection_patch():
    # Pricing
    S, K, T, r, sigma = 22000.0, 22000.0, 0.05, 0.07, 0.15
    res = black_scholes_greeks(S, K, T, r, sigma, "C")
    assert res["price"] > 0

    # IV solver: verifying bisection patch does not fail on broadcasting mismatch
    target_ivs = [0.12, 0.18, 0.22]
    prices = []
    for iv in target_ivs:
        prices.append(black_schreeks_price(S, K, T, r, iv))

    # Solved IV check
    solved_ivs = implied_volatility(
        price=prices, S=S, K=K, T=T, r=r, option_type="C"
    )
    assert len(solved_ivs) == 3
    for solved, target in zip(solved_ivs, target_ivs):
        assert solved == pytest.approx(target, abs=1e-3)


def black_schreeks_price(S, K, T, r, sigma):
    """Helper method to fetch price."""
    return black_scholes_greeks(S, K, T, r, sigma, "C")["price"]


def test_basis_tracker():
    tracker = BasisTracker(window_size=10)

    # Feed ticks
    tracker.add_tick(future_price=22015.0, spot_price=22000.0)  # Spread = 15.0
    tracker.add_tick(future_price=22016.0, spot_price=22000.0)  # Spread = 16.0
    tracker.add_tick(future_price=22014.0, spot_price=22000.0)  # Spread = 14.0
    tracker.add_tick(future_price=22015.0, spot_price=22000.0)  # Spread = 15.0
    tracker.add_tick(future_price=22015.0, spot_price=22000.0)  # Spread = 15.0

    stats = tracker.get_basis_stats()
    assert stats["mean"] == pytest.approx(15.0)
    assert stats["std"] > 0.0

    # Push speculative premium anomaly tick
    tracker.add_tick(future_price=22025.0, spot_price=22000.0)  # Spread = 25.0
    anomaly_stats = tracker.get_basis_stats()
    assert anomaly_stats["z_score"] > 1.5


def test_dynamic_option_window():
    client = TrueDataWSClient()
    assert len(client.subscribed_option_symbols) == 0

    # Set initial spot = 22025.00 (ATM strike: 22000)
    client.update_dynamic_option_window(22025.00)
    assert client.latest_atm_strike == 22000
    
    # Check strikes subscription size (ATM +/- 5 is 11 strikes. For CE and PE, it's 22 contracts)
    assert len(client.subscribed_option_symbols) == 22
    assert "NIFTY26MAY22000CE" in client.subscribed_option_symbols
    assert "NIFTY26MAY22000PE" in client.subscribed_option_symbols

    # Shift spot price up to 22220.00 (ATM strike: 22200)
    client.update_dynamic_option_window(22220.00)
    assert client.latest_atm_strike == 22200
    assert len(client.subscribed_option_symbols) == 22
    assert "NIFTY26MAY22200CE" in client.subscribed_option_symbols
    
    # Far strike like 21700 (previously ATM-6) should be unsubscribed
    assert "NIFTY26MAY21700CE" not in client.subscribed_option_symbols


@pytest.mark.asyncio
async def test_indian_intraday_system_momentum_breakout():
    # Initialize active coordinator
    system = GexMicroSystem()

    # Bind Mock Time Regimes
    system.state_machine.time_manager.get_current_regime = lambda: "MOMENTUM"

    # Feed setup
    await system.data_client.connect()
    await system.data_client.subscribe_futures("NIFTY_FUT")

    # Feed enough basis ticks to build standard statistics
    for _ in range(10):
        system.basis_tracker.add_tick(future_price=22015.0, spot_price=22000.0)

    # Options walls
    gex_levels = {
        "zero_gamma": 21950.00,
        "call_wall": 22100.00,
        "put_wall": 21800.00,
    }

    # Trigger Market Breakout Ticks
    breakout_tick = {
        "symbol": "NIFTY_FUT",
        "type": "trade",
        "price": 22175.00,
        "spot_reference": 22150.00,
        "volume": 15000.0,
        "bid": 22174.00,
        "ask": 22176.00,
    }
    
    # Feed options chain snap matching breakout: 22100 strike CE has massive open interest (forces Call Wall = 22100)
    chain_tick = {
        "symbol": "NIFTY_OPTION_CHAIN",
        "type": "chain",
        "spot": 22150.00,
        "strikes": [21800, 21900, 22000, 22100, 22200] * 2,
        "ivs": [0.15] * 10,
        "open_interests": [30000, 30000, 30000, 100000, 20000, 30000, 30000, 30000, 30000, 30000],
        "option_types": ["C", "C", "C", "C", "C", "P", "P", "P", "P", "P"],
        "expiries_days": [6] * 10,
    }

    # Execute callbacks to feed system
    system.on_market_update(breakout_tick)
    system.on_market_update(chain_tick)

    # Check strategy filled successfully
    positions = system.router.get_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos["symbol"] == "NIFTY_FUT"
    assert pos["side"] == "BUY"
    
    # Verify rounded NSE Tick formatting on fill price
    assert pos["entry_price"] == 22175.50

    # Teardown
    await system.data_client.disconnect()


@pytest.mark.asyncio
async def test_live_data_recorder():
    import os
    import time
    import pandas as pd
    from indian_intraday_system.utils.data_recorder import LiveDataRecorder

    test_dir = "./datasets/test_raw_ticks"
    recorder = LiveDataRecorder(output_dir=test_dir)
    
    # Clean existing test files
    date_str = time.strftime("%Y-%m-%d")
    target_file = os.path.join(test_dir, f"ticks_{date_str}.csv")
    if os.path.exists(target_file):
        os.remove(target_file)

    # 1. Start recorder
    recorder.start()
    assert recorder.active is True

    # 2. Record mock ticks
    mock_tick_1 = {
        "symbol": "NIFTY_FUT",
        "type": "trade",
        "price": 22015.00,
        "spot_reference": 22000.00,
        "volume": 100.0,
    }
    mock_tick_2 = {
        "symbol": "NIFTY_OPTION_CHAIN",
        "type": "chain",
        "spot": 22000.00,
    }

    await recorder.record_tick(mock_tick_1)
    await recorder.record_tick(mock_tick_2)

    # 3. Force manual flush to disk
    await recorder.flush_to_disk()
    assert len(recorder.buffer) == 0

    # 4. Stop recorder
    await recorder.stop()
    assert recorder.active is False

    # 5. Verify file created and has correct columns
    assert os.path.exists(target_file) is True
    
    df = pd.read_csv(target_file)
    assert len(df) == 2
    assert list(df["symbol"].values) == ["NIFTY_FUT", "NIFTY_OPTION_CHAIN"]
    assert "local_timestamp" in df.columns

    # Cleanup
    if os.path.exists(target_file):
        os.remove(target_file)

