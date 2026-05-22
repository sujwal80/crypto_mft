import sys
import os
import pytest

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backtester")))
from swing_backtester import SwingBacktestEngine

def test_swing_backtester_live_log():
    """Verify SwingBacktestEngine successfully runs on a live market log file."""
    # Use a smaller dataset to run tests quickly
    file_path = "/Users/singhujwal/crypto_mft/datasets/synthetic_market_data_extremely_downtrend.log"
    if not os.path.exists(file_path):
        pytest.skip("Dataset not found, skipping test.")
        
    engine = SwingBacktestEngine(initial_cash=10000.0)
    results = engine.stream_backtest(file_path)
    
    assert "final_balance" in results
    assert "net_pnl" in results
    assert "max_drawdown" in results
    assert "total_trades" in results
    assert "total_fees_paid" in results
    assert len(results["hourly_reports"]) > 0
