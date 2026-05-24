import sys
import os
import pytest

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backtester")))
from engine import GexBacktestEngine
from run_gex_backtest import generate_backtest_data

@pytest.mark.asyncio
async def test_gex_backtest_engine():
    """Verify GexBacktestEngine completes runs successfully and returns valid metrics."""
    engine = GexBacktestEngine(initial_cash=10000.0, grace_window_ticks=0, resample_ticks=1)
    ticks = generate_backtest_data()
    
    results = await engine.run_backtest(
        ticks=ticks,
        key_strike=60000.0,
        gex_value=150.0,
        deribit_index=60000.0
    )
    
    # Verify return structure
    assert "final_balance" in results
    assert "net_pnl" in results
    assert "net_percentage_return" in results
    assert "max_drawdown" in results
    assert "total_trades" in results
    assert "win_rate" in results
    
    # Ensure it registered the entry & dynamic invalidation trade
    assert results["total_trades"] == 1
    assert results["win_rate"] == 0.0 # It was an invalidation cut, so it's a loss
    assert results["total_fees_paid"] > 0.0
