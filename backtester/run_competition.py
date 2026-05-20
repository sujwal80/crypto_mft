import os
import sys
# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workspace_path)

import time
import numpy as np
from typing import List

from core.schemas import InternalTick
from backtester.run_backtest import generate_synthetic_market_data
from backtester.engine import FastBacktestEngine
from intelligence.alpha_engine import AlphaModel

# Optimized parameters found via grid search
MODEL_PARAMS = {
    "ML": {"lookback": 50, "tp_margin": 0.0060, "sl_margin": 0.0030},
    "HYBRID": {"lookback": 50, "tp_margin": 0.0180, "sl_margin": 0.0060, "threshold": 0.3},
    "KALMAN": {"lookback": 50, "tp_margin": 0.0060, "sl_margin": 0.0030, "threshold": 0.0015}
}

def run_competitor(alpha_type: str, ticks: List[InternalTick]) -> dict:
    """Configures backtest engine with specific model type and returns performance outcomes."""
    # Reset seed for reproducibility across competitors
    np.random.seed(42)
    
    params = MODEL_PARAMS.get(alpha_type, {})
    
    backtester = FastBacktestEngine(
        initial_cash=10000.0,
        latency_ticks=1,
        maker_fee=0.0002,     # Unified maker fee (0.02%)
        taker_fee=0.0004,     # Unified taker fee (0.04%)
        slippage_std=0.0001,
        tp_margin=params.get("tp_margin"),
        sl_margin=params.get("sl_margin"),
        lookback=params.get("lookback"),
        reversal_threshold=params.get("reversal_threshold"),
        timeout_seconds=params.get("timeout_seconds")
    )
    # Override internal AlphaModel type selection
    threshold = params.get("threshold")
    if threshold is not None:
        backtester.alpha_model = AlphaModel(alpha_type=alpha_type, threshold=threshold)
    else:
        backtester.alpha_model = AlphaModel(alpha_type=alpha_type)

    return backtester.run_backtest(ticks)

def main():
    print("=================================================================================")
    print("📈 ENTERPRISE MFT - QUANTITATIVE MODEL LEAGUE HARNESS")
    print("=================================================================================")

    # 1. Generate 15,000 ticks to provide enough data for regime testing
    ticks = generate_synthetic_market_data(num_ticks=15000)
    print(f"Generated {len(ticks)} high-frequency L2 test ticks successfully.\n")

    competitors = ["HYBRID", "KALMAN"]
    if os.path.exists("weights.lgb") or os.path.exists("weights.npy"):
        competitors.append("ML")
        print("Trained ML weights ('weights.lgb' or 'weights.npy') detected. Including ML in league!")
    else:
        print("No trained ML weights found. (Run 'python3 train_ml_model.py' to enable ML).")

    results = {}

    print("\nExecuting league simulations...")
    for model in competitors:
        print(f"Running backtest for Competitor: [{model}]...")
        results[model] = run_competitor(model, ticks)

    print("\n=================================================================================")
    print("👑 THE QUANTITATIVE MODEL LEAGUE TABLE:")
    print("=================================================================================")
    # Format Results Table
    print(f"{'Model':<12} | {'P&L ($)':<12} | {'Return (%)':<12} | {'Max DD (%)':<12} | {'Win Rate':<10} | {'Trades':<8} | {'Fees ($)':<8}")
    print("-" * 85)
    for model, res in results.items():
        if "error" in res:
            print(f"{model:<12} | Error: {res['error']}")
            continue
        print(
            f"{model:<12} | "
            f"${res['net_pnl']:+10.2f} | "
            f"{res['net_percentage_return']:+10.2f}% | "
            f"{res['max_drawdown']:9.2f}% | "
            f"{res['win_rate']:7.2f}% | "
            f"{res['total_trades']:8d} | "
            f"${res['total_fees_paid']:7.2f}"
        )
    print("=================================================================================")
    print("💡 NOTE: Performance is calculated using 100ms latency and 0.1%/0.1% fees.")
    print("=================================================================================")

if __name__ == "__main__":
    main()
