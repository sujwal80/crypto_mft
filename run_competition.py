import time
import sys
import os
import numpy as np
from typing import List

from core.schemas import InternalTick
from run_backtest import generate_synthetic_market_data
from backtester.engine import FastBacktestEngine
from intelligence.alpha_engine import AlphaModel

def run_competitor(alpha_type: str, ticks: List[InternalTick]) -> dict:
    """Configures backtest engine with specific model type and returns performance outcomes."""
    # Force mock instance to load competitor alpha type
    backtester = FastBacktestEngine(
        initial_cash=10000.0,
        latency_ticks=1,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_std=0.0001
    )
    # Override internal AlphaModel type selection
    backtester.alpha_model = AlphaModel(alpha_type=alpha_type)

    return backtester.run_backtest(ticks)

def main():
    print("=================================================================================")
    print("📈 ENTERPRISE MFT - MATHEMATICAL ALPHA COMPETITION HARNESS")
    print("=================================================================================")

    # 1. Generate 15,000 ticks to provide enough data for regime testing
    ticks = generate_synthetic_market_data(num_ticks=15000)
    print(f"Generated {len(ticks)} high-frequency L2 test ticks successfully.\n")

    competitors = ["OU", "KALMAN", "OFI", "HEURISTIC"]
    results = {}

    print("Executing competitions across all models...")
    for model in competitors:
        print(f"Running backtest for Competitor: [{model}]...")
        results[model] = run_competitor(model, ticks)

    print("\n=================================================================================")
    print("👑 THE MATHEMATICAL MODEL LEAGUE TABLE:")
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
    print("💡 NOTE: Performance is calculated using 100ms latency and 0.02%/0.04% fees.")
    print("=================================================================================")

if __name__ == "__main__":
    main()
