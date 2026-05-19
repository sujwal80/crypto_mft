import sys
import os
import time
import numpy as np

# Add workspace to path
workspace_path = "/Users/singhujwal/crypto_mft"
sys.path.append(workspace_path)

from core.schemas import InternalTick
from run_backtest import generate_synthetic_market_data
from backtester.engine import FastBacktestEngine
from intelligence.alpha_engine import AlphaModel

def main():
    print("=================================================================================")
    print("📈 DIRECTIONAL BIAS STRATEGY - PROFIT-ONLY EXIT VALIDATION")
    print("=================================================================================")

    # Reset seed for reproducibility
    np.random.seed(42)

    # 1. Generate 15,000 ticks
    ticks = generate_synthetic_market_data(num_ticks=15000)
    print(f"Generated {len(ticks)} L2 test ticks.\n")

    # 2. Configure Backtester
    # TP = 0.5%, SL = Disabled (999.0), Timeout = Disabled (99999999)
    backtester = FastBacktestEngine(
        initial_cash=10000.0,
        latency_ticks=1,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_std=0.0001,
        tp_margin=0.005,         # 0.5% Take Profit
        sl_margin=999.0,         # Effectively Disabled Stop Loss
        timeout_seconds=99999999, # Effectively Disabled Timeout
        lookback=50             # Standard lookback
    )
    
    # Set strategy to DIRECTIONAL
    backtester.alpha_model = AlphaModel(alpha_type="DIRECTIONAL")

    # 3. Run Simulation
    print("Executing backtest for Competitor: [DIRECTIONAL]...")
    start_time = time.time()
    results = backtester.run_backtest(ticks)
    duration = time.time() - start_time

    # 4. Report Results
    print(f"\nBacktest completed in {duration:.3f} seconds.")
    print("=================================================================================")
    print("📊 PERFORMANCE REPORT:")
    print("=================================================================================")
    print(f"Starting Balance       : $10,000.00")
    print(f"Ending Balance         : ${results['final_balance']:.2f}")
    print(f"Net Profit/Loss        : ${results['net_pnl']:+.2f} ({results['net_percentage_return']:+.2f}%)")
    print(f"Max Drawdown           : {results['max_drawdown']:.2f}%")
    print(f"Total Executed Trades  : {results['total_trades']}")
    print(f"Win Rate               : {results['win_rate']:.2f}%")
    print(f"Total Exchange Fees    : ${results['total_fees_paid']:.2f}")
    print("=================================================================================")
    print("💡 NOTE: Executed with 0.5% TP, Disabled SL, and Disabled Timeout.")
    print("=================================================================================")

if __name__ == "__main__":
    main()
