import os
import sys
import time
import json
import numpy as np
from typing import List, Dict

# Add workspace to path
workspace_path = "/Users/singhujwal/crypto_mft"
sys.path.append(workspace_path)

from core.schemas import InternalTick
from run_backtest import generate_synthetic_market_data
from backtester.engine import FastBacktestEngine
from intelligence.alpha_engine import AlphaModel

REAL_DATA_PATH = os.path.join(workspace_path, "real_market_data_10m.jsonl")

def load_real_market_data(filepath: str) -> List[InternalTick]:
    """Parses JSONL file containing real-time recorded Binance L2 ticks."""
    ticks = []
    if not os.path.exists(filepath):
        print(f"❌ Error: Real market data file not found at: {filepath}")
        return []
    
    with open(filepath, "r") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    tick = InternalTick.model_validate_json(line_str)
                    ticks.append(tick)
                except Exception as e:
                    print(f"Warning: Skipping invalid tick line: {e}")
    return ticks

def run_grid_search(ticks: List[InternalTick], dataset_name: str, top_n: int = 5) -> List[Dict]:
    """Executes parameter grid sweep over the quantitative strategy metrics."""
    print(f"\n=================================================================================")
    print(f"🔍 STARTING PARAMETER GRID SWEEP FOR: [{dataset_name}] ({len(ticks)} ticks)")
    print(f"=================================================================================")
    
    # Parameter Search Space (Medium-Frequency Trend-Following Sweep: Wider margins, no timeouts)
    thresholds = [0.20, 0.30, 0.40]
    tp_margins = [0.0035, 0.0050, 0.0070]  # 0.35%, 0.50%, 0.70%
    sl_margins = [0.0015, 0.0020, 0.0030]  # 0.15%, 0.20%, 0.30%
    reversal_thresholds = [0.0010, 0.0020] # 10 bps, 20 bps reversal thresholds
    
    results = []
    total_runs = len(thresholds) * len(tp_margins) * len(sl_margins) * len(reversal_thresholds)
    run_count = 0
    
    start_time = time.time()
    
    for thresh in thresholds:
        for tp in tp_margins:
            for sl in sl_margins:
                for rev in reversal_thresholds:
                    run_count += 1
                    if run_count % 30 == 0 or run_count == total_runs:
                        elapsed = time.time() - start_time
                        print(f"Sweep Progress: {run_count}/{total_runs} ({(run_count/total_runs)*100:.1f}%) | Elapsed: {elapsed:.1f}s")
                    
                    # Reset seed for exact reproducibility across iterations
                    np.random.seed(42)
                    
                    # Initialize engine with custom exit boundaries
                    backtester = FastBacktestEngine(
                        initial_cash=10000.0,
                        latency_ticks=1,
                        maker_fee=0.0002,
                        taker_fee=0.0004,
                        slippage_std=0.0001,
                        tp_margin=tp,
                        sl_margin=sl,
                        lookback=50,
                        reversal_threshold=rev
                    )
                    
                    # Configure AlphaModel with MICRO_TREND and specific threshold argument
                    backtester.alpha_model = AlphaModel(alpha_type="MICRO_TREND", threshold=thresh)
                    
                    # Run simulation
                    res = backtester.run_backtest(ticks)
                    
                    # Save configuration and performance metrics
                    results.append({
                        "threshold": thresh,
                        "tp_margin": tp,
                        "sl_margin": sl,
                        "reversal_threshold": rev,
                        "net_pnl": res["net_pnl"],
                        "net_percentage_return": res["net_percentage_return"],
                        "max_drawdown": res["max_drawdown"],
                        "win_rate": res["win_rate"],
                        "total_trades": res["total_trades"],
                        "total_fees_paid": res["total_fees_paid"],
                        "sharpe_ratio": res["sharpe_ratio"]
                    })
                    
    # Sort by Net PnL descending
    results.sort(key=lambda x: x["net_pnl"], reverse=True)
    
    print(f"Grid sweep for [{dataset_name}] complete in {time.time() - start_time:.2f} seconds.")
    
    # Display Top N Configurations
    print(f"\n🏆 TOP {top_n} CONFIGURATIONS FOR [{dataset_name}]:")
    print(f"{'Rank':<4} | {'Thresh':<6} | {'TP (%)':<6} | {'SL (%)':<6} | {'Rev (%)':<7} | {'P&L ($)':<9} | {'Return':<7} | {'WinRate':<7} | {'Trades':<6} | {'Fees ($)':<8}")
    print("-" * 92)
    for rank, cfg in enumerate(results[:top_n], 1):
        rev_str = f"{cfg['reversal_threshold']*100:.2f}%" if cfg['reversal_threshold'] is not None else "None"
        print(
            f"{rank:<4} | "
            f"{cfg['threshold']:<6.2f} | "
            f"{cfg['tp_margin']*100:<6.2f}% | "
            f"{cfg['sl_margin']*100:<6.2f}% | "
            f"{rev_str:<7} | "
            f"${cfg['net_pnl']:+8.2f} | "
            f"{cfg['net_percentage_return']:+6.2f}% | "
            f"{cfg['win_rate']:6.2f}% | "
            f"{cfg['total_trades']:<6d} | "
            f"${cfg['total_fees_paid']:<7.2f}"
        )
        
    return results

def main():
    print("=================================================================================")
    print("🧪 QUANTITATIVE MICRO-TREND STRATEGY - DEEP BACKTEST VALIDATION SUITE")
    print("=================================================================================")
    
    # 1. Generate 15,000 high-frequency synthetic ticks
    synthetic_ticks = generate_synthetic_market_data(num_ticks=15000)
    
    # 2. Load 10-minute real-time recorded Binance data
    print(f"\nLoading real-world depth data from: {REAL_DATA_PATH} ...")
    real_ticks = load_real_market_data(REAL_DATA_PATH)
    print(f"Successfully loaded {len(real_ticks)} real-world Binance depth ticks.")
    
    # 3. Run Grid Sweep on Synthetic Data
    synth_results = run_grid_search(synthetic_ticks, "SYNTHETIC DATASET (15K Ticks)", top_n=8)
    
    # 4. Run Grid Sweep on Real Data
    real_results = []
    if real_ticks:
        real_results = run_grid_search(real_ticks, "REAL RECORDED DATASET (5.9K Ticks)", top_n=8)
        
    print("\n=================================================================================")
    print("💡 VALIDATION COMPLETE")
    print("=================================================================================")

if __name__ == "__main__":
    main()
