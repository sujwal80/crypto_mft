import json
import os
import sys
import time
import numpy as np
from typing import List

# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workspace_path)

from core.schemas import InternalTick
from backtester.engine import FastBacktestEngine
from intelligence.alpha_engine import AlphaModel
from backtester.backtest_dlq_rejections import load_dlq_data, extract_price_feed

# Standard optimized model parameters
MODEL_PARAMS = {
    "ML": {"lookback": 50, "tp_margin": 0.0060, "sl_margin": 0.0030},
    "HYBRID": {"lookback": 50, "tp_margin": 0.0180, "sl_margin": 0.0060, "threshold": 0.3},
    "KALMAN": {"lookback": 50, "tp_margin": 0.0060, "sl_margin": 0.0030, "threshold": 0.0015}
}

def reconstruct_ticks_from_feed(price_feed: list) -> List[InternalTick]:
    """
    Converts raw price feed ticks into standard InternalTick objects with dynamically 
    synthesized L2 bid/ask size depth dynamics using a Stochastic Liquidity Replenishment (SLR) model.
    """
    ticks = []
    print("Reconstructing L2 InternalTicks with synthesized SLR depth dynamics...")
    
    # Baseline parameters
    baseline_depth = 2.5        # Average book size at best bid/ask (BTC)
    min_depth = 0.1
    replenishment_rate = 0.15   # 15% decay rate back to baseline per tick
    
    # State variables for current book levels
    current_bid_size = baseline_depth
    current_ask_size = baseline_depth
    
    # Seed randomizer for reproducibility
    np.random.seed(42)
    
    for idx, item in enumerate(price_feed):
        price = item["price"]
        timestamp_ns = item["timestamp_ns"]
        
        # 1. Simulate bid/ask prices with tight spread (spread = 1.0 point)
        bid = price - 0.5
        ask = price + 0.5
        
        # 2. Reconstruct buying/selling pressure from price differentials
        if idx > 0:
            prev_price = price_feed[idx - 1]["price"]
            price_diff = price - prev_price
            
            # Apply mean-reversion replenishment to previous state
            current_bid_size += replenishment_rate * (baseline_depth - current_bid_size)
            current_ask_size += replenishment_rate * (baseline_depth - current_ask_size)
            
            if price_diff > 0:
                # Buying pressure -> Deplete ask size, bid size expands (bids join)
                impact = min(2.0, price_diff * 0.15)
                current_ask_size = max(min_depth, current_ask_size - impact)
                current_bid_size = min(8.0, current_bid_size + impact * 0.8)
            elif price_diff < 0:
                # Selling pressure -> Deplete bid size, ask size expands
                impact = min(2.0, abs(price_diff) * 0.15)
                current_bid_size = max(min_depth, current_bid_size - impact)
                current_ask_size = min(8.0, current_ask_size + impact * 0.8)
        
        # 3. Inject minor stochastic white noise (simulates normal L2 cancellations/additions)
        noise_bid = np.random.uniform(-0.2, 0.2)
        noise_ask = np.random.uniform(-0.2, 0.2)
        
        final_bid_size = float(np.clip(current_bid_size + noise_bid, min_depth, 10.0))
        final_ask_size = float(np.clip(current_ask_size + noise_ask, min_depth, 10.0))
        
        tick = InternalTick(
            symbol="BTCUSDT",
            exchange="BINANCE",
            bid=bid,
            ask=ask,
            bid_size=final_bid_size,
            ask_size=final_ask_size,
            timestamp_ns=timestamp_ns
        )
        ticks.append(tick)
        
    return ticks

def run_competitor_on_dlq(alpha_type: str, ticks: List[InternalTick]) -> dict:
    """Runs FastBacktestEngine over reconstructed DLQ ticks for a specific model type."""
    np.random.seed(42)
    params = MODEL_PARAMS.get(alpha_type, {})
    
    backtester = FastBacktestEngine(
        initial_cash=10000.0,
        latency_ticks=1,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_std=0.0001,
        tp_margin=params.get("tp_margin"),
        sl_margin=params.get("sl_margin"),
        lookback=params.get("lookback"),
        reversal_threshold=params.get("reversal_threshold"),
        timeout_seconds=params.get("timeout_seconds")
    )
    
    threshold = params.get("threshold")
    if threshold is not None:
        backtester.alpha_model = AlphaModel(alpha_type=alpha_type, threshold=threshold)
    else:
        backtester.alpha_model = AlphaModel(alpha_type=alpha_type)
        
    return backtester.run_backtest(ticks)

def main():
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DLQ_FILE = os.path.join(PROJECT_DIR, "dlq_audit.json")
    
    print("==================================================================")
    print("📈 DYNAMIC DLQ HISTORICAL COMPETITION HARNESS")
    print("==================================================================")
    
    # Load DLQ raw records
    records = load_dlq_data(DLQ_FILE)
    if not records:
        return
        
    # Extract spot price trajectory
    price_feed = extract_price_feed(records)
    if not price_feed:
        return
        
    # Reconstruct high-precision InternalTicks
    ticks = reconstruct_ticks_from_feed(price_feed)
    
    competitors = ["HYBRID", "KALMAN"]
    if os.path.exists(os.path.join(PROJECT_DIR, "weights.lgb")) or os.path.exists(os.path.join(PROJECT_DIR, "weights.npy")):
        competitors.append("ML")
        
    results = {}
    print("\nSimulating competitors on the reconstructed DLQ price path...")
    for model in competitors:
        print(f"Running backtest for: [{model}]...")
        results[model] = run_competitor_on_dlq(model, ticks)
        
    print("\n=================================================================================")
    print("👑 COMPETITION LEAGUE TABLE (ON RECONSTRUCTED DLQ PATH):")
    print("=================================================================================")
    print(f"{'Model':<12} | {'P&L ($)':<12} | {'Return (%)':<12} | {'Max DD (%)':<12} | {'Win Rate':<10} | {'Trades':<8} | {'Fees ($)':<8}")
    print("-" * 85)
    for model, res in results.items():
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

if __name__ == "__main__":
    main()
