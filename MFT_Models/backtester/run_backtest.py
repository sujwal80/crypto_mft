import os
import sys
# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workspace_path)

import time
import numpy as np
from typing import List

from core.schemas import InternalTick
from backtester.engine import FastBacktestEngine

def generate_synthetic_market_data(num_ticks: int = 10000) -> List[InternalTick]:
    """Generates a calibrated synthetic sequence of high-frequency ticks matching real-world Binance L2 stats."""
    ticks = []
    base_price = 77500.0
    current_trend = 0.0
    np.random.seed(42)
    
    # AR(1) price step history for return momentum (lag-1 autocorrelation ~ +0.155)
    prev_step = 0.0
    momentum_coeff = 0.155

    print(f"Generating {num_ticks} ticks of calibrated synthetic L2 order book data...")

    for i in range(num_ticks):
        # Inject regime cycles (ranging vs sharp breakout trends)
        if i % 3000 == 0:
            # Trend strength scaled to real-world scales (0.02 basis points drift per tick)
            current_trend = np.random.choice([-0.015, 0.0, 0.015])

        # Step Price with AR(1) return momentum model
        noise = np.random.normal(loc=current_trend, scale=0.65)
        price_step = (momentum_coeff * prev_step) + ((1.0 - momentum_coeff) * noise)
        base_price += price_step
        prev_step = price_step

        # Simulate Real-World Bid/Ask Spread ($0.01 to $0.02)
        spread = np.random.uniform(0.01, 0.02)
        bid = base_price - (spread / 2.0)
        ask = base_price + (spread / 2.0)

        # Simulate Volumetric Imbalance (OBI)
        if current_trend > 0:
            bid_size = np.random.uniform(1.5, 5.0)
            ask_size = np.random.uniform(0.1, 1.0)
        elif current_trend < 0:
            bid_size = np.random.uniform(0.1, 1.0)
            ask_size = np.random.uniform(1.5, 5.0)
        else:
            bid_size = np.random.uniform(0.5, 2.0)
            ask_size = np.random.uniform(0.5, 2.0)

        tick = InternalTick(
            symbol="BTCUSDT",
            exchange="BINANCE",
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp_ns=int((time.time() - (num_ticks - i) * 0.1) * 1e9) # 100ms spacing
        )
        ticks.append(tick)

    return ticks

def main():
    print("=================================================================================")
    print("🚀 ENTERPRISE MFT BACKTEST HARNESS - SYNTHETIC VALIDATION SESSION")
    print("=================================================================================")

    # #1. Generate 10,000 ticks of high-frequency order book data
    ticks = generate_synthetic_market_data(num_ticks=10000)

    # #2. Initialize Backtester (1 tick latency - 100ms, 0.02% Maker Fee)
    backtester = FastBacktestEngine(
        initial_cash=10000.0,
        latency_ticks=1,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_std=0.0001
    )

    # #3. Execute Simulation
    print("\nExecuting backtest over synthetic tick series...")
    start_time = time.time()
    results = backtester.run_backtest(ticks)
    duration = time.time() - start_time

    # #4. Print Performance Analytics Report
    print(f"Backtest completed in {duration:.3f} seconds.")
    print("=================================================================================")
    print("📊 PERFORMANCE REPORT CARD:")
    print("=================================================================================")
    print(f"Starting Balance       : $10,000.00")
    print(f"Ending Balance         : ${results['final_balance']:.2f}")
    print(f"Net Profit/Loss        : ${results['net_pnl']:+.2f} ({results['net_percentage_return']:+.2f}%)")
    print(f"Max Drawdown           : {results['max_drawdown']:.2f}%")
    print(f"Annualized Sharpe      : {results['sharpe_ratio']:.2f}")
    print(f"Total Executed Trades  : {results['total_trades']}")
    print(f"Maker Win Rate         : {results['win_rate']:.2f}%")
    print(f"Total Exchange Fees    : ${results['total_fees_paid']:.2f}")
    print("=================================================================================")

if __name__ == "__main__":
    main()
