import os
import sys
import time
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from engine import GexBacktestEngine, BacktestTick

def generate_backtest_data() -> list:
    """
    Generates a calibrated dataset representing spot approaching a major GEX Put Wall.
    Includes three distinct phases:
      - Phase 1: Spot wanders down from $61,000 to $60,200 (Hibernation)
      - Phase 2: Spot enters GEX Proximity at $60,000 (Armed -> Execution Matrix confirmation)
      - Phase 3: Sudden market bid dump/vanish, triggering Dynamic Invalidation exit.
    """
    ticks = []
    timestamp = time.time()
    
    # 1. Phase 1: Wandering down (Ticks 0 to 100)
    for i in range(100):
        price = 61000.0 - (i * 8.0) # Smooth decay towards 60,200
        ticks.append(BacktestTick(
            timestamp=timestamp + i * 0.1,
            price=price,
            bid_qty=10.0,
            ask_qty=10.0
        ))
        
    # 2. Phase 2: Proximity & Target Hit at 60,000 (Ticks 100 to 200)
    # We prime the CVD engine with heavy taker selling first
    for i in range(100):
        ticks.append(BacktestTick(
            timestamp=timestamp + (100 + i) * 0.1,
            price=60000.0,
            bid_qty=10.0,
            ask_qty=10.0,
            is_trade=True,
            trade_price=60000.0,
            trade_qty=25.0,
            is_buyer_maker=True # Seller taker
        ))
        
    # 3. Sniper Confirm Trigger: Spike Bid volume to stack bids (Z-score > 2.0)
    ticks.append(BacktestTick(
        timestamp=timestamp + 200 * 0.1,
        price=60000.0,
        bid_qty=800.0,
        ask_qty=10.0,
        is_trade=True,
        trade_price=60000.0,
        trade_qty=15.0,
        is_buyer_maker=True
    ))
    
    # 4. Phase 3: Sudden Market Breakdown (Ticks 201 to 250)
    # Bids vanish, CVD selling continues, spot falls
    for i in range(50):
        price = 60000.0 - (i * 10.0)
        ticks.append(BacktestTick(
            timestamp=timestamp + (201 + i) * 0.1,
            price=price,
            bid_qty=1.0,
            ask_qty=1000.0, # Massive Ask volume representing sell panic
            is_trade=True,
            trade_price=price,
            trade_qty=50.0,
            is_buyer_maker=True
        ))
        
    return ticks

async def main():
    print("=================================================================================")
    print("🚀 GEX-MICRO STATE MACHINE QUANT BACKTESTER - EVENT SIMULATION")
    print("=================================================================================")
    
    # 1. Load data
    ticks = generate_backtest_data()
    print(f"Generated {len(ticks)} ticks representing GEX wall interaction.")
    
    # 2. Instantiate Backtester
    engine = GexBacktestEngine(
        initial_cash=10000.0,
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_pct=0.0003
    )
    
    # 3. Run simulation on $60,000 Put Wall
    print("\nRunning historical event-driven backtest...")
    results = await engine.run_backtest(
        ticks=ticks,
        key_strike=60000.0,
        gex_value=150.0,
        deribit_index=60000.0
    )
    
    # 4. Print Performance Report
    print("=================================================================================")
    print("📊 PERFORMANCE METRICS:")
    print("=================================================================================")
    print(f"Starting Capital       : $10,000.00")
    print(f"Ending Capital         : ${results['final_balance']:.2f}")
    print(f"Net Profit/Loss        : ${results['net_pnl']:+.2f} ({results['net_percentage_return']:+.2f}%)")
    print(f"Max Portfolio Drawdown : {results['max_drawdown']:.2f}%")
    print(f"Estimated Sharpe Ratio : {results['sharpe_ratio']:.2f}")
    print(f"Total Executed Trades  : {results['total_trades']}")
    print(f"Maker Win Rate         : {results['win_rate']:.2f}%")
    print(f"Total Exchange Fees    : ${results['total_fees_paid']:.2f}")
    print("=================================================================================")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
