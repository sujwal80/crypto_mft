import os
import sys
import time

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from swing_backtester import SwingBacktestEngine

def main():
    print("=================================================================================")
    print("🚀 GEX-MICRO STATE MACHINE - HISTORICAL REAL MARKET BACKTEST")
    print("=================================================================================")
    
    file_path = "/Users/singhujwal/crypto_mft/datasets/real_market_data_live.log"
    if not os.path.exists(file_path):
        print(f"Error: Dataset {file_path} not found.")
        return
        
    # Instantiate backtest engine with 0.1% maker and 0.1% taker fees
    engine = SwingBacktestEngine(
        initial_cash=10000.0,
        maker_fee=0.0010,
        taker_fee=0.0010
    )
    
    start_time = time.time()
    results = engine.stream_backtest(file_path)
    duration = time.time() - start_time
    
    print("\n=================================================================================")
    print("📊 REAL MARKET DATA BACKTEST REPORT:")
    print("=================================================================================")
    print(f"Starting Balance       : $10,000.00")
    print(f"Ending Balance         : ${results['final_balance']:.2f}")
    print(f"Net Profit/Loss        : ${results['net_pnl']:+.2f} ({results['net_percentage_return']:+.2f}%)")
    print(f"Max Portfolio Drawdown : {results['max_drawdown']:.2f}%")
    print(f"Total Trades Executed  : {results['total_trades']}")
    print(f"Maker Win Rate         : {results['win_rate']:.2f}%")
    print(f"Total Exchange Fees    : ${results['total_fees_paid']:.2f}")
    print(f"Execution Duration     : {duration:.2f} seconds")
    print("=================================================================================")

if __name__ == "__main__":
    main()
