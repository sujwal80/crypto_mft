import os
import sys
import time
import logging
import json

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from swing_backtester import SwingBacktestEngine

logger = logging.getLogger("MasterBacktest")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

async def main():
    print("=================================================================================")
    print("🚀 ENTERPRISE GEX-MICRO STATE MACHINE BACKTEST SUITE - DEPLOYMENT STAGE")
    print("=================================================================================")
    
    datasets_dir = "/Users/singhujwal/crypto_mft/datasets"
    if not os.path.exists(datasets_dir):
        print(f"Error: Datasets directory {datasets_dir} not found.")
        return
        
    # Find all log files in the datasets folder
    files = [os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir) if f.endswith(".log")]
    if not files:
        print("No log files found in the datasets directory.")
        return
        
    # Sort files to process smaller ones first and massive one last
    files.sort(key=os.path.getsize)
    
    print(f"Detected {len(files)} datasets in root folder. Starting backtesting cycle...\n")
    
    # Instantiate high-performance engine
    # Strict 0.1% maker fee (0.0010) and 0.1% taker fee (0.0010)
    engine = SwingBacktestEngine(
        initial_cash=10000.0,
        maker_fee=0.0010,
        taker_fee=0.0010
    )
    
    all_results = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        print("=================================================================================")
        print(f"📁 DATASET: {filename} ({size_mb:.2f} MB)")
        print("=================================================================================")
        
        start_time = time.time()
        results = await engine.stream_backtest(file_path)
        duration = time.time() - start_time
        
        print(f"\nProcessing completed in {duration:.2f} seconds.")
        print(f"Dataset Net Profit/Loss: ${results['net_pnl']:+.2f} ({results['net_percentage_return']:+.2f}%)")
        print(f"Total Trades Executed  : {results['total_trades']}")
        print(f"Exchange Fees Paid     : ${results['total_fees_paid']:.2f}")
        print(f"Maximum Drawdown       : {results['max_drawdown']:.2f}%")
        print("=================================================================================\n")
        
        all_results[filename] = results
        
    print("=================================================================================")
    print("📊 COMPREHENSIVE MULTI-REGIME REPORT CARD:")
    print("=================================================================================")
    
    total_pnl = 0.0
    total_fees = 0.0
    total_trades = 0
    
    for name, res in all_results.items():
        total_pnl += res["net_pnl"]
        total_fees += res["total_fees_paid"]
        total_trades += res["total_trades"]
        
        print(f"{name:<50}: Net PnL: ${res['net_pnl']:+.2f} ({res['net_percentage_return']:+.2f}%) | Max DD: {res['max_drawdown']:.2f}%")
        
    print("=================================================================================")
    print(f"Aggregate Net PnL      : ${total_pnl:+.2f}")
    print(f"Aggregate Exchange Fees: ${total_fees:.2f}")
    print(f"Total Trades Across All: {total_trades}")
    print("=================================================================================")
    
    # Save rich performance history to JSON file for LLM parsing
    report_path = "/Users/singhujwal/crypto_mft/backtest_report_summary.json"
    try:
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📝 LLM-Ready Backtest Report saved successfully to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save LLM report: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
