import json
import os
import sys
import time
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../5_arbitrage")))

from pairs_trader import CointegratedPairsTrader

def main():
    input_file = "/Users/singhujwal/crypto_mft/datasets/real_12months_btc_eth.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the 12-month downloader first.")
        return
        
    with open(input_file, "r") as f:
        aligned_data = json.load(f)
        
    # Extract exactly the last 5 days (7,200 1-minute candles) from the master database
    aligned_data = aligned_data[-7200:]
        
    print("=================================================================================")
    print("🚀 COINTEGRATED PAIRS TRADING - LIVE 5-DAY MARKET BACKTEST (1-MINUTE DATA)")
    print("=================================================================================")
    print(f"Dataset Loaded: {len(aligned_data)} aligned 1-minute price candles.")
    print(f"Timeline      : {aligned_data[0]['date']} to {aligned_data[-1]['date']}")
    print("=================================================================================")
    
    # Initialize Pairs Trader (120-minute rolling regression window)
    # Initialize Pairs Trader with robust 120-minute rolling window
    trader = CointegratedPairsTrader(
        lookback_window=120,  # 120 minutes rolling OLS
        entry_z=2.0,
        exit_z=0.2,
        stop_loss_z=4.5
    )
    
    initial_capital = 10000.0
    cash = initial_capital
    total_fees = 0.0
    wins = 0
    completed_trades = []
    
    active_trade = None
    ticks_processed = 0
    
    # Read allocation fraction from env variable (default to 5%)
    allocation_fraction = float(os.getenv("ALLOCATION_FRACTION", "0.05"))
    print(f"Position Sizing: {allocation_fraction * 100:.1f}% capital allocation per trade.")
    print("=================================================================================\n")
    
    # Stream 1-minute candles tick-by-tick
    for item in aligned_data:
        ticks_processed += 1
        date_str = item["date"]
        btc_price = item["btc"]["close"]
        eth_price = item["eth"]["close"]
        
        # Ingest prices
        z_score = trader.ingest_prices(btc_price=btc_price, eth_price=eth_price)
        if z_score is None:
            # Warming up rolling OLS
            continue
            
        # Evaluate trade setup
        cmd = trader.evaluate_trade_setup(btc_price=btc_price, eth_price=eth_price, z_score=z_score, capital=cash)
        
        if cmd:
            if cmd["action"] == "ENTRY":
                btc_qty = cmd["btc_order"]["qty"]
                eth_qty = cmd["eth_order"]["qty"]
                entry_fee = (btc_qty * btc_price * 0.001) + (eth_qty * eth_price * 0.001)  # 0.1% maker fee
                cash -= entry_fee
                total_fees += entry_fee
                
                active_trade = {
                    "type": cmd["type"],
                    "entry_date": date_str,
                    "btc_entry_price": btc_price,
                    "eth_entry_price": eth_price,
                    "btc_qty": btc_qty,
                    "eth_qty": eth_qty,
                    "entry_fee": entry_fee,
                    "beta": trader.beta
                }
                print(f"🟢 ENTRY Paired | Date: {date_str} | Z-score: {z_score:+.2f} | Beta: {trader.beta:.2f} | Size: {btc_qty:.4f} BTC / {eth_qty:.4f} ETH")
                
            elif cmd["action"] == "EXIT" and active_trade:
                btc_qty_size = active_trade["btc_qty"]
                eth_qty_size = active_trade["eth_qty"]
                exit_fee = (btc_qty_size * btc_price * 0.001) + (eth_qty_size * eth_price * 0.001)  # 0.1% taker fee
                slippage = 50.0 * btc_qty_size  # Paired slippage scaled by position size
                
                # Re-calculate final trade PnL
                btc_pnl = btc_qty_size * ((active_trade["btc_entry_price"] - btc_price) if active_trade["type"] == "SHORT_SPREAD" else (btc_price - active_trade["btc_entry_price"]))
                eth_pnl = eth_qty_size * ((eth_price - active_trade["eth_entry_price"]) if active_trade["type"] == "SHORT_SPREAD" else (active_trade["eth_entry_price"] - eth_price))
                gross_pnl = btc_pnl + eth_pnl
                net_pnl = gross_pnl - (active_trade["entry_fee"] + exit_fee + slippage)
                
                cash += gross_pnl - exit_fee - slippage
                total_fees += exit_fee
                
                active_trade["exit_date"] = date_str
                active_trade["net_pnl"] = net_pnl
                completed_trades.append(active_trade)
                
                if net_pnl > 0:
                    wins += 1
                    
                print(f"🏁 EXIT Paired  | Date: {date_str} | Z-score: {z_score:+.2f} | Type: {cmd['type']} | Net PnL: ${net_pnl:+,.2f}")
                print(f"   -> BTC Entry: ${active_trade['btc_entry_price']:.2f} | Exit: ${btc_price:.2f} | PnL: {btc_pnl:+,.2f}")
                print(f"   -> ETH Entry: ${active_trade['eth_entry_price']:.2f} | Exit: ${eth_price:.2f} | PnL: {eth_pnl:+,.2f}")
                active_trade = None

    # If position is open at end of dataset, force market close
    if active_trade:
        final_item = aligned_data[-1]
        btc_price = final_item["btc"]["close"]
        eth_price = final_item["eth"]["close"]
        btc_qty_size = active_trade["btc_qty"]
        eth_qty_size = active_trade["eth_qty"]
        
        exit_fee = (btc_qty_size * btc_price * 0.001) + (eth_qty_size * eth_price * 0.001)
        slippage = 50.0 * btc_qty_size
        
        btc_pnl = btc_qty_size * ((active_trade["btc_entry_price"] - btc_price) if active_trade["type"] == "SHORT_SPREAD" else (btc_price - active_trade["btc_entry_price"]))
        eth_pnl = eth_qty_size * ((eth_price - active_trade["eth_entry_price"]) if active_trade["type"] == "SHORT_SPREAD" else (active_trade["eth_entry_price"] - eth_price))
        gross_pnl = btc_pnl + eth_pnl
        net_pnl = gross_pnl - (active_trade["entry_fee"] + exit_fee + slippage)
        
        cash += gross_pnl - exit_fee - slippage
        total_fees += exit_fee
        
        active_trade["exit_date"] = final_item["date"]
        active_trade["net_pnl"] = net_pnl
        completed_trades.append(active_trade)
        print(f"🏁 FORCE EXIT  | Date: {final_item['date']} | Close Price | Net PnL: ${net_pnl:+,.2f} (Dataset Ended)")

    # 4. Print Consolidated Cointegrated Stat-Arb Report Card
    print("\n=================================================================================")
    print("📊 LIVE 5-DAY COINTEGRATED STATISTICAL ARBITRAGE PERFORMANCE CARD")
    print("=================================================================================")
    total_trades = len(completed_trades)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    net_pnl = cash - initial_capital
    net_return = (net_pnl / initial_capital) * 100.0
    
    print(f"Initial Arbitrage Capital: $10,000.00")
    print(f"Final Arbitrage Capital  : ${cash:,.2f}")
    print(f"Net Realized Stat PnL    : ${net_pnl:+,.2f} ({net_return:+.2f}%)")
    print(f"Total Trades Executed    : {total_trades}")
    print(f"Wins / Losses            : {wins} W / {losses} L")
    print(f"Realized Win Rate        : {win_rate:.2f}%")
    print(f"Total Exchange Fees Paid : ${total_fees:,.2f}")
    print(f"Total Slippage Friction  : ${sum(50.0 * t['btc_qty'] for t in completed_trades):,.2f}")
    print("=================================================================================")

if __name__ == "__main__":
    main()
