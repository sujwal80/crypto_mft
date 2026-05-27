import urllib.request
import json
import time
import os
import sys
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../5_arbitrage")))

from pairs_trader import CointegratedPairsTrader

def fetch_daily_klines(symbol: str, limit: int = 60) -> list:
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1d&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching {symbol} data: {e}")
        return []

def main():
    print("=================================================================================")
    # Professional Quant Header
    print("🚀 COINTEGRATED PAIRS TRADING ENGINE - HISTORICAL REAL MARKET BACKTEST")
    print("=================================================================================")
    
    # 1. Fetch 60 days of real historical daily closes for BTC and ETH
    btc_raw = fetch_daily_klines("BTCUSDT", limit=60)
    eth_raw = fetch_daily_klines("ETHUSDT", limit=60)
    
    if not btc_raw or not eth_raw:
        print("Error: Failed to retrieve historical price feeds from Binance.")
        return
        
    # Align datasets by timestamp
    btc_dict = {k[0]: float(k[4]) for k in btc_raw}  # close price by open time
    eth_dict = {k[0]: float(k[4]) for k in eth_raw}
    
    aligned_timestamps = sorted(list(set(btc_dict.keys()).intersection(set(eth_dict.keys()))))
    
    print(f"Successfully aligned {len(aligned_timestamps)} days of real historical BTC & ETH price candles.")
    
    # 2. Initialize Pairs Trader (14-day lookback window for OLS regression)
    trader = CointegratedPairsTrader(
        lookback_window=14,
        entry_z=2.0,
        exit_z=0.2,
        stop_loss_z=3.0
    )
    
    initial_capital = 10000.0
    cash = initial_capital
    total_fees = 0.0
    wins = 0
    completed_trades = []
    
    # Execution tracking
    active_trade = None
    
    # 3. Stream aligned prices day-by-day into the Cointegrated Engine
    for ts in aligned_timestamps:
        date_str = time.strftime('%Y-%m-%d', time.localtime(ts/1000))
        btc_price = btc_dict[ts]
        eth_price = eth_dict[ts]
        
        # Ingest prices and fit cointegration model
        z_score = trader.ingest_prices(btc_price=btc_price, eth_price=eth_price)
        if z_score is None:
            # Warming up the rolling OLS window
            continue
            
        # Evaluate trade setup with current portfolio capital cash
        cmd = trader.evaluate_trade_setup(btc_price=btc_price, eth_price=eth_price, z_score=z_score, capital=cash)
        
        if cmd:
            if cmd["action"] == "ENTRY":
                # Paired entry execution
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
                print(f"🟢 ENTRY Pailed | Date: {date_str} | Z-score: {z_score:+.2f} | Beta: {trader.beta:.2f} | Trade Size (5%): {btc_qty:.4f} BTC (${btc_qty*btc_price:.2f}) / {eth_qty:.4f} ETH (${eth_qty*eth_price:.2f})")
                
            elif cmd["action"] == "EXIT" and active_trade:
                # Paired exit execution
                btc_qty_size = active_trade["btc_qty"]
                eth_qty_size = active_trade["eth_qty"]
                exit_fee = (btc_qty_size * btc_price * 0.001) + (eth_qty_size * eth_price * 0.001)  # 0.1% taker fee
                # Scale slippage proportionally with position size (baseline $50 for 1.0 BTC)
                slippage = 50.0 * btc_qty_size
                
                # Re-calculate final trade PnL using frozen entry quantities
                btc_pnl = btc_qty_size * ((active_trade["btc_entry_price"] - btc_price) if active_trade["type"] == "SHORT_SPREAD" else (btc_price - active_trade["btc_entry_price"]))
                eth_pnl = eth_qty_size * ((eth_price - active_trade["eth_entry_price"]) if active_trade["type"] == "SHORT_SPREAD" else (active_trade["eth_entry_price"] - eth_price))
                gross_pnl = btc_pnl + eth_pnl
                net_pnl = gross_pnl - (active_trade["entry_fee"] + exit_fee + slippage)
                
                cash += gross_pnl - exit_fee - slippage
                total_fees += exit_fee
                
                active_trade["exit_date"] = date_str
                active_trade["exit_price"] = f"BTC: ${btc_price:.2f} / ETH: ${eth_price:.2f}"
                active_trade["exit_type"] = cmd["type"]
                active_trade["net_pnl"] = net_pnl
                
                completed_trades.append(active_trade)
                
                if net_pnl > 0:
                    wins += 1
                    
                print(f"🏁 EXIT Paired  | Date: {date_str} | Z-score: {z_score:+.2f} | Type: {cmd['type']} | Net PnL: ${net_pnl:+,.2f}")
                print(f"   -> BTC Entry: ${active_trade['btc_entry_price']:.2f} | Exit: ${btc_price:.2f} | Size: {btc_qty_size:.4f} | PnL: {btc_pnl:+,.2f}")
                print(f"   -> ETH Entry: ${active_trade['eth_entry_price']:.2f} | Exit: ${eth_price:.2f} | Size: {eth_qty_size:.4f} | PnL: {eth_pnl:+,.2f}")
                active_trade = None

    # 4. Print Consolidated Cointegrated Stat-Arb Report Card
    print("\n=================================================================================")
    print("📊 CONSOLIDATED STATISTICAL ARBITRAGE PERFORMANCE REPORT")
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
