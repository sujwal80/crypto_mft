import json
import os
import sys
import time
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../5_arbitrage")))

from portfolio_selector import PortfolioStatArbSelector
from pairs_trader import CointegratedPairsTrader
from leung_calibrator import LeungThresholdCalibrator

def main():
    input_file = "/Users/singhujwal/crypto_mft/datasets/real_12months_btc_eth.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the downloader first.")
        return
        
    print("=================================================================================")
    print("🚀 Loading 12-Month Historical Database...")
    start_load = time.time()
    with open(input_file, "r") as f:
        aligned_data = json.load(f)
    print(f"Successfully loaded {len(aligned_data):,} price candles in {time.time() - start_load:.2f}s.")
    print("=================================================================================")
    
    # 1. Resample 1-minute candles into 1-hour (60-minute) macro bars
    print("📊 Resampling 1-minute candles into 1-hour (60-minute) macro bars...")
    resampled_data = []
    for i in range(0, len(aligned_data), 60):
        chunk = aligned_data[i:i+60]
        if len(chunk) < 60:
            break
        btc_close = chunk[-1]["btc"]["close"]
        eth_close = chunk[-1]["eth"]["close"]
        btc_vol = sum(c["btc"]["volume"] for c in chunk)
        eth_vol = sum(c["eth"]["volume"] for c in chunk)
        resampled_data.append({
            "date": chunk[-1]["date"],
            "btc": {"close": btc_close, "volume": btc_vol},
            "eth": {"close": eth_close, "volume": eth_vol}
        })
    aligned_data = resampled_data
    print(f"Successfully resampled to {len(aligned_data):,} hourly macro bars.")
    print("=================================================================================")
    
    print("=================================================================================")
    print("🚀 COINTEGRATED PORTFOLIO SELECTOR - 12-MONTH PRODUCTION BACKTEST")
    print("=================================================================================")
    print(f"Timeline: {aligned_data[0]['date']} to {aligned_data[-1]['date']}")
    print("=================================================================================")
    
    # 2. Initialize the Portfolio Selector with a highly reactive 72-hour (3-day) lookback window
    selector = PortfolioStatArbSelector(
        candidate_pairs=[("BTC", "ETH")], 
        window=72, 
        hurst_threshold=0.83
    )
    
    # Initialize a baseline pairs trader to handle execution states (lookback matches selector)
    trader = CointegratedPairsTrader(
        lookback_window=72,
        entry_z=2.0,
        exit_z=0.2,
        stop_loss_z=3.8
    )
    
    # Instantiate Tim Leung's HJB Stochastic Threshold Calibrator
    calibrator = LeungThresholdCalibrator(round_trip_fee=0.0020)
    
    initial_capital = 10000.0
    cash = initial_capital
    total_fees = 0.0
    wins = 0
    completed_trades = []
    active_trade = None
    ticks_processed = 0
    
    # Sizing fraction: 5% allocation
    allocation_fraction = float(os.getenv("ALLOCATION_FRACTION", "0.05"))
    print(f"Position Sizing: {allocation_fraction * 100:.1f}% capital allocation per trade.")
    print("=================================================================================\n")
    
    start_backtest = time.time()
    
    # Stream hourly candles
    for item in aligned_data:
        ticks_processed += 1
        date_str = item["date"]
        btc_price = item["btc"]["close"]
        eth_price = item["eth"]["close"]
        
        # A. Ingest prices into the Portfolio Selector
        selector.ingest_prices({"BTC": btc_price, "ETH": eth_price})
        
        # B. Ingest prices into the execution trader to keep price logs synced
        z_score = trader.ingest_prices(btc_price=btc_price, eth_price=eth_price)
        if z_score is None:
            continue
            
        # C. Query the Portfolio Selector for active coin selection
        selected_pairs = selector.rank_and_select_pairs(max_active_pairs=1)
        is_pair_armed = len(selected_pairs) > 0
        
        # D. Evaluate trade setup (Enforce Hurst regime block using selector's state)
        if not trader.in_position and not is_pair_armed:
            # Cointegration broken / Spread is trending -> Block entries!
            continue
            
        # If pair is selected, we can evaluate triggers using the OLS beta and alpha returned by selector!
        if is_pair_armed and not trader.in_position:
            trader.beta = selected_pairs[0]["beta"]
            trader.alpha = selected_pairs[0]["alpha"]
            
        # E. Dynamic HJB Threshold Calibration (Tim Leung Stochastic Optimal stopping)
        if is_pair_armed and len(trader.spread_history) >= 24:
            spreads = np.array(list(trader.spread_history))
            # Discretize OU process to AR(1) model
            x_ar = spreads[:-1]
            y_ar = spreads[1:]
            a, b = np.polyfit(x_ar, y_ar, 1)
            a_clipped = max(min(a, 0.999), 1e-4)
            lambda_mr = -np.log(a_clipped)
            sigma_spread = float(np.std(spreads))
            
            # Solve HJB optimal thresholds dynamically
            opt_xe, opt_xs, opt_sl = calibrator.calibrate_optimal_thresholds(
                lambda_mr=lambda_mr, 
                sigma=sigma_spread, 
                theta=float(np.mean(spreads))
            )
            
            # Dynamic boundary injection
            trader.entry_z = abs(opt_xe)
            trader.exit_z = opt_xs
            trader.stop_loss_z = abs(opt_sl)
            
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
                print(f"🟢 ENTRY Paired | Date: {date_str} | Z: {z_score:+.2f} | Beta (OLS): {trader.beta:.2f} | Size: {btc_qty:.4f} BTC / {eth_qty:.4f} ETH")
                
            elif cmd["action"] == "EXIT" and active_trade:
                btc_qty_size = active_trade["btc_qty"]
                eth_qty_size = active_trade["eth_qty"]
                exit_fee = (btc_qty_size * btc_price * 0.001) + (eth_qty_size * eth_price * 0.001)  # 0.1% taker fee
                slippage = 50.0 * btc_qty_size
                
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
                    
                print(f"🏁 EXIT Paired  | Date: {date_str} | Z: {z_score:+.2f} | Type: {cmd['type']} | Net PnL: ${net_pnl:+,.2f}")
                print(f"   -> BTC Entry: ${active_trade['btc_entry_price']:.2f} | Exit: ${btc_price:.2f} | PnL: {btc_pnl:+,.2f}")
                print(f"   -> ETH Entry: ${active_trade['eth_entry_price']:.2f} | Exit: ${eth_price:.2f} | PnL: {eth_pnl:+,.2f}")
                active_trade = None

    # Force market close if trade is open at end of dataset
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

    backtest_duration = time.time() - start_backtest
    print(f"\n=================================================================================")
    print(f"✅ BACKTEST COMPLETE: Processed {ticks_processed:,} hourly bars in {backtest_duration:.2f}s.")
    print("=================================================================================")

    # 4. Print Consolidated Cointegrated Stat-Arb Report Card
    print("\n=================================================================================")
    print("📊 12-MONTH COINTEGRATED PORTFOLIO SELECTOR PERFORMANCE CARD")
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
