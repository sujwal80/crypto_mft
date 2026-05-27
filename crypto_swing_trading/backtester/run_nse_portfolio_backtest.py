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
    input_file = "/Users/singhujwal/crypto_mft/datasets/nse_tcs_infy_1year.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the download_nse_data.py script first.")
        return
        
    print("=================================================================================")
    print("🚀 Loading 1-Year Indian NSE Database (TCS vs INFY)...")
    start_load = time.time()
    with open(input_file, "r") as f:
        aligned_data = json.load(f)
    print(f"Successfully loaded {len(aligned_data):,} hourly bars in {time.time() - start_load:.2f}s.")
    print("=================================================================================")
    
    print("=================================================================================")
    print("🚀 INDIAN STOCK MARKET STATISTICAL ARBITRAGE - 1-YEAR HOURLY BACKTEST")
    print("=================================================================================")
    print(f"Assets   : TCS.NS vs. INFY.NS (Tata Consultancy Services vs Infosys)")
    print(f"Timeline : {aligned_data[0]['date']} to {aligned_data[-1]['date']}")
    print("=================================================================================")
    
    # 1. Initialize the Portfolio Selector with a highly reactive 72-hour (approx. 11 trading days) lookback
    selector = PortfolioStatArbSelector(
        candidate_pairs=[("TCS", "INFY")], 
        window=72, 
        hurst_threshold=0.83
    )
    
    # Initialize the dynamic pairs trader (lookback window matches selector)
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
    
    # Stream hourly NSE candles
    for item in aligned_data:
        ticks_processed += 1
        date_str = item["date"]
        tcs_price = item["tcs"]["close"]
        infy_price = item["infy"]["close"]
        
        # A. Ingest prices into the Portfolio Selector
        selector.ingest_prices({"TCS": tcs_price, "INFY": infy_price})
        
        # B. Ingest prices into the execution trader to keep price logs synced
        z_score = trader.ingest_prices(btc_price=tcs_price, eth_price=infy_price)
        if z_score is None:
            continue
            
        # C. Query the Portfolio Selector for active coin selection
        selected_pairs = selector.rank_and_select_pairs(max_active_pairs=1)
        is_pair_armed = len(selected_pairs) > 0
        
        # D. Evaluate trade setup (Enforce Hurst regime block using selector's state)
        if not trader.in_position and not is_pair_armed:
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
            a_clipped = max(min(a, 0.99), 1e-4)
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
            
        cmd = trader.evaluate_trade_setup(btc_price=tcs_price, eth_price=infy_price, z_score=z_score, capital=cash)
        
        if cmd:
            if cmd["action"] == "ENTRY":
                # In equity markets, shares are rounded to whole numbers
                tcs_qty = int(cmd["btc_order"]["qty"])
                infy_qty = int(cmd["eth_order"]["qty"])
                
                # Enforce minimum execution bounds
                tcs_qty = max(tcs_qty, 1)
                infy_qty = max(infy_qty, 1)
                
                entry_fee = (tcs_qty * tcs_price * 0.001) + (infy_qty * infy_price * 0.001)  # 0.1% one-way fee
                cash -= entry_fee
                total_fees += entry_fee
                
                active_trade = {
                    "type": cmd["type"],
                    "entry_date": date_str,
                    "btc_entry_price": tcs_price,
                    "eth_entry_price": infy_price,
                    "btc_qty": tcs_qty,
                    "eth_qty": infy_qty,
                    "entry_fee": entry_fee,
                    "beta": trader.beta
                }
                print(f"🟢 ENTRY Paired | Date: {date_str} | Z: {z_score:+.2f} | Beta (OLS): {trader.beta:.2f} | Size: {tcs_qty} TCS / {infy_qty} INFY")
                
            elif cmd["action"] == "EXIT" and active_trade:
                tcs_qty_size = active_trade["btc_qty"]
                infy_qty_size = active_trade["eth_qty"]
                exit_fee = (tcs_qty_size * tcs_price * 0.001) + (infy_qty_size * infy_price * 0.001)  # 0.1% exit fee
                slippage = 2.0 * tcs_qty_size  # Modest 2 Rupee slippage per TCS share
                
                btc_pnl = tcs_qty_size * ((active_trade["btc_entry_price"] - tcs_price) if active_trade["type"] == "SHORT_SPREAD" else (tcs_price - active_trade["btc_entry_price"]))
                eth_pnl = infy_qty_size * ((infy_price - active_trade["eth_entry_price"]) if active_trade["type"] == "SHORT_SPREAD" else (active_trade["eth_entry_price"] - infy_price))
                gross_pnl = btc_pnl + eth_pnl
                net_pnl = gross_pnl - (active_trade["entry_fee"] + exit_fee + slippage)
                
                cash += gross_pnl - exit_fee - slippage
                total_fees += exit_fee
                
                active_trade["exit_date"] = date_str
                active_trade["net_pnl"] = net_pnl
                
                completed_trades.append(active_trade)
                
                if net_pnl > 0:
                    wins += 1
                    
                print(f"🏁 EXIT Paired  | Date: {date_str} | Z: {z_score:+.2f} | Type: {cmd['type']} | Net PnL: ₹{net_pnl:+,.2f}")
                print(f"   -> TCS Entry: ₹{active_trade['btc_entry_price']:.2f} | Exit: ₹{tcs_price:.2f} | PnL: ₹{btc_pnl:+,.2f}")
                print(f"   -> INFY Entry: ₹{active_trade['eth_entry_price']:.2f} | Exit: ₹{infy_price:.2f} | PnL: ₹{eth_pnl:+,.2f}")
                active_trade = None

    # Force market close if trade is open at end of dataset
    if active_trade:
        final_item = aligned_data[-1]
        tcs_price = final_item["tcs"]["close"]
        infy_price = final_item["infy"]["close"]
        tcs_qty_size = active_trade["btc_qty"]
        infy_qty_size = active_trade["eth_qty"]
        
        exit_fee = (tcs_qty_size * tcs_price * 0.001) + (infy_qty_size * infy_price * 0.001)
        slippage = 2.0 * tcs_qty_size
        
        btc_pnl = tcs_qty_size * ((active_trade["btc_entry_price"] - tcs_price) if active_trade["type"] == "SHORT_SPREAD" else (tcs_price - active_trade["btc_entry_price"]))
        eth_pnl = infy_qty_size * ((infy_price - active_trade["eth_entry_price"]) if active_trade["type"] == "SHORT_SPREAD" else (active_trade["eth_entry_price"] - infy_price))
        gross_pnl = btc_pnl + eth_pnl
        net_pnl = gross_pnl - (active_trade["entry_fee"] + exit_fee + slippage)
        
        cash += gross_pnl - exit_fee - slippage
        total_fees += exit_fee
        
        active_trade["exit_date"] = final_item["date"]
        active_trade["net_pnl"] = net_pnl
        completed_trades.append(active_trade)
        print(f"🏁 FORCE EXIT  | Date: {final_item['date']} | Close Price | Net PnL: ₹{net_pnl:+,.2f} (Dataset Ended)")

    backtest_duration = time.time() - start_backtest
    print(f"\n=================================================================================")
    print(f"✅ BACKTEST COMPLETE: Processed {ticks_processed:,} hourly bars in {backtest_duration:.2f}s.")
    print("=================================================================================")

    # 4. Print Consolidated Cointegrated Stat-Arb Report Card
    print("\n=================================================================================")
    print("📊 1-YEAR INDIAN NSE COINTEGRATED ARBITRAGE PERFORMANCE CARD")
    print("=================================================================================")
    total_trades = len(completed_trades)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    net_pnl = cash - initial_capital
    net_return = (net_pnl / initial_capital) * 100.0
    
    print(f"Initial Arbitrage Capital: ₹10,000.00")
    print(f"Final Arbitrage Capital  : ₹{cash:,.2f}")
    print(f"Net Realized Stat PnL    : ₹{net_pnl:+,.2f} ({net_return:+.2f}%)")
    print(f"Total Trades Executed    : {total_trades}")
    print(f"Wins / Losses            : {wins} W / {losses} L")
    print(f"Realized Win Rate        : {win_rate:.2f}%")
    print(f"Total Exchange Fees Paid : ₹{total_fees:,.2f}")
    print(f"Total Slippage Friction  : ₹{sum(2.0 * t['btc_qty'] for t in completed_trades):,.2f}")
    print("=================================================================================")

if __name__ == "__main__":
    main()
