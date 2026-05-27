import json
import os
import sys
import time
import numpy as np
from typing import Dict, List

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../layer_2_macro")))

from vwap_bands import IntradayVwapBands
from hmm_regime import GaussianHMMClassifier

def main():
    input_file = "/Users/singhujwal/crypto_mft/indian_intraday_system/data/5min_60days.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the download_reliance_data.py script first.")
        return
        
    with open(input_file, "r") as f:
        candles = json.load(f)
        
    print("=================================================================================")
    print("🚀 INTRADAY VWAP ENVELOPE STRATEGY - 60-DAY SIMULATION")
    print("=================================================================================")
    print(f"Asset    : RELIANCE.NS (Reliance Industries)")
    print(f"Timeline : {candles[0]['date']} to {candles[-1]['date']}")
    print(f"Total data: {len(candles):,} high-resolution 5-minute candles.")
    print("=================================================================================\n")
    
    # 1. Group candles by trading day (Date yyyy-mm-dd) to handle session resets
    daily_candles: Dict[str, List[Dict]] = {}
    for c in candles:
        date_str = c["date"].split(" ")[0]  # Extract yyyy-mm-dd
        if date_str not in daily_candles:
            daily_candles[date_str] = []
        daily_candles[date_str].append(c)
        
    # Strategy Core Engines
    vwap_engine = IntradayVwapBands()
    hmm_engine = GaussianHMMClassifier()
    
    initial_capital = 100000.0  # ₹1 Lakh testing capital
    cash = initial_capital
    allocation_fraction = 0.50  # Optimized 50% portfolio value per trade
    
    total_trades = 0
    wins = 0
    total_fees = 0.0
    total_slippage = 0.0
    trade_log = []
    
    # Loop daily sessions sequentially
    for date_day, day_bars in sorted(daily_candles.items()):
        # Reset VWAP and HMM running accumulators on new day open (9:15 AM)
        vwap_engine.reset_session()
        hmm_engine.alpha = np.array([0.34, 0.33, 0.33])
        hmm_engine.samples_count = 0
        
        in_position = False
        position_type = None
        entry_price = 0.0
        qty = 0
        
        # Stream the day's 5-minute candles
        for idx, bar in enumerate(day_bars):
            price = bar["close"]
            volume = bar["volume"]
            
            # Update recursive bands
            res = vwap_engine.update(price=price, volume=volume)
            if res is None or vwap_engine.ticks_count < 9:
                # Warm up for the first 45 minutes of the session (9 bars of 5-min = 45 minutes)
                continue
                
            z_score = res["z_score"]
            
            # Calculate rolling 1-hour price trend slope (last 12 bars of 5-min = 60 minutes)
            price_trend = 0.0
            if idx >= 12:
                y_trend = np.array([b["close"] for b in day_bars[idx-12:idx]])
                x_trend = np.arange(12)
                slope, _ = np.polyfit(x_trend, y_trend, 1)
                price_trend = slope / price
                
            # Query HMM Volatility Regime
            active_regime = hmm_engine.classify_tick(z_score=z_score, spread_trend=price_trend)
            
            # State 2 = Decoupling -> Block all entries!
            if not in_position and active_regime == 2:
                continue
                
            is_ranging = (active_regime == 0)
            
            if not in_position:
                # --- ENTRY SIGNALS (Allowed strictly in HMM Range consolidations) ---
                if is_ranging:
                    if z_score <= -2.2:
                        # Short breakdown entry: riding downward momentum
                        in_position = True
                        position_type = "SHORT"
                        entry_price = price
                        qty = max(int((allocation_fraction * cash) / price), 1)
                        
                        entry_fee = qty * price * 0.001  # 0.1% broker fee
                        cash -= entry_fee
                        total_fees += entry_fee
                        
                        active_trade = {
                            "date": bar["date"],
                            "type": "SHORT",
                            "entry_price": price,
                            "qty": qty,
                            "entry_fee": entry_fee
                        }
                        
                    elif z_score >= 2.2:
                        # Long breakout entry: riding upward momentum
                        in_position = True
                        position_type = "LONG"
                        entry_price = price
                        qty = max(int((allocation_fraction * cash) / price), 1)
                        
                        entry_fee = qty * price * 0.001
                        cash -= entry_fee
                        total_fees += entry_fee
                        
                        active_trade = {
                            "date": bar["date"],
                            "type": "LONG",
                            "entry_price": price,
                            "qty": qty,
                            "entry_fee": entry_fee
                        }
            else:
                # --- ACTIVE POSITION MANAGEMENT (EXITS) ---
                should_exit = False
                exit_type = None
                
                # HMM Emergency Liquidation Stop: if spread enters Decoupling state, instantly exit!
                active_regime_exit = hmm_engine.classify_tick(z_score=z_score, spread_trend=price_trend)
                if active_regime_exit == 2:
                    should_exit = True
                    exit_type = "STOP_LOSS"
                    
                # Trend-Following Exits:
                # Take profit when momentum extends to +3.5 or -3.5 standard deviations
                elif position_type == "LONG" and z_score >= 3.5:
                    should_exit = True
                    exit_type = "TAKE_PROFIT"
                elif position_type == "SHORT" and z_score <= -3.5:
                    should_exit = True
                    exit_type = "TAKE_PROFIT"
                    
                # Tight Dynamic Stop Loss: exit immediately if breakout pulls back by 0.5 standard deviations
                elif position_type == "LONG" and z_score <= 1.7:
                    should_exit = True
                    exit_type = "STOP_LOSS"
                elif position_type == "SHORT" and z_score >= -1.7:
                    should_exit = True
                    exit_type = "STOP_LOSS"
                    
                # Force EOD (End of Day) Square Off at the last 5-minute candle (3:25 PM IST)
                elif idx == len(day_bars) - 1:
                    should_exit = True
                    exit_type = "EOD_CLOSE"
                    
                if should_exit:
                    exit_fee = qty * price * 0.001
                    trade_slippage = 2.0 * qty  # ₹2.00 slippage per Reliance share
                    
                    pnl = qty * ((entry_price - price) if position_type == "SHORT" else (price - entry_price))
                    net_pnl = pnl - (active_trade["entry_fee"] + exit_fee + trade_slippage)
                    
                    cash += pnl - exit_fee - trade_slippage
                    total_fees += exit_fee
                    total_slippage += trade_slippage
                    
                    active_trade["exit_price"] = price
                    active_trade["net_pnl"] = net_pnl
                    active_trade["exit_type"] = exit_type
                    trade_log.append(active_trade)
                    
                    total_trades += 1
                    if net_pnl > 0:
                        wins += 1
                        
                    in_position = False
                    position_type = None
                    qty = 0
                    
    # Print Backtest Performance Card
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    net_pnl = cash - initial_capital
    net_return = (net_pnl / initial_capital) * 100.0
    
    print("=================================================================================")
    print("📊 INTRADAY VWAP STRATEGY PERFORMANCE REPORT CARD")
    print("=================================================================================")
    print(f"Initial Capital          : ₹{initial_capital:,.2f}")
    print(f"Final Capital            : ₹{cash:,.2f}")
    print(f"Net Portfolio PnL        : ₹{net_pnl:+,.2f} ({net_return:+.2f}%)")
    print(f"Total Trades Executed    : {total_trades}")
    print(f"Wins / Losses            : {wins} W / {losses} L")
    print(f"Realized Win Rate        : {win_rate:.2f}%")
    print(f"Total Exchange Fees Paid : ₹{total_fees:,.2f}")
    print(f"Total Slippage Friction  : ₹{total_slippage:,.2f}")
    print("=================================================================================")
    
    if total_trades > 0:
        net_pnls = [t["net_pnl"] for t in trade_log]
        print(f"Average Trade Net PnL    : ₹{np.mean(net_pnls):+,.2f}")
        print(f"Largest Win              : ₹{max(net_pnls):+,.2f}")
        print(f"Largest Loss             : ₹{min(net_pnls):+,.2f}")
        print("=================================================================================\n")

if __name__ == "__main__":
    main()
