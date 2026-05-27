import os
import json
import sys
import time
import glob
import logging
import numpy as np
from typing import List, Dict, Optional

# Setup production paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "3_alpha_micro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "4_execution")))

from state_machine import GexMicroStateMachine

# Setup logging
logging.basicConfig(
    level=logging.WARNING, # Suppress low-level info logs during batch runs to prevent console flooding
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OptionL2Backtester")

class OptionL2Backtester:
    """
    High-Fidelity Level 2 Option Contract Backtester.
    Dynamically reconstructs the L2 options order book from Bybit snapshot/delta feeds,
    replays ticks chronologically, and simulates swing trading on the option contract itself.
    """
    def __init__(self, file_path: str, initial_capital: float = 10000.0):
        self.file_path = file_path
        self.initial_capital = initial_capital
        
        basename = os.path.basename(file_path)
        self.symbol = basename.split("_")[-1].replace(".ob25", "")
        
        # Instantiate Master State Machine
        self.state_machine = GexMicroStateMachine(
            symbol=self.symbol,
            mode="SHADOW",
            grace_window_ticks=100,
            resample_ticks=25,
            prioritize_time=False
        )
        
        # Initialize GEX wall profile near option premium boundary to trigger entries
        self.state_machine.update_gex_profile(
            key_strike=13000.0,
            gex_value=150.0,
            deribit_index=13000.0
        )

    def start_simulation(self) -> Dict:
        """Runs L2 chronological replay. Returns backtest results dict."""
        bids_book = {} 
        asks_book = {} 
        
        prev_bid_price = None
        prev_ask_price = None
        prev_bid_qty = 0.0
        prev_ask_qty = 0.0
        
        line_count = 0
        mid_price = 0.0
        
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                        
                    line_count += 1
                    try:
                        packet = json.loads(line)
                    except Exception:
                        continue
                        
                    data = packet.get("data", {})
                    bids_data = data.get("b", [])
                    asks_data = data.get("a", [])
                    
                    # Update reconstructed order book
                    for level in bids_data:
                        price_str = level[0]
                        qty = float(level[1])
                        if qty == 0.0:
                            bids_book.pop(price_str, None)
                        else:
                            bids_book[price_str] = qty
                            
                    for level in asks_data:
                        price_str = level[0]
                        qty = float(level[1])
                        if qty == 0.0:
                            asks_book.pop(price_str, None)
                        else:
                            asks_book[price_str] = qty
                            
                    if not bids_book or not asks_book:
                        continue
                        
                    sorted_bids = sorted([(float(p), q) for p, q in bids_book.items()], key=lambda x: x[0], reverse=True)
                    sorted_asks = sorted([(float(p), q) for p, q in asks_book.items()], key=lambda x: x[0])
                    
                    if not sorted_bids or not sorted_asks:
                        continue
                        
                    best_bid_price = sorted_bids[0][0]
                    best_bid_qty = sorted_bids[0][1]
                    best_ask_price = sorted_asks[0][0]
                    best_ask_qty = sorted_asks[0][1]
                    
                    mid_price = (best_bid_price + best_ask_price) / 2.0
                    tick_ns = int(packet["ts"] * 1e6)
                    
                    is_inferred_trade = False
                    trade_price = 0.0
                    trade_qty = 0.0
                    is_buyer_maker = False
                    
                    if prev_bid_price is not None and prev_ask_price is not None:
                        if best_bid_price == prev_bid_price and best_bid_qty < prev_bid_qty:
                            is_inferred_trade = True
                            trade_price = best_bid_price
                            trade_qty = prev_bid_qty - best_bid_qty
                            is_buyer_maker = True
                        elif best_ask_price == prev_ask_price and best_ask_qty < prev_ask_qty:
                            is_inferred_trade = True
                            trade_price = best_ask_price
                            trade_qty = prev_ask_qty - best_ask_qty
                            is_buyer_maker = False
                            
                    prev_bid_price = best_bid_price
                    prev_ask_price = best_ask_price
                    prev_bid_qty = best_bid_qty
                    prev_ask_qty = best_ask_qty
                    
                    # Process tick
                    import asyncio
                    asyncio.run(self.state_machine.process_market_tick(
                        mid_price=mid_price,
                        bid_qty=best_bid_qty,
                        ask_qty=best_ask_qty,
                        is_trade=is_inferred_trade,
                        trade_price=trade_price,
                        trade_qty=trade_qty,
                        is_buyer_maker=is_buyer_maker,
                        timestamp_ns=tick_ns
                    ))
                    
            # Force close open swing positions
            if self.state_machine.in_position and mid_price > 0.0:
                exit_side = "SELL" if self.state_machine.position_side == "LONG" else "BUY"
                asyncio.run(self.state_machine.smart_router.place_market_order(
                    symbol=self.symbol,
                    side=exit_side,
                    amount=1.0,
                    shadow_price=mid_price
                ))
                self.state_machine.in_position = False
                
        except Exception as e:
            logger.error(f"Error processing {self.symbol}: {e}")
            
        # Compile result dictionary
        journal = self.state_machine.smart_router.trade_journal
        return {
            "symbol": self.symbol,
            "ticks_processed": line_count,
            "journal": journal
        }

# --------------------------------------------------------------------------------
# Multi-Contract Portfolio Batch Backtest Runner
# --------------------------------------------------------------------------------
def run_portfolio_backtest(expiry_filter: Optional[str] = None, 
                           strike_filter: Optional[float] = None, 
                           type_filter: Optional[str] = None):
    """
    Scans all 5 ob25 datasets directories, filters contracts, 
    runs chronological L2 backtests, and consolidates portfolio performance.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))
    ob25_dirs = sorted([d for d in os.listdir(base_dir) if d.endswith("_BTC_USDT.ob25") and os.path.isdir(os.path.join(base_dir, d))], reverse=True)
    
    all_files = []
    for d in ob25_dirs:
        all_files.extend(glob.glob(os.path.join(base_dir, d, "*.ob25")))
        
    filtered_files = []
    
    # Apply dynamic filters
    for f in all_files:
        basename = os.path.basename(f)
        if "_BTC" not in basename:
            continue
            
        try:
            contract_suffix = basename.split("_BTC")[1]
            parts = contract_suffix.split("-")
            
            expiry = parts[1]
            strike = float(parts[2])
            opt_type = parts[3]
            
            if expiry_filter and expiry_filter not in expiry:
                continue
            if strike_filter and strike_filter != strike:
                continue
            if type_filter and type_filter != opt_type:
                continue
                
            filtered_files.append(f)
        except Exception:
            continue
            
    total_matched = len(filtered_files)
    
    print("================================================================================")
    print("💼 MULTI-CONTRACT L2 OPTIONS PORTFOLIO BACKTEST RUNNER")
    print("================================================================================")
    print(f"   • Match Filters : Expiry={expiry_filter or 'ALL'} | Strike={strike_filter or 'ALL'} | Type={type_filter or 'ALL'}")
    print(f"   • Matched Files : {total_matched} active contracts (from 5-day historical folders)")
    print("================================================================================\n")
    
    if total_matched == 0:
        print("⚠️ No option contracts matched your filters. Aborting run.")
        return
        
    initial_capital_per_contract = 10000.0
    total_initial_capital = total_matched * initial_capital_per_contract
    total_final_balance = total_initial_capital
    total_completed_trades = 0
    total_wins = 0
    aggregate_fees = 0.0
    
    trade_records = []
    
    # Sequential batch execution
    for idx, f in enumerate(filtered_files):
        sys.stdout.write(f"\r⚙️ Running Contract Backtest [{idx+1}/{total_matched}]: {os.path.basename(f)}...")
        sys.stdout.flush()
        
        backtester = OptionL2Backtester(file_path=f, initial_capital=initial_capital_per_contract)
        result = backtester.start_simulation()
        
        journal = result["journal"]
        active_pos = None
        
        for fill in journal:
            aggregate_fees += fill["fee"]
            
            if not active_pos:
                active_pos = {
                    "entry_price": fill["price"],
                    "entry_fee": fill["fee"],
                    "side": "LONG" if fill["side"] == "BUY" else "SHORT",
                    "symbol": result["symbol"]
                }
            else:
                entry = active_pos["entry_price"]
                exit = fill["price"]
                side = active_pos["side"]
                
                gross_pnl = (exit - entry) if side == "LONG" else (entry - exit)
                net_pnl = gross_pnl - (active_pos["entry_fee"] + fill["fee"])
                
                trade_records.append({
                    "symbol": active_pos["symbol"],
                    "side": side,
                    "entry": entry,
                    "exit": exit,
                    "net_pnl": net_pnl
                })
                
                total_final_balance += net_pnl
                total_completed_trades += 1
                if net_pnl > 0:
                    total_wins += 1
                active_pos = None
                
    print("\n\n================================================================================")
    print("📊 CONSOLIDATED 5-DAY L2 OPTIONS PORTFOLIO REPORT CARD")
    print("================================================================================")
    print(f"Total Contracts Tested : {total_matched}")
    print(f"Total Completed Trades : {total_completed_trades}")
    
    if total_completed_trades > 0:
        win_rate = (total_wins / total_completed_trades) * 100.0
        print(f"Portfolio Win Rate     : {win_rate:.2f}%")
    else:
        print("Portfolio Win Rate     : 0.00%")
        
    print(f"Aggregate Fees Paid    : ${aggregate_fees:.2f}")
    print(f"Initial Portfolio Cap  : ${total_initial_capital:,.2f}")
    print(f"Final Portfolio Balance: ${total_final_balance:,.2f}")
    
    net_percentage = ((total_final_balance - total_initial_capital) / total_initial_capital) * 100.0
    print(f"Net Portfolio Return   : ${total_final_balance - total_initial_capital:+.2f} ({net_percentage:+.2f}%)")
    print("================================================================================\n")
    
    # Print top executed trade highlights if any occurred
    if trade_records:
        print("📝 EXECUTED SWING TRADE RECORD HIGHLIGHTS (Sorted by PnL):")
        sorted_trades = sorted(trade_records, key=lambda x: x["net_pnl"], reverse=True)
        for i, t in enumerate(sorted_trades[:10]): # Limit to top 10 highlights
            print(f"   Highlight {i+1}: {t['symbol']} | {t['side']} | Entry: ${t['entry']:.2f} | Exit: ${t['exit']:.2f} | Net PnL: {t['net_pnl']:+.2f}")
        print("================================================================================\n")

if __name__ == "__main__":
    # Quick programmatic focus: Backtest all $64,000 strike Call/Put contracts across the 5 days!
    # (Perfect for fast validation testing over specific high-probability strikes!)
    run_portfolio_backtest(
        expiry_filter="5JUN26",
        strike_filter=64000.0
    )
