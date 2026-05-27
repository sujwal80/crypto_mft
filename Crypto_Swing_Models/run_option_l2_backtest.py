import os
import json
import sys
import time
import logging
import numpy as np
from typing import Optional

# Setup production paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "3_alpha_micro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "4_execution")))

from state_machine import GexMicroStateMachine

# Setup logging to stdout
logging.basicConfig(
    level=logging.INFO,
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
        
        # Extract symbol from filename
        basename = os.path.basename(file_path)
        # Example: 2026-05-26_BTC-5JUN26-64000-C-USDT.ob25
        self.symbol = basename.split("_")[-1].replace(".ob25", "")
        
        logger.info(f"Initializing L2 Option Backtester for contract: {self.symbol}")
        
        # Instantiate Master State Machine (Setting prioritize_time=False to enforce tick-based simulation boundaries)
        self.state_machine = GexMicroStateMachine(
            symbol=self.symbol,
            mode="SHADOW",
            grace_window_ticks=100,
            resample_ticks=25,
            prioritize_time=False
        )
        
        # Initialize static GEX wall profile just below/above the option premium boundary to trigger snipers
        self.state_machine.update_gex_profile(
            key_strike=13000.0,
            gex_value=150.0,
            deribit_index=13000.0
        )

    def start_simulation(self):
        logger.warning("================================================================================")
        logger.warning(f"🚀 STARTING LEVEL 2 OPTION CONTRACT BACKTEST SIMULATION")
        logger.warning(f"Contract File: {os.path.basename(self.file_path)}")
        logger.warning("================================================================================")
        
        # Local reconstructed L2 order book state
        bids_book = {} # price_str -> size_float
        asks_book = {} # price_str -> size_float
        
        prev_bid_price = None
        prev_ask_price = None
        prev_bid_qty = 0.0
        prev_ask_qty = 0.0
        
        line_count = 0
        filled_trades = 0
        
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
                    
                    # 1. Update local reconstructed Bids orderbook state
                    for level in bids_data:
                        price_str = level[0]
                        qty = float(level[1])
                        if qty == 0.0:
                            bids_book.pop(price_str, None)
                        else:
                            bids_book[price_str] = qty
                            
                    # 2. Update local reconstructed Asks orderbook state
                    for level in asks_data:
                        price_str = level[0]
                        qty = float(level[1])
                        if qty == 0.0:
                            asks_book.pop(price_str, None)
                        else:
                            asks_book[price_str] = qty
                            
                    # If book is still warming up, wait
                    if not bids_book or not asks_book:
                        continue
                        
                    # 3. Extract sorted Best Bids and Asks
                    sorted_bids = sorted([(float(p), q) for p, q in bids_book.items()], key=lambda x: x[0], reverse=True)
                    sorted_asks = sorted([(float(p), q) for p, q in asks_book.items()], key=lambda x: x[0])
                    
                    if not sorted_bids or not sorted_asks:
                        continue
                        
                    best_bid_price = sorted_bids[0][0]
                    best_bid_qty = sorted_bids[0][1]
                    best_ask_price = sorted_asks[0][0]
                    best_ask_qty = sorted_asks[0][1]
                    
                    mid_price = (best_bid_price + best_ask_price) / 2.0
                    tick_ns = int(packet["ts"] * 1e6) # Decode Bybit ms epoch timestamp
                    
                    # 4. Microstructure Trade Inference logic for CVD mapping
                    is_inferred_trade = False
                    trade_price = 0.0
                    trade_qty = 0.0
                    is_buyer_maker = False
                    
                    if prev_bid_price is not None and prev_ask_price is not None:
                        # Imbalance decreases at Bid price -> seller taker trade event
                        if best_bid_price == prev_bid_price and best_bid_qty < prev_bid_qty:
                            is_inferred_trade = True
                            trade_price = best_bid_price
                            trade_qty = prev_bid_qty - best_bid_qty
                            is_buyer_maker = True
                        # Imbalance decreases at Ask price -> buyer taker trade event
                        elif best_ask_price == prev_ask_price and best_ask_qty < prev_ask_qty:
                            is_inferred_trade = True
                            trade_price = best_ask_price
                            trade_qty = prev_ask_qty - best_ask_qty
                            is_buyer_maker = False
                            
                    # Update memory anchors
                    prev_bid_price = best_bid_price
                    prev_ask_price = best_ask_price
                    prev_bid_qty = best_bid_qty
                    prev_ask_qty = best_ask_qty
                    
                    # 5. Feed Reconstructed Tick to the Master State Machine
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
                    
            # Force close out open position at the end to record stats
            if self.state_machine.in_position and mid_price > 0.0:
                logger.warning("⏰ Final Closeout: Forcing market close of open option position at session end...")
                exit_side = "SELL" if self.state_machine.position_side == "LONG" else "BUY"
                asyncio.run(self.state_machine.smart_router.place_market_order(
                    symbol=self.symbol,
                    side=exit_side,
                    amount=1.0,
                    shadow_price=mid_price
                ))
                self.state_machine.in_position = False
                
        except KeyboardInterrupt:
            logger.warning("⚠️ Backtest manually interrupted.")
        finally:
            self._print_performance_report(line_count)

    def _print_performance_report(self, ticks_processed: int):
        journal = self.state_machine.smart_router.trade_journal
        
        print("\n================================================================================")
        print("📊 LEVEL 2 OPTION CONTRACT SWING BACKTEST REPORT CARD")
        print("================================================================================")
        print(f"Contract Symbol       : {self.symbol}")
        print(f"Total Ticks Processed : {ticks_processed:,}")
        
        if not journal:
            print("⚠️ No trades were executed during this option backtest session.")
            print("================================================================================\n")
            return
            
        cash = self.initial_capital
        total_fees = 0.0
        wins = 0
        completed_trades = []
        active_pos = None
        
        for fill in journal:
            total_fees += fill["fee"]
            
            if not active_pos:
                active_pos = {
                    "entry_price": fill["price"],
                    "entry_fee": fill["fee"],
                    "side": "LONG" if fill["side"] == "BUY" else "SHORT"
                }
            else:
                # Pairing exit
                entry = active_pos["entry_price"]
                exit = fill["price"]
                side = active_pos["side"]
                
                gross_pnl = (exit - entry) if side == "LONG" else (entry - exit)
                net_pnl = gross_pnl - (active_pos["entry_fee"] + fill["fee"])
                
                completed_trades.append({
                    "side": side,
                    "entry": entry,
                    "exit": exit,
                    "net_pnl": net_pnl
                })
                
                cash += net_pnl
                if net_pnl > 0:
                    wins += 1
                active_pos = None
                
        win_rate = (wins / len(completed_trades)) * 100.0 if completed_trades else 0.0
        net_percentage = ((cash - self.initial_capital) / self.initial_capital) * 100.0
        
        print(f"Completed Trades      : {len(completed_trades)}")
        print(f"Win Rate              : {win_rate:.2f}%")
        print(f"Total Exchange Fees   : ${total_fees:.2f}")
        print(f"Initial Portfolio     : ${self.initial_capital:,.2f}")
        print(f"Final Portfolio       : ${cash:,.2f}")
        print(f"Net Return            : ${cash - self.initial_capital:+.2f} ({net_percentage:+.2f}%)")
        print("-" * 80)
        
        for i, t in enumerate(completed_trades):
            print(f"   Trade {i+1}: {t['side']} | Entry: ${t['entry']:.2f} | Exit: ${t['exit']:.2f} | Net PnL: {t['net_pnl']:+.2f}")
        print("================================================================================\n")

if __name__ == "__main__":
    target_file = "/Users/singhujwal/crypto_mft/datasets/2026-05-26_BTC_USDT.ob25/2026-05-26_BTC-5JUN26-64000-C-USDT.ob25"
    
    # Run simulation
    backtester = OptionL2Backtester(file_path=target_file)
    backtester.start_simulation()
