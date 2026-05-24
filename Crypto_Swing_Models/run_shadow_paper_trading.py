import asyncio
import json
import sys
import os
import time
import logging

# Configure production logging (Set to WARNING by default to suppress trace noise)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("shadow_paper_trading.log")
    ]
)
logger = logging.getLogger("ShadowPaperTrading")

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "1_data_layer")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "2_alpha_macro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "3_alpha_micro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "4_execution")))

from state_machine import GexMicroStateMachine
from gex_mapper import GexMapper

class HistoricTickPlayer:
    """
    Historical Tick Replayer & Continuous Shadow Paper Trader.
    Streams a real-world historical order book snapshot file line-by-line,
    replays it tick-by-tick with calibrated time delays (optionally scaled),
    and runs the GEX-Micro State Machine in continuous Shadow Paper Trading mode.
    """
    def __init__(self, 
                 file_path: str, 
                 speedup_factor: float = 10.0, # Replay at 10x speed by default (1x for exact real-time)
                 symbol: str = "BTCUSDT"):
        self.file_path = file_path
        self.speedup = speedup_factor
        self.symbol = symbol
        
        # Instantiate logic
        self.state_machine = GexMicroStateMachine(symbol=self.symbol, mode="SHADOW")
        self.mapper = GexMapper(model_type="COIN_MARGINED")
        
        # Memory states for GEX updates
        self.last_gex_update_ns = None
        self.last_print_time = 0.0
        self.last_journal_len = 0
        
    async def start_paper_trading(self):
        """Starts the tick player loop."""
        if not os.path.exists(self.file_path):
            logger.error(f"Historical dataset {self.file_path} not found.")
            return
            
        logger.warning("================================================================================")
        logger.warning("🚀 STARTING CONTINUOUS SHADOW PAPER TRADING SESSION ON HISTORIC DATA")
        logger.warning(f"Dataset: {os.path.basename(self.file_path)} | Replay Speed: {self.speedup}x")
        logger.warning("================================================================================")
        
        prev_tick_ns = None
        prev_tick_real_time = None
        
        # Order book reconstruction memory
        prev_bid_price = None
        prev_ask_price = None
        prev_bid_qty = 0.0
        prev_ask_qty = 0.0
        
        line_count = 0
        
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                        
                    line_count += 1
                    try:
                        tick = json.loads(line)
                    except Exception as e:
                        continue
                        
                    bid_price = float(tick["bid"])
                    ask_price = float(tick["ask"])
                    bid_qty = float(tick["bid_size"])
                    ask_qty = float(tick["ask_size"])
                    tick_ns = int(tick["timestamp_ns"])
                    
                    mid_price = (bid_price + ask_price) / 2.0
                    
                    # 1. Perform dynamic Options GEX Walls updates (Fixed until 1.5% price drift)
                    deribit_idx = self.state_machine.basis_tracker.deribit_index_price
                    if (self.state_machine.adjusted_target_price is None or 
                        deribit_idx is None or 
                        abs(mid_price - deribit_idx) / deribit_idx > 0.015):
                        
                        strike_spacing = max(1.0, round(mid_price * 0.001, 1))
                        strikes = np_strikes = [round(mid_price + i * strike_spacing, 1) for i in range(-10, 11)]
                        
                        # Calibrate mock Open Interest (High Put OI below, High Call OI above)
                        import numpy as np
                        call_oi = np.full_like(np_strikes, 10000.0)
                        put_oi = np.full_like(np_strikes, 10000.0)
                        call_oi[14] = 150000.0  # Call Wall
                        put_oi[6] = 150000.0    # Put Wall
                        sigmas = np.full_like(np_strikes, 0.40)
                        multipliers = -np.ones_like(np_strikes)
                        multipliers[6] = 1.0    # Dealers net long puts at index 6 to create Support Put Wall
                        
                        gex_profile = self.mapper.calculate_gex_profile(
                            spot_price=mid_price,
                            strikes=np.array(np_strikes),
                            call_oi=call_oi,
                            put_oi=put_oi,
                            sigmas=sigmas,
                            t=7.0/365.0,
                            r=0.05,
                            q=0.00,
                            dealer_multipliers=multipliers
                        )
                        
                        mapping = self.mapper.map_structural_hedging(gex_profile)
                        if mapping["support_walls"]:
                            self.state_machine.update_gex_profile(
                                key_strike=mapping["support_walls"][0][0],
                                gex_value=mapping["support_walls"][0][1],
                                deribit_index=mid_price
                            )
                    
                    # 2. Order Book Trade Reconstruction for CVD absorption confirmations
                    is_inferred_trade = False
                    trade_price = 0.0
                    trade_qty = 0.0
                    is_buyer_maker = False
                    
                    if prev_bid_price is not None and prev_ask_price is not None:
                        if bid_price == prev_bid_price and bid_qty < prev_bid_qty:
                            is_inferred_trade = True
                            trade_price = bid_price
                            trade_qty = prev_bid_qty - bid_qty
                            is_buyer_maker = True
                        elif ask_price == prev_ask_price and ask_qty < prev_ask_qty:
                            is_inferred_trade = True
                            trade_price = ask_price
                            trade_qty = prev_ask_qty - ask_qty
                            is_buyer_maker = False
                            
                    # Update reconstruction memory
                    prev_bid_price = bid_price
                    prev_ask_price = ask_price
                    prev_bid_qty = bid_qty
                    prev_ask_qty = ask_qty
                    
                    # 3. Simulate time delay to next tick
                    if prev_tick_ns is not None:
                        elapsed_ns = tick_ns - prev_tick_ns
                        # Convert nanoseconds to seconds and apply speedup scaling factor
                        sleep_duration = (elapsed_ns / 1e9) / self.speedup
                        
                        # Cap sleep at 5 seconds to bypass long inactive trading gaps quickly
                        sleep_duration = min(5.0, max(0.0, sleep_duration))
                        
                        if sleep_duration > 0.001:
                            await asyncio.sleep(sleep_duration)
                            
                    prev_tick_ns = tick_ns
                    
                    # 4. Feed tick to State Machine
                    await self.state_machine.process_market_tick(
                        mid_price=mid_price,
                        bid_qty=bid_qty,
                        ask_qty=ask_qty,
                        is_trade=is_inferred_trade,
                        trade_price=trade_price,
                        trade_qty=trade_qty,
                        is_buyer_maker=is_buyer_maker,
                        timestamp_ns=tick_ns
                    )
                    
                    # Check if a new trade exits/fills completed and save report card in real-time!
                    current_journal_len = len(self.state_machine.smart_router.trade_journal)
                    if current_journal_len > self.last_journal_len and current_journal_len % 2 == 0:
                        await self._save_session_report(line_count)
                        self.last_journal_len = current_journal_len
                    
                    # 5. Live Status Output (Log progress every 10 seconds of real execution time to prevent flood)
                    current_time = time.time()
                    if current_time - self.last_print_time >= 10.0:
                        active_wall = self.state_machine.active_gex_strike
                        logger.warning(
                            f"⏰ Shadow Session Status: Spot={mid_price:.2f} | "
                            f"Active GEX Wall={active_wall} | "
                            f"State=STATE {self.state_machine.state} | "
                            f"In Position={self.state_machine.in_position} | "
                            f"Ticks Processed={line_count}"
                        )
                        self.last_print_time = current_time
        except asyncio.CancelledError:
            logger.warning("\n⚠️ Shadow paper trading session was cancelled asynchronously.")
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Shadow paper trading session was manually interrupted (Ctrl+C).")
        finally:
            # Compile and save the rich LLM evaluation report before exiting!
            await self._save_session_report(line_count)

    async def _save_session_report(self, line_count: int):
        """Compiles rich performance history to a JSON file for LLM strategic evaluations."""
        journal = self.state_machine.smart_router.trade_journal
        if not journal:
            logger.warning("\n📝 No trades were executed during this shadow session. No report saved.")
            return
            
        logger.warning("\n================================================================================")
        logger.warning("📊 COMPILING SHADOW PAPER TRADING SESSION REPORT CARD...")
        logger.warning("================================================================================")
        
        initial_capital = 10000.0
        cash = initial_capital
        wins = 0
        total_fees = 0.0
        total_trades = 0
        trade_records = []
        
        active_pos = None
        
        for fill in journal:
            total_fees += fill["fee"]
            
            # If we are not in an active position, this fill must be the entry fill
            if not active_pos:
                active_pos = {
                    "side": "LONG" if fill["side"] == "BUY" else "SHORT",
                    "entry_price": fill["price"],
                    "entry_fee": fill["fee"],
                    "entry_time_ns": fill["timestamp"] * 1e9
                }
            # If we are already in an active position, this fill must be the exit fill
            else:
                exit_price = fill["price"]
                entry_price = active_pos["entry_price"]
                
                if active_pos["side"] == "LONG":
                    trade_pnl = exit_price - entry_price
                else:
                    trade_pnl = entry_price - exit_price
                    
                net_pnl = trade_pnl - (active_pos["entry_fee"] + fill["fee"])
                cash += net_pnl
                
                total_trades += 1
                if net_pnl > 0:
                    wins += 1
                    
                trade_records.append({
                    "trade_id": total_trades,
                    "side": active_pos["side"],
                    "entry_price": entry_price,
                    "entry_fee": active_pos["entry_fee"],
                    "exit_price": exit_price,
                    "exit_fee": fill["fee"],
                    "gross_pnl": trade_pnl,
                    "net_pnl": net_pnl,
                    "duration_seconds": fill["timestamp"] - (active_pos["entry_time_ns"] / 1e9)
                })
                active_pos = None
                
        net_return_pct = ((cash - initial_capital) / initial_capital) * 100.0
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        
        report = {
            "session_metrics": {
                "initial_capital": initial_capital,
                "final_balance": cash,
                "net_pnl": cash - initial_capital,
                "net_percentage_return": net_return_pct,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_fees_paid": total_fees,
                "ticks_processed": line_count
            },
            "trade_journal": trade_records
        }
        
        report_path = "/Users/singhujwal/crypto_mft/shadow_paper_trading_report.json"
        try:
            import json
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.warning(f"📝 LLM-Ready Shadow Session Report saved successfully to: {report_path}")
            logger.warning("================================================================================")
        except Exception as e:
            logger.error(f"Failed to save shadow session report: {e}")

async def main():
    file_path = "/Users/singhujwal/crypto_mft/datasets/futures_market_data_5days.log"
    
    # Set up a historic ticker replayer running at configurable speed (defaults to infinity for backtests)
    speedup_env = os.getenv("SPEEDUP", "inf")
    speedup_factor = float('inf') if speedup_env.lower() == "inf" else float(speedup_env)
    
    player = HistoricTickPlayer(
        file_path=file_path,
        speedup_factor=speedup_factor,
        symbol="BTCUSDT"
    )
    
    await player.start_paper_trading()

if __name__ == "__main__":
    asyncio.run(main())
