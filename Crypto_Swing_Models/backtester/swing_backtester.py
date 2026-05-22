import json
import sys
import os
import time
import numpy as np
import logging

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_alpha_macro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../3_alpha_micro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../4_execution")))

from state_machine import GexMicroStateMachine
from gex_mapper import GexMapper
from smart_router import SmartRouter

logger = logging.getLogger("SwingBacktester")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SwingBacktestEngine:
    """
    Institutional-grade Event-Driven Backtester for the GEX-Micro State Machine.
    Streams massive log files (up to 10GB+) line-by-line with O(1) memory overhead.
    Reconstructs aggressive trades from L2 order book changes to feed CVD engines,
    dynamically moves GEX Put/Call walls, and prints hourly performance reports.
    """
    def __init__(self, 
                 initial_cash: float = 10000.0,
                 maker_fee: float = 0.0010, # 0.1% Maker Fee as requested
                 taker_fee: float = 0.0010): # 0.1% Taker Fee as requested
                 
        self.initial_cash = initial_cash
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
    def stream_backtest(self, file_path: str) -> dict:
        """
        Streams market data from the log file, processes ticks, and simulates trading.
        """
        # Instantiate mapping and state engines
        mapper = GexMapper(model_type="COIN_MARGINED")
        sm = GexMicroStateMachine(symbol="BTCUSDT", mode="SHADOW")
        
        # Performance tracking states
        cash = self.initial_cash
        equity = self.initial_cash
        peak_equity = self.initial_cash
        max_dd = 0.0
        
        in_position = False
        position_side = None
        entry_price = 0.0
        
        total_trades = 0
        wins = 0
        total_fees = 0.0
        
        # L2 state reconstruction memory
        prev_bid_price = None
        prev_ask_price = None
        prev_bid_qty = 0.0
        prev_ask_qty = 0.0
        
        # Simulated time states
        start_time_ns = None
        last_gex_update_ns = None
        current_hour = 0
        hourly_start_equity = self.initial_cash
        
        # Metrics
        hourly_reports = []
        
        # Loop through file line-by-line using low-overhead streaming
        logger.info(f"Initiating streaming backtest on dataset: {os.path.basename(file_path)}")
        
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    tick_data = json.loads(line)
                except Exception as e:
                    logger.warning(f"Skip malformed line {line_num}: {e}")
                    continue
                
                bid_price = float(tick_data["bid"])
                ask_price = float(tick_data["ask"])
                bid_qty = float(tick_data["bid_size"])
                ask_qty = float(tick_data["ask_size"])
                timestamp_ns = int(tick_data["timestamp_ns"])
                
                mid_price = (bid_price + ask_price) / 2.0
                
                if start_time_ns is None:
                    start_time_ns = timestamp_ns
                    last_gex_update_ns = timestamp_ns
                    hourly_start_equity = cash
                    logger.info(f"Backtest baseline established at price {mid_price:.2f}")
                    
                # 1. Order Book Trade Reconstruction
                is_inferred_trade = False
                trade_price = 0.0
                trade_qty = 0.0
                is_buyer_maker = False
                
                if prev_bid_price is not None and prev_ask_price is not None:
                    # If bid size dropped at same bid price -> Taker Sell (buyer is maker)
                    if bid_price == prev_bid_price and bid_qty < prev_bid_qty:
                        is_inferred_trade = True
                        trade_price = bid_price
                        trade_qty = prev_bid_qty - bid_qty
                        is_buyer_maker = True
                    # If ask size dropped at same ask price -> Taker Buy (buyer is taker)
                    elif ask_price == prev_ask_price and ask_qty < prev_ask_qty:
                        is_inferred_trade = True
                        trade_price = ask_price
                        trade_qty = prev_ask_qty - ask_qty
                        is_buyer_maker = False
                        
                # Update memory
                prev_bid_price = bid_price
                prev_ask_price = ask_price
                prev_bid_qty = bid_qty
                prev_ask_qty = ask_qty
                
                # 2. Dynamic Options GEX Recalculation (Only on drift > 1.5% or first run)
                # Keep strikes FIXED until a major drift occurs so price can actually approach the walls
                deribit_idx = sm.basis_tracker.deribit_index_price
                if (sm.adjusted_target_price is None or 
                    deribit_idx is None or 
                    abs(mid_price - deribit_idx) / deribit_idx > 0.015):
                    
                    # Generate options strikes centered on current mid price
                    strike_spacing = max(1.0, round(mid_price * 0.001, 1))
                    strikes = np.array([round(mid_price + i * strike_spacing, 1) for i in range(-10, 11)])
                    
                    # Calibrate mock Open Interest (High Put OI below, High Call OI above)
                    call_oi = np.full_like(strikes, 10000.0)
                    put_oi = np.full_like(strikes, 10000.0)
                    
                    # Put Wall at -0.4% index, Call Wall at +0.4% index
                    call_oi[14] = 150000.0  # Call Wall at strike +4 steps
                    put_oi[6] = 150000.0    # Put Wall at strike -4 steps
                    
                    sigmas = np.full_like(strikes, 0.40)
                    
                    # Estimate dealer multipliers (-1.0 for short options walls)
                    multipliers = -np.ones_like(strikes)
                    
                    # Compute GEX Profile
                    gex_profile = mapper.calculate_gex_profile(
                        spot_price=mid_price,
                        strikes=strikes,
                        call_oi=call_oi,
                        put_oi=put_oi,
                        sigmas=sigmas,
                        t=7.0/365.0,
                        r=0.05,
                        q=0.00,
                        dealer_multipliers=multipliers
                    )
                    
                    # Map Walls
                    mapping = mapper.map_structural_hedging(gex_profile)
                    
                    if mapping["support_walls"]:
                        # Update state machine Put wall
                        sm.update_gex_profile(
                            key_strike=mapping["support_walls"][0][0],
                            gex_value=mapping["support_walls"][0][1],
                            deribit_index=mid_price
                        )
                        
                # 3. Hourly Performance Reports
                elapsed_hours = int((timestamp_ns - start_time_ns) // (3600 * 1e9))
                if elapsed_hours > current_hour:
                    # Hour crossed! Calculate and print hourly stats
                    current_equity = cash
                    if in_position:
                        if position_side == "LONG":
                            current_equity = cash + (1.0 * (mid_price - entry_price))
                        else:
                            current_equity = cash + (1.0 * (entry_price - mid_price))
                            
                    hourly_pnl = current_equity - hourly_start_equity
                    cumulative_pnl = current_equity - self.initial_cash
                    
                    report = {
                        "hour": current_hour + 1,
                        "hourly_pnl": hourly_pnl,
                        "cumulative_pnl": cumulative_pnl,
                        "equity": current_equity
                    }
                    hourly_reports.append(report)
                    
                    print(
                        f"⏰ Hourly Report | Hour {report['hour']:02d} | "
                        f"Hourly PnL: {report['hourly_pnl']:+.2f} | "
                        f"Cumulative PnL: {report['cumulative_pnl']:+.2f} | "
                        f"Equity: {report['equity']:.2f}"
                    )
                    
                    # Reset hourly variables
                    current_hour = elapsed_hours
                    hourly_start_equity = current_equity
                    
                # 4. Execute State Machine Tick Processing
                prev_state = sm.state
                
                # Run tick (using sync runner since process_market_tick is async)
                import asyncio
                asyncio.run(sm.process_market_tick(
                    mid_price=mid_price,
                    bid_qty=bid_qty,
                    ask_qty=ask_qty,
                    is_trade=is_inferred_trade,
                    trade_price=trade_price,
                    trade_qty=trade_qty,
                    is_buyer_maker=is_buyer_maker
                ))
                
                # 5. Trade Execution and Fee Deductions
                if prev_state == 2 and sm.state == 3:
                    # Limit order filled (Maker entry)
                    in_position = True
                    position_side = sm.position_side
                    entry_price = mid_price
                    
                    fee = entry_price * self.maker_fee
                    cash -= fee
                    total_fees += fee
                    logger.info(f"TRADE_ENTRY: Filled Maker {position_side} order @ {entry_price:.2f}. Fee paid: ${fee:.2f}")
                    
                elif prev_state == 3 and sm.state == 0:
                    # Market order filled (Taker exit)
                    in_position = False
                    
                    if position_side == "LONG":
                        trade_pnl = mid_price - entry_price
                    else: # SHORT
                        trade_pnl = entry_price - mid_price
                        
                    fee = mid_price * self.taker_fee
                    cash += trade_pnl - fee
                    total_fees += fee
                    
                    total_trades += 1
                    if trade_pnl > 0:
                        wins += 1
                        
                    logger.info(
                        f"TRADE_EXIT: Filled Taker exit @ {mid_price:.2f}. "
                        f"PnL: {trade_pnl:+.2f} (net: {trade_pnl - fee:+.2f}). Fee paid: ${fee:.2f}"
                    )
                    
                    # Reset states
                    position_side = None
                    entry_price = 0.0
                    
                # Track drawdowns
                current_equity = cash
                if in_position:
                    if position_side == "LONG":
                        current_equity = cash + (1.0 * (mid_price - entry_price))
                    else:
                        current_equity = cash + (1.0 * (entry_price - mid_price))
                        
                if current_equity > peak_equity:
                    peak_equity = current_equity
                drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
                if drawdown > max_dd:
                    max_dd = drawdown
                    
        # Final hour report at end of dataset
        final_equity = cash
        if in_position:
            if position_side == "LONG":
                final_equity = cash + (1.0 * (mid_price - entry_price))
            else:
                final_equity = cash + (1.0 * (entry_price - mid_price))
                
        hourly_pnl = final_equity - hourly_start_equity
        cumulative_pnl = final_equity - self.initial_cash
        
        report = {
            "hour": current_hour + 1,
            "hourly_pnl": hourly_pnl,
            "cumulative_pnl": cumulative_pnl,
            "equity": final_equity
        }
        hourly_reports.append(report)
        
        print(
            f"⏰ Hourly Report | Hour {report['hour']:02d} (Final) | "
            f"Hourly PnL: {report['hourly_pnl']:+.2f} | "
            f"Cumulative PnL: {report['cumulative_pnl']:+.2f} | "
            f"Equity: {report['equity']:.2f}"
        )
        
        net_pnl = final_equity - self.initial_cash
        net_return = (net_pnl / self.initial_cash) * 100.0
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        
        return {
            "final_balance": final_equity,
            "net_pnl": net_pnl,
            "net_percentage_return": net_return,
            "max_drawdown": max_dd * 100.0,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_fees_paid": total_fees,
            "hourly_reports": hourly_reports
        }
