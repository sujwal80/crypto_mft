import numpy as np
import sys
import os
import logging
from typing import List, Dict, Optional

# Setup import paths to find state machine components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_alpha_macro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../3_alpha_micro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../4_execution")))

from state_machine import GexMicroStateMachine

logger = logging.getLogger(__name__)

class BacktestTick:
    """Represent a unified market event tick (either L2 order book change or trade sweep)."""
    def __init__(self, timestamp: float, price: float, bid_qty: float, ask_qty: float, 
                 is_trade: bool = False, trade_price: float = 0.0, trade_qty: float = 0.0, 
                 is_buyer_maker: bool = False):
        self.timestamp = timestamp
        self.price = price
        self.bid_qty = bid_qty
        self.ask_qty = ask_qty
        self.is_trade = is_trade
        self.trade_price = trade_price
        self.trade_qty = trade_qty
        self.is_buyer_maker = is_buyer_maker

class GexBacktestEngine:
    """
    High-precision Event-Driven Historical Backtesting Engine 
    for the GEX-Micro State Machine.
    Simulates execution with full latency, maker/taker fees, and slippage friction.
    """
    def __init__(self, 
                 initial_cash: float = 10000.0,
                 maker_fee: float = 0.001, # 0.1% maker fee
                 taker_fee: float = 0.001, # 0.1% taker fee
                 slippage_pct: float = 0.0003,
                 grace_window_ticks: int = 3000,
                 resample_ticks: int = 1500): 
                 
        self.initial_cash = initial_cash
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage_pct
        self.grace_window = grace_window_ticks
        self.resample_ticks = resample_ticks
        
    async def run_backtest(self, ticks: List[BacktestTick], key_strike: float, gex_value: float, deribit_index: float) -> Dict:
        """
        Executes backtest over the tick sequence with the designated GEX profile.
        """
        # Instantiate a State Machine in Shadow Mode to simulate executions
        sm = GexMicroStateMachine(symbol="BTCUSDT", mode="SHADOW", grace_window_ticks=self.grace_window, resample_ticks=self.resample_ticks)
        
        # Set the options GEX wall boundary
        sm.update_gex_profile(key_strike=key_strike, gex_value=gex_value, deribit_index=deribit_index)
        
        cash = self.initial_cash
        equity = self.initial_cash
        peak_equity = self.initial_cash
        max_drawdown = 0.0
        
        total_trades = 0
        wins = 0
        total_fees = 0.0
        
        trade_returns = []
        
        # Override the state machine's smart router placing methods to direct PnL here
        # This lets us cleanly compute backtest metrics based on state transitions!
        class BacktestOrderState:
            def __init__(self):
                self.in_position = False
                self.side = None
                self.entry_price = 0.0
                
        order_state = BacktestOrderState()
        
        processed_fills_count = 0
        
        for tick in ticks:
            # Track mid-price
            mid_price = tick.price
            
            # Calculate current portfolio equity
            if order_state.in_position:
                if order_state.side == "LONG":
                    equity = cash + (1.0 * (mid_price - order_state.entry_price))
                else: # SHORT
                    equity = cash + (1.0 * (order_state.entry_price - mid_price))
            else:
                equity = cash
                
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
            # Feed tick into GexMicroStateMachine Master Loop
            await sm.process_market_tick(
                mid_price=tick.price,
                bid_qty=tick.bid_qty,
                ask_qty=tick.ask_qty,
                is_trade=tick.is_trade,
                trade_price=tick.trade_price,
                trade_qty=tick.trade_qty,
                is_buyer_maker=tick.is_buyer_maker,
                timestamp_ns=int(tick.timestamp * 1e9)
            )
            
            # Poll SmartRouter trade journal for new fills (Immune to State Aliasing!)
            journal = sm.smart_router.trade_journal
            while processed_fills_count < len(journal):
                fill = journal[processed_fills_count]
                
                if fill["status"] == "FILLED":
                    if not order_state.in_position:
                        # Entry Filled!
                        order_state.in_position = True
                        order_state.side = "LONG" if fill["side"] == "BUY" else "SHORT"
                        order_state.entry_price = fill["price"]
                        
                        # Use exact fee charged by the router
                        fee = fill["fee"]
                        cash -= fee
                        total_fees += fee
                        
                    else:
                        # Exit Filled (Maker/Taker)!
                        exit_price = fill["price"]
                        
                        if order_state.side == "LONG":
                            trade_pnl = (exit_price - order_state.entry_price)
                        else: # SHORT
                            trade_pnl = (order_state.entry_price - exit_price)
                            
                        # Use exact fee charged by the router
                        fee = fill["fee"]
                        net_trade_pnl = trade_pnl - fee
                        cash += trade_pnl - fee
                        total_fees += fee
                        
                        total_trades += 1
                        if trade_pnl > 0:
                            wins += 1
                            
                        # Track return for Sharpe calculation
                        trade_return = net_trade_pnl / order_state.entry_price
                        trade_returns.append(trade_return)
                        
                        # Clear order state
                        order_state.in_position = False
                        order_state.side = None
                        order_state.entry_price = 0.0
                        
                processed_fills_count += 1

                
        final_balance = equity
        net_pnl = final_balance - self.initial_cash
        net_return = (net_pnl / self.initial_cash) * 100.0
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        
        # Annualized Sharpe estimation based on trade returns distribution
        if len(trade_returns) > 1:
            std_return = np.std(trade_returns)
            mean_return = np.mean(trade_returns)
            # Annualize assuming ~252 trading days and e.g. 5 trades per day average (1260 trades/year)
            # This is a rough estimate, but much better than hardcoded 3.24
            sharpe = (mean_return / std_return * np.sqrt(1260)) if std_return > 1e-8 else 0.0
        else:
            sharpe = 0.0
            
        return {
            "final_balance": final_balance,
            "net_pnl": net_pnl,
            "net_percentage_return": net_return,
            "max_drawdown": max_drawdown * 100.0,
            "sharpe_ratio": sharpe,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_fees_paid": total_fees
        }
