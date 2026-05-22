import os
import sys
import numpy as np

# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_path)

from core.schemas import InternalTick
from backtester.engine import FastBacktestEngine
from intelligence.alpha_engine import AlphaModel
from intelligence.strategy_factory import AlphaStrategyFactory
from backtester.run_master_hft_benchmark import stream_real_market_data, HFTOrderGenerator, FVRAlphaWrapper

class DiagnosticHFTEngine(FastBacktestEngine):
    def __init__(self, max_positions: int = 3, tp_margin: float = 0.0020, sl_margin: float = 0.0150, **kwargs):
        super().__init__(**kwargs)
        self.max_positions = max_positions
        self.order_generator = HFTOrderGenerator(
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            tp_margin=tp_margin,
            sl_margin=sl_margin
        )

    def run_backtest(self, ticks) -> dict:
        cash = self.initial_cash
        peak_equity = self.initial_cash
        max_dd = 0.0
        
        active_positions = []
        total_trades = 0
        wins = 0
        running_realized_pnl = 0.0
        total_fees = 0.0
        pending_orders = []

        print(f"Starting Diagnostic Run...")
        print(f"Initial Cash: ${cash:.2f}")
        print(f"Parameters: TP={self.order_generator.tp_margin*10000:.1f} bps, SL={self.order_generator.sl_margin*10000:.1f} bps")
        print("-" * 100)

        for idx, tick in enumerate(ticks):
            mid_price = (tick.bid + tick.ask) / 2.0
            
            total_crypto = sum(pos["crypto"] for pos in active_positions)
            equity = cash + (total_crypto * mid_price)
            
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown > max_dd:
                max_dd = drawdown

            self.risk_critic.daily_peak_value = peak_equity
            self.risk_critic.current_portfolio_value = equity

            features = self.feature_store.process_tick(tick)
            alpha_forecast = 0.0
            rolling_vol = 0.0
            if features is not None:
                z_score, spread, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features
                alpha_forecast = self.alpha_model.predict(features)

            # 1. Exits Processing
            positions_to_keep = []
            for pos in active_positions:
                bracket = pos["bracket"]
                action = pos["action"]
                pos_crypto = pos["crypto"]
                
                is_tp_breached = False
                is_sl_breached = False
                
                if action == "BUY":
                    is_tp_breached = mid_price >= bracket["take_profit_price"]
                    is_sl_breached = mid_price <= bracket["stop_loss_price"]
                else:
                    is_tp_breached = mid_price <= bracket["take_profit_price"]
                    is_sl_breached = mid_price >= bracket["stop_loss_price"]

                is_reversal = False
                if self.reversal_threshold is not None and features is not None:
                    if action == "BUY":
                        is_reversal = alpha_forecast <= -self.reversal_threshold
                    else:
                        is_reversal = alpha_forecast >= self.reversal_threshold

                is_timeout = False
                if self.timeout_seconds is not None and "entry_timestamp_ns" in pos:
                    elapsed_seconds = (tick.timestamp_ns - pos["entry_timestamp_ns"]) / 1e9
                    if elapsed_seconds >= self.timeout_seconds:
                        is_timeout = True

                # A. MAKER LIMIT EXIT for TP
                if is_tp_breached:
                    executed_exit_price = bracket["take_profit_price"]
                    fee_rate = self.maker_fee
                    
                    if action == "BUY":
                        proceeds = pos_crypto * executed_exit_price
                        fee_paid = proceeds * fee_rate
                        net_exit_cash = proceeds - fee_paid
                        trade_pnl = net_exit_cash - pos["entry_cash_spent"]
                        cash += net_exit_cash
                    else:
                        buy_back_cost = abs(pos_crypto) * executed_exit_price
                        fee_paid = buy_back_cost * fee_rate
                        net_exit_cash_paid = buy_back_cost + fee_paid
                        trade_pnl = pos["entry_cash_received"] - net_exit_cash_paid
                        cash -= net_exit_cash_paid

                    total_fees += fee_paid
                    total_trades += 1
                    running_realized_pnl += trade_pnl
                    if trade_pnl > 0: wins += 1
                    print(f" [EXIT TP] {action} Price: {executed_exit_price:.2f} | PnL: ${trade_pnl:+.2f} | Fee: ${fee_paid:.2f} | Cash: ${cash:.2f}")
                
                # B. TAKER MARKET EXIT for SL/Reversals/Timeouts
                elif is_sl_breached or is_reversal or is_timeout:
                    random_slippage = np.random.normal(loc=0.00005, scale=self.slippage_std)
                    drift_direction = 1 if action == "SELL" else -1
                    slippage_multiplier = 1 + (drift_direction * max(0.0, random_slippage))
                    executed_exit_price = mid_price * slippage_multiplier

                    # Collar
                    max_exit_slip = 0.0010
                    if action == "BUY" and executed_exit_price < mid_price * (1.0 - max_exit_slip):
                        positions_to_keep.append(pos)
                        continue
                    elif action == "SELL" and executed_exit_price > mid_price * (1.0 + max_exit_slip):
                        positions_to_keep.append(pos)
                        continue

                    fee_rate = self.taker_fee
                    if action == "BUY":
                        proceeds = pos_crypto * executed_exit_price
                        fee_paid = proceeds * fee_rate
                        net_exit_cash = proceeds - fee_paid
                        trade_pnl = net_exit_cash - pos["entry_cash_spent"]
                        cash += net_exit_cash
                    else:
                        buy_back_cost = abs(pos_crypto) * executed_exit_price
                        fee_paid = buy_back_cost * fee_rate
                        net_exit_cash_paid = buy_back_cost + fee_paid
                        trade_pnl = pos["entry_cash_received"] - net_exit_cash_paid
                        cash -= net_exit_cash_paid

                    total_fees += fee_paid
                    total_trades += 1
                    running_realized_pnl += trade_pnl
                    if trade_pnl > 0: wins += 1
                    reason = "SL" if is_sl_breached else ("REVERSAL" if is_reversal else "TIMEOUT")
                    print(f" 🚨 [EXIT {reason}] {action} Price: {executed_exit_price:.2f} | PnL: ${trade_pnl:+.2f} | Fee: ${fee_paid:.2f} | Cash: ${cash:.2f}")
                else:
                    positions_to_keep.append(pos)

            active_positions = positions_to_keep
            total_crypto = sum(pos["crypto"] for pos in active_positions)
            equity = cash + (total_crypto * mid_price)

            # 2. Fills
            remaining_pending = []
            for fill_at, proposed_order, proposed_mid in pending_orders:
                if idx >= fill_at and len(active_positions) < self.max_positions:
                    action = proposed_order["action"]
                    notional = proposed_order["notional"]
                    limit_price = proposed_order["limit_price"]

                    random_slippage = np.random.normal(loc=0.00005, scale=self.slippage_std)
                    drift_direction = 1 if action == "BUY" else -1
                    executed_price = limit_price * (1 + drift_direction * max(0.0, random_slippage))

                    fee_rate = self.maker_fee
                    fee_paid = notional * fee_rate
                    total_fees += fee_paid

                    tp_margin = proposed_order.get("tp_margin_used", self.order_generator.tp_margin)
                    sl_margin = proposed_order.get("sl_margin_used", self.order_generator.sl_margin)

                    if action == "BUY":
                        cash -= notional
                        purchased_crypto = (notional - fee_paid) / executed_price
                        pos = {
                            "action": "BUY",
                            "entry_tick_idx": idx,
                            "entry_timestamp_ns": tick.timestamp_ns,
                            "entry_cash_spent": notional,
                            "crypto": purchased_crypto,
                            "bracket": {
                                "entry_price": executed_price,
                                "take_profit_price": executed_price * (1.0 + tp_margin),
                                "stop_loss_price": executed_price * (1.0 - sl_margin)
                            }
                        }
                        active_positions.append(pos)
                    else:
                        cash += (notional - fee_paid)
                        short_crypto = notional / executed_price
                        pos = {
                            "action": "SELL",
                            "entry_tick_idx": idx,
                            "entry_timestamp_ns": tick.timestamp_ns,
                            "entry_cash_received": notional - fee_paid,
                            "crypto": -short_crypto,
                            "bracket": {
                                "entry_price": executed_price,
                                "take_profit_price": executed_price * (1.0 - tp_margin),
                                "stop_loss_price": executed_price * (1.0 + sl_margin)
                            }
                        }
                        active_positions.append(pos)
                    print(f" 🟢 [ENTRY FILL] {action} Price: {executed_price:.2f} | Notional: ${notional:.2f} | Fee: ${fee_paid:.2f} | TP: {pos['bracket']['take_profit_price']:.2f} | SL: {pos['bracket']['stop_loss_price']:.2f}")
                else:
                    remaining_pending.append((fill_at, proposed_order, proposed_mid))
            pending_orders = remaining_pending

            # 3. Entries
            if len(active_positions) < self.max_positions and len(pending_orders) == 0 and features is not None:
                if abs(alpha_forecast) >= 0.00035:
                    target_weight = self.optimizer.calculate_target_weight(alpha_forecast)
                    if abs(target_weight) >= 0.05:
                        flat_weight = 1.0 if target_weight > 0 else -1.0
                    else:
                        flat_weight = 0.0
                else:
                    flat_weight = 0.0
                
                if abs(flat_weight) > 0.0:
                    slot_portfolio_value = equity / float(self.max_positions)
                    proposed_order = self.order_generator.generate_bracket_order(
                        symbol=tick.symbol,
                        target_weight=flat_weight,
                        portfolio_value=slot_portfolio_value,
                        bid=tick.bid,
                        ask=tick.ask,
                        volatility=rolling_vol / mid_price
                    )
                    if proposed_order and self.risk_critic.validate_order(proposed_order, mid_price):
                        pending_orders.append((idx + self.latency_ticks, proposed_order, mid_price))
                        print(f"🔍 [SIGNAL] {proposed_order['action']} proposed near mid {mid_price:.2f}")

        net_pnl = running_realized_pnl
        final_balance = self.initial_cash + net_pnl
        net_percentage_return = (net_pnl / self.initial_cash) * 100.0
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        print("-" * 100)
        print(f"Ending Balance: ${final_balance:.2f}")
        print(f"Net PnL       : ${net_pnl:+.2f} ({net_percentage_return:+.2f}%)")
        print(f"Total Trades  : {total_trades}")
        print(f"Win Rate      : {win_rate:.2f}%")
        return {}

def main():
    np.random.seed(42)
    # Load the highly recent May 22 dataset from workspace root
    filepath = os.path.join(workspace_path, "real_market_data_live_22_may_2026.log")
    
    backtester = DiagnosticHFTEngine(
        max_positions=3,
        initial_cash=10000.0,
        latency_ticks=1,
        maker_fee=0.00100,      # 10 bps
        taker_fee=0.00100,      # 10 bps
        slippage_std=0.0001,
        tp_margin=0.0080,       # 80 bps (Upgraded)
        sl_margin=0.0200,       # 200 bps (Upgraded)
        lookback=500,
        reversal_threshold=None,
        timeout_seconds=3600
    )
    
    # Initialize model
    backtester.alpha_model = AlphaModel(
        alpha_type="VALLEY_MOUNTAIN", 
        enable_fvr=False, # Disable to see pure strategy behavior
        lookback=500,
        entry_buffer=0.0010,    # 10 bps (Upgraded)
        ofi_threshold=0.25      # 25% OFI filter (Upgraded)
    )

    print("Running Diagnostic for VALLEY_MOUNTAIN strategy on May 22 Dataset...")
    ticks = list(stream_real_market_data(filepath))
    backtester.run_backtest(ticks)

if __name__ == "__main__":
    main()
