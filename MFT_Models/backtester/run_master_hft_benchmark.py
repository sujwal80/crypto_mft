import os
import sys
import time
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Optional

# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_path)

from core.schemas import InternalTick
from backtester.engine import FastBacktestEngine
from intelligence.alpha_engine import AlphaModel
from intelligence.strategy_factory import AlphaStrategyFactory
from intelligence.legacy.micro_trend_alpha import MicroTrendMomentumAlpha
from intelligence.legacy.gex_oi_alpha import GEXAlphaStrategy
from intelligence.legacy.kalman_alpha import KalmanFilterAlpha
from intelligence.legacy.ml_alpha_model import MLAlphaModel
from intelligence.legacy.vol_micro_trend_alpha import VolMicroTrendStrategy
from intelligence.vsabs_alpha import VSABSAlpha
from intelligence.base_strategy import BaseAlphaStrategy
from intelligence.order_generator import OrderGenerator
def stream_real_market_data(filepath: str):
    """Yields InternalTick objects line-by-line from a JSONL file, keeping RAM usage near 0 MB."""
    if not os.path.exists(filepath):
        print(f"❌ Error: Real market data file not found at: {filepath}")
        return
        
    with open(filepath, "r") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    yield InternalTick.model_validate_json(line_str)
                except Exception:
                    pass

# Register legacy strategies dynamically in factory
AlphaStrategyFactory._REGISTRY["MICRO_TREND"] = MicroTrendMomentumAlpha
AlphaStrategyFactory._REGISTRY["GEX_OI"] = GEXAlphaStrategy
AlphaStrategyFactory._REGISTRY["KALMAN"] = KalmanFilterAlpha
AlphaStrategyFactory._REGISTRY["ML"] = MLAlphaModel
AlphaStrategyFactory._REGISTRY["VOL_MICRO_TREND"] = VolMicroTrendStrategy
AlphaStrategyFactory._REGISTRY["VSABS"] = VSABSAlpha

# Optimized institutional HFT parameters presets
MODEL_PARAMS = {
    "HYBRID": {"lookback": 50, "tp_margin": 0.0012, "sl_margin": 0.0150, "threshold": 0.3, "min_return_threshold": 0.0010, "reversal_threshold": None, "timeout_seconds": 3600},
    "KALMAN": {"lookback": 50, "tp_margin": 0.0012, "sl_margin": 0.0150, "threshold": 0.0008, "reversal_threshold": 0.0005, "timeout_seconds": 3600},
    "ML": {"lookback": 50, "tp_margin": 0.0012, "sl_margin": 0.0150, "min_return_threshold": 0.00015, "reversal_threshold": None, "timeout_seconds": 3600},
    "MICRO_TREND": {"lookback": 50, "tp_margin": 0.0012, "sl_margin": 0.0150, "threshold": 0.35, "reversal_threshold": None, "timeout_seconds": 3600},
    "GEX_OI": {"lookback": 50, "tp_margin": 0.0020, "sl_margin": 0.0150, "threshold": 0.3, "reversal_threshold": None, "timeout_seconds": 3600},
    "VOL_MICRO_TREND": {"lookback": 50, "tp_margin": 0.0012, "sl_margin": 0.0150, "threshold": 0.35, "min_return_threshold": 0.0015, "reversal_threshold": None, "timeout_seconds": 3600},
    "VSABS": {"lookback": 50, "tp_margin": 0.0012, "sl_margin": 0.0150, "threshold": 0.35, "min_return_threshold": 0.0, "reversal_threshold": None, "timeout_seconds": 3600, "fee_rate": 0.00100, "fvr_multiplier": 2.5, "vol_ceiling": 0.0150},
    "VALLEY_MOUNTAIN": {"lookback": 500, "tp_margin": 0.0080, "sl_margin": 0.0200, "entry_buffer": 0.0010, "ofi_threshold": 0.25, "timeout_seconds": 3600}
}

class HFTOrderGenerator(OrderGenerator):
    """
    HFT OrderGenerator applying volatility-adaptive exits and fee-inclusive clamps.
    """
    def __init__(self, tp_margin: float, sl_margin: float, maker_fee: float = 0.00100, taker_fee: float = 0.00100, **kwargs):
        super().__init__(tp_margin=tp_margin, sl_margin=sl_margin, **kwargs)
        # TP covers standard fees and secures at least 10 bps net profit (20 bps minimum floor)
        self.tp_floor = 0.0020
        # SL relaxed to avoid local noise-stopouts
        self.sl_floor = 0.0150

    def generate_bracket_order(
        self,
        symbol: str,
        target_weight: float,
        portfolio_value: float,
        bid: float,
        ask: float,
        volatility: Optional[float] = None
    ) -> Optional[Dict]:
        order = super().generate_bracket_order(
            symbol=symbol,
            target_weight=target_weight,
            portfolio_value=portfolio_value,
            bid=bid,
            ask=ask,
            volatility=volatility
        )
        if order:
            action = order["action"]
            tp_margin = self.tp_margin_long if action == "BUY" and self.tp_margin_long is not None else self.tp_margin
            sl_margin = self.sl_margin_long if action == "BUY" and self.sl_margin_long is not None else self.sl_margin
            if action == "SELL":
                tp_margin = self.tp_margin_short if self.tp_margin_short is not None else self.tp_margin
                sl_margin = self.sl_margin_short if self.sl_margin_short is not None else self.sl_margin
            
            if volatility is not None and volatility > 0.0:
                scale_factor = volatility / 0.00015
                tp_margin = tp_margin * scale_factor
                sl_margin = sl_margin * scale_factor
            
            tp_clamped = max(self.tp_floor, tp_margin)
            sl_clamped = max(self.sl_floor, sl_margin)
            
            limit_price = order["limit_price"]
            if action == "BUY":
                tp_price = limit_price * (1.0 + tp_clamped)
                sl_price = limit_price * (1.0 - sl_clamped)
            else:
                tp_price = limit_price * (1.0 - tp_clamped)
                sl_price = limit_price * (1.0 + sl_clamped)
                
            order["bracket"]["take_profit_price"] = tp_price
            order["bracket"]["stop_loss_price"] = sl_price
            order["tp_margin_used"] = tp_clamped
            order["sl_margin_used"] = sl_clamped
        return order

class FVRAlphaWrapper(BaseAlphaStrategy):
    def __init__(self, wrapped_strategy: BaseAlphaStrategy, f_total: float = 0.00015, fvr_limit: float = 2.5):
        self.wrapped = wrapped_strategy
        self.f_total = f_total
        self.fvr_limit = fvr_limit

    def predict(self, features: np.ndarray) -> float:
        rolling_vol = features[4]
        mid_price = features[5]
        vol_rel = rolling_vol / (mid_price + 1e-8)
        
        if vol_rel > 0.0 and self.fvr_limit is not None:
            fvr = self.f_total / vol_rel
            if fvr > self.fvr_limit:
                return 0.0
        return self.wrapped.predict(features)

class MasterHFTBacktestEngine(FastBacktestEngine):
    """
    Master institutional HFT Backtest Simulator integrating closed-trade realized P&L,
    Maker Limit exits for Take-Profits, and multi-position concurrent execution.
    """
    def __init__(self, max_positions: int = 3, tp_margin: float = 0.0012, sl_margin: float = 0.0150, **kwargs):
        super().__init__(**kwargs)
        self.max_positions = max_positions
        self.order_generator = HFTOrderGenerator(
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            tp_margin=tp_margin,
            sl_margin=sl_margin
        )

    def run_backtest(self, ticks: List[InternalTick]) -> Dict:
        cash = self.initial_cash
        peak_equity = self.initial_cash
        max_dd = 0.0
        
        active_positions: List[Dict] = []
        total_trades = 0
        wins = 0
        running_realized_pnl = 0.0
        total_fees = 0.0
        pending_orders = []

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

                # A. MAKER LIMIT EXIT for TP (0 slippage, Maker fee rate)
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
                    if trade_pnl > 0:
                        wins += 1
                
                # B. TAKER MARKET EXIT for SL/Reversals/Timeouts (Slippage, Taker fee rate)
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
                    if trade_pnl > 0:
                        wins += 1
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
                    filled_any = True
                else:
                    remaining_pending.append((fill_at, proposed_order, proposed_mid))
            pending_orders = remaining_pending

            # 3. Entries
            if len(active_positions) < self.max_positions and len(pending_orders) == 0 and features is not None:
                # Expected Return entry barrier
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

        net_pnl = running_realized_pnl
        final_balance = self.initial_cash + net_pnl
        net_percentage_return = (net_pnl / self.initial_cash) * 100.0
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        return {
            "net_pnl": net_pnl,
            "net_percentage_return": net_percentage_return,
            "max_drawdown": max_dd * 100.0,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_fees_paid": total_fees
        }

def run_master_competitor(alpha_type: str, filepath: str, max_positions: int) -> dict:
    np.random.seed(42)
    params = MODEL_PARAMS.get(alpha_type, {})
    
    backtester = MasterHFTBacktestEngine(
        max_positions=max_positions,
        initial_cash=10000.0,
        latency_ticks=1,
        maker_fee=0.00100,      # 10.0 basis points (0.1%)
        taker_fee=0.00100,      # 10.0 basis points (0.1%)
        slippage_std=0.0001,
        tp_margin=params.get("tp_margin"),
        sl_margin=params.get("sl_margin"),
        lookback=params.get("lookback"),
        reversal_threshold=params.get("reversal_threshold"),
        timeout_seconds=params.get("timeout_seconds")
    )
    
    threshold = params.get("threshold")
    min_return_threshold = params.get("min_return_threshold", 0.0)
    
    model_kwargs = {"enable_fvr": False, "min_return_threshold": min_return_threshold}
    if threshold is not None:
        model_kwargs["threshold"] = threshold
    
    for k, v in params.items():
        if k not in ["tp_margin", "sl_margin", "lookback", "reversal_threshold", "timeout_seconds", "threshold", "min_return_threshold"]:
            model_kwargs[k] = v
        
    backtester.alpha_model = AlphaModel(alpha_type=alpha_type, **model_kwargs)

    # Decorate with HFT FVR protection
    original = backtester.alpha_model.active_strategy
    backtester.alpha_model.active_strategy = FVRAlphaWrapper(
        wrapped_strategy=original,
        f_total=0.00200,
        fvr_limit=2.5
    )
    ticks_generator = stream_real_market_data(filepath)
    return backtester.run_backtest(ticks_generator)

async def main():
    # 2. Standard Sweep Mode
    print("=================================================================================")
    print("📈 ENTERPRISE MFT - HIGH-FIDELITY MULTI-REGIME STRESS TEST HARNESS")
    print("=================================================================================")

    # Define the 7 Stress-Testing Scenarios
    scenarios = {
        "RECENT_MAY22": os.path.join(workspace_path, "real_market_data_live_22_may_2026.log"),
    }

    competitors = ["HYBRID", "KALMAN", "ML", "MICRO_TREND", "GEX_OI", "VOL_MICRO_TREND", "VSABS", "VALLEY_MOUNTAIN"]

    # Data structures to hold returns across all scenarios
    matrix_1 = {model: {} for model in competitors}
    matrix_3 = {model: {} for model in competitors}

    loop = asyncio.get_running_loop()
    tasks = []

    with ProcessPoolExecutor() as executor:
        for scenario_name, filepath in scenarios.items():
            if not os.path.exists(filepath):
                print(f"⚠️ Scenario [{scenario_name}] ticks missing. Skipping.")
                continue
            
            for model in competitors:
                # 1 Slot Task
                t1 = loop.run_in_executor(executor, run_master_competitor, model, filepath, 1)
                tasks.append((model, scenario_name, 1, t1))
                # 3 Slots Task
                t3 = loop.run_in_executor(executor, run_master_competitor, model, filepath, 3)
                tasks.append((model, scenario_name, 3, t3))

        print(f"⚡ Scheduled {len(tasks)} concurrent simulations across CPU cores using ProcessPoolExecutor...")
        start_time = time.time()

        # Wait and gather all tasks concurrently
        for model, scenario_name, max_positions, t in tasks:
            res = await t
            if max_positions == 1:
                matrix_1[model][scenario_name] = res["net_percentage_return"]
            else:
                matrix_3[model][scenario_name] = res["net_percentage_return"]
            print(f"  ✅ Finished: {model:<15} on {scenario_name:<12} ({max_positions} Position{'s' if max_positions > 1 else ''}) | PnL: ${res['net_pnl']:+.2f} ({res['net_percentage_return']:+.2f}%) | Trades: {res['total_trades']}")

        duration = time.time() - start_time
        print(f"\n⚡ All parallel HFT backtests completed in {duration:.2f}s!")

    print("\n=================================================================================")
    print("👑 REGIME PROFITABILITY MATRIX — PART 1: baseline setup (MAX 1 CONCURRENT TRADE)")
    print("=================================================================================")
    header = f"{'Model':<15} | " + " | ".join(f"{sc:<11}" for sc in scenarios.keys()) + " | Robustness"
    print(header)
    print("-" * len(header))
    
    for model in competitors:
        row_str = f"{model:<15} | "
        profitable_count = 0
        total_scenarios = 0
        for sc in scenarios.keys():
            ret = matrix_1[model].get(sc, 0.0)
            row_str += f"{ret:+10.2f}% | "
            # Strategy preserves capital if return is flat or positive
            if ret >= -0.05:
                profitable_count += 1
            total_scenarios += 1
        
        robustness = (profitable_count / total_scenarios * 100.0) if total_scenarios > 0 else 0.0
        row_str += f"{robustness:9.1f}%"
        print(row_str)
    print("-" * len(header))

    print("\n=================================================================================")
    print("👑 REGIME PROFITABILITY MATRIX — PART 2: UPGRADED SETUP (MAX 3 CONCURRENT TRADES)")
    print("=================================================================================")
    print(header)
    print("-" * len(header))
    
    for model in competitors:
        row_str = f"{model:<15} | "
        profitable_count = 0
        total_scenarios = 0
        for sc in scenarios.keys():
            ret = matrix_3[model].get(sc, 0.0)
            row_str += f"{ret:+10.2f}% | "
            if ret >= -0.05:
                profitable_count += 1
            total_scenarios += 1
        
        robustness = (profitable_count / total_scenarios * 100.0) if total_scenarios > 0 else 0.0
        row_str += f"{robustness:9.1f}%"
        print(row_str)
    print("-" * len(header))
    
    print("\n💡 NOTE: Robustness represents the % of scenarios where the strategy preserved capital (Return >= -0.05%).")
    print("💡 Fees: maker_fee = 10.0 bps, taker_fee = 10.0 bps (0.1%). Exits: Maker TP Limit, Taker SL.")
    print("=================================================================================")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark execution interrupted by user.")
