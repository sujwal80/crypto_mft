"""Backtest runner executing intraday swing strategies on daily Bhavcopy files."""

import numpy as np
from indian_intraday_system import config
from indian_intraday_system.backtest.replay_engine import ReplayEngine
from indian_intraday_system.layer_2_macro.gex_mapper import map_gex_levels
from indian_intraday_system.layer_3_micro.cvd_engine import CVDEngine
from indian_intraday_system.layer_4_execution.shadow_router import ShadowRouter


class BacktestRunner:
    """Historical Backtesting engine matching live production event structures."""

    def __init__(self, starting_capital: float = 150000.0):
        self.replay_engine = ReplayEngine()
        self.router = ShadowRouter(starting_capital=starting_capital)
        self.cvd_engine = CVDEngine()

    def run_backtest_day(self, date_str: str, prev_close: float = 22000.0, force_eod_spot: float = None):
        """Runs an entire intraday simulated paper trading day."""
        print(f"\n=== STARTING BACKTEST DAY: {date_str} ===")
        print(f"Previous Close Spot: {prev_close:.2f}")

        self.cvd_engine.reset()

        tick_generator = self.replay_engine.generate_intraday_replay(
            date_str=date_str, prev_close=prev_close, force_eod_spot=force_eod_spot
        )

        for tick in tick_generator:
            spot = tick["spot"]
            vol = tick["volume"]
            minute = tick["minute_index"]

            self.cvd_engine.process_trade(
                price=spot, volume=vol, bid=spot - 0.5, ask=spot + 0.5
            )

            if minute % 10 == 0:
                gex_levels = map_gex_levels(
                    spot=spot,
                    strikes=tick["strikes"],
                    expiries_days=tick["expiries_days"],
                    ivs=tick["ivs"],
                    open_interests=tick["open_interests"],
                    option_types=tick["option_types"],
                )

                self._evaluate_strategy(spot, gex_levels, minute)

            self.router.process_feed_tick("NIFTY_FUT", spot)

            if minute == 360:
                self.router.emergency_square_off()

    def _evaluate_strategy(self, spot: float, gex_levels: dict, minute: int):
        """Intraday Swing Strategy rules."""
        if not (30 <= minute < 360):
            return

        call_wall = gex_levels["call_wall"]
        put_wall = gex_levels["put_wall"]
        cvd_ratio = self.cvd_engine.get_taker_ratio()

        positions = self.router.get_positions()

        # Bullish Squeeze
        if spot > call_wall and cvd_ratio > 0.20:
            if not positions:
                print(
                    f"  [STRATEGY] Minute {minute:03d}: Bullish Squeeze! "
                    f"Spot ({spot:.2f}) > CallWall ({call_wall:.2f}). Placing MARKET BUY."
                )
                self.router.place_order(
                    symbol="NIFTY_FUT",
                    action="BUY",
                    qty=config.LOT_SIZE_NIFTY,
                    order_type="MARKET",
                    price=spot,
                )

        # Bearish Breakdown
        elif spot < put_wall and cvd_ratio < -0.20:
            if not positions:
                print(
                    f"  [STRATEGY] Minute {minute:03d}: Bearish Breakdown! "
                    f"Spot ({spot:.2f}) < PutWall ({put_wall:.2f}). Placing MARKET SELL."
                )
                self.router.place_order(
                    symbol="NIFTY_FUT",
                    action="SELL",
                    qty=config.LOT_SIZE_NIFTY,
                    order_type="MARKET",
                    price=spot,
                )

        # Exits
        elif positions:
            pos = positions[0]
            side = pos["side"]
            zero_gamma = gex_levels["zero_gamma"]

            if side == "BUY" and spot < zero_gamma:
                print(
                    f"  [STRATEGY] Minute {minute:03d}: Exit Long. Spot ({spot:.2f}) < ZeroGamma ({zero_gamma:.2f})."
                )
                self.router.place_order(
                    symbol="NIFTY_FUT",
                    action="SELL",
                    qty=config.LOT_SIZE_NIFTY,
                    order_type="MARKET",
                    price=spot,
                )
            elif side == "SELL" and spot > zero_gamma:
                print(
                    f"  [STRATEGY] Minute {minute:03d}: Exit Short. Spot ({spot:.2f}) > ZeroGamma ({zero_gamma:.2f})."
                )
                self.router.place_order(
                    symbol="NIFTY_FUT",
                    action="BUY",
                    qty=config.LOT_SIZE_NIFTY,
                    order_type="MARKET",
                    price=spot,
                )

    def print_backtest_report(self):
        """Outputs comprehensive backtest performance metrics."""
        trades = self.router.trade_log
        funds = self.router.get_funds()

        print("\n" + "=" * 80)
        print("                INDIAN SWING TRADING SYSTEM: BACKTEST AUDIT REPORT             ")
        print("=" * 80)
        print(f"Starting Capital: INR {funds['starting_capital']:,.2f}")
        print(f"Ending Capital:   INR {funds['balance']:,.2f}")
        print(f"Net PnL:          INR {funds['net_pnl']:,.2f}")
        print(f"Total ROI:        {(funds['net_pnl'] / funds['starting_capital'] * 100):.2f}%")
        print(f"Total Trades:     {len(trades)}")

        if not trades:
            print("No trades executed during backtest.")
            print("=" * 80)
            return

        gross_profits = [t["gross_pnl"] for t in trades if t["gross_pnl"] > 0]
        gross_losses = [t["gross_pnl"] for t in trades if t["gross_pnl"] <= 0]
        net_pnl = [t["net_pnl"] for t in trades]
        fees = [t["fee"] for t in trades]

        winning_trades = sum(1 for p in net_pnl if p > 0)
        win_rate = (winning_trades / len(trades)) * 100

        print(f"Win Rate:         {win_rate:.2f}% ({winning_trades}/{len(trades)})")
        print(f"Total Fees/Friction Deducted: INR {sum(fees):,.2f} (₹90 flat per trade)")
        print(f"Average Net Trade PnL:        INR {np.mean(net_pnl):,.2f}")
        print(f"Largest Win:                  INR {max(net_pnl) if net_pnl else 0:,.2f}")
        print(f"Largest Loss:                 INR {min(net_pnl) if net_pnl else 0:,.2f}")
        print("=" * 80)

        print("\nTRADE HISTORY LOGS:")
        for idx, t in enumerate(trades):
            print(
                f"  Trade {idx+1:02d}: {t['side']} {t['qty']} Lots | "
                f"Entry: {t['entry_price']:.2f} | Exit: {t['exit_price']:.2f} | "
                f"Gross: ₹{t['gross_pnl']:.2f} | Net: ₹{t['net_pnl']:.2f}"
            )
        print("=" * 80)
