"""Time-based State Machine controlling options walls swing strategies and micro sniping."""

from indian_intraday_system.config import round_to_nse_tick
from indian_intraday_system.layer_4_execution.time_manager import TimeManager


class SwingStateMachine:
    """Asynchronous state transition machine managing GEX macro and CVD micro triggers."""

    def __init__(self, router, cvd_engine, time_manager: TimeManager = None):
        self.router = router
        self.cvd_engine = cvd_engine
        self.time_manager = time_manager or TimeManager()

        self.state = "LOCK"  # Initial state
        self.basis_z_threshold = 2.0  # Standard Basis tracking threshold

    def get_state(self) -> str:
        return self.state

    def update_state(self):
        """Transition state dynamically according to Time Regimes."""
        new_regime = self.time_manager.get_current_regime()
        if self.state != new_regime:
            print(f"[StateMachine] Transitioning state: {self.state} ---> {new_regime}")
            self.state = new_regime

            # If transitioning to KILL, run immediate hard flatten
            if self.state == "KILL":
                self.router.emergency_square_off()

    def evaluate_signals(self, spot: float, future_price: float, gex_levels: dict, basis_z_score: float, lot_size: int):
        """Evaluates triggers based on the active State."""
        # Keep state updated
        self.update_state()

        if self.state == "LOCK" or self.state == "KILL":
            return

        positions = self.router.get_positions()
        call_wall = gex_levels["call_wall"]
        put_wall = gex_levels["put_wall"]
        zero_gamma = gex_levels["zero_gamma"]
        cvd_ratio = self.cvd_engine.get_taker_ratio()

        # ==========================================================================
        # STATE 1: MEAN REVERSION REGIME (9:45 AM - 1:30 PM)
        # Target GEX pinning. Sell Call wall rejections, Buy Put wall rejections.
        # ==========================================================================
        if self.state == "MEAN_REVERSION":
            if not positions and abs(spot - put_wall) / spot <= 0.005:
                if cvd_ratio > 0.10 and basis_z_score >= -1.5:
                    print(
                        f"[StateMachine] [Mean-Reversion] Spot ({spot:.2f}) at Put Wall ({put_wall:.2f}). "
                        f"CVD support ({cvd_ratio:.3f}). Placing Limit BUY."
                    )
                    self.router.place_order(
                        symbol="NIFTY_FUT",
                        action="BUY",
                        qty=lot_size,
                        order_type="MARKET",
                        price=future_price,
                    )

            elif not positions and abs(spot - call_wall) / spot <= 0.005:
                if cvd_ratio < -0.10 and basis_z_score <= 1.5:
                    print(
                        f"[StateMachine] [Mean-Reversion] Spot ({spot:.2f}) at Call Wall ({call_wall:.2f}). "
                        f"CVD resistance ({cvd_ratio:.3f}). Placing Limit SELL."
                    )
                    self.router.place_order(
                        symbol="NIFTY_FUT",
                        action="SELL",
                        qty=lot_size,
                        order_type="MARKET",
                        price=future_price,
                    )

            # C. Mean Reversion Exits: Close trade as spot converges back to Zero-Gamma pinning
            elif positions:
                pos = positions[0]
                side = pos.get("side") or pos.get("transactionType")
                
                if side == "BUY" and spot >= zero_gamma:
                    print(f"[StateMachine] [Mean-Reversion Exit] Spot reached Zero-Gamma ({zero_gamma:.2f}). Closing Long.")
                    self.router.place_order(
                        symbol="NIFTY_FUT",
                        action="SELL",
                        qty=lot_size,
                        order_type="MARKET",
                        price=future_price,
                    )
                elif side == "SELL" and spot <= zero_gamma:
                    print(f"[StateMachine] [Mean-Reversion Exit] Spot reached Zero-Gamma ({zero_gamma:.2f}). Closing Short.")
                    self.router.place_order(
                        symbol="NIFTY_FUT",
                        action="BUY",
                        qty=lot_size,
                        order_type="MARKET",
                        price=future_price,
                    )

        # ==========================================================================
        # STATE 2: MOMENTUM BREAKOUT REGIME / 0DTE SQUEEZE (1:30 PM - 3:15 PM)
        # Ride dealer delta hedging sweeps past walls.
        # ==========================================================================
        elif self.state == "MOMENTUM":
            # A. Bullish Breakout: Spot punches ABOVE Call Wall AND CVD shows strong buying pressure
            # AND Basis confirms futures spec premium buying (Z-score > 0.5)
            if not positions and spot > call_wall:
                if cvd_ratio > 0.20 and basis_z_score >= 0.5:
                    print(
                        f"[StateMachine] [Momentum Squeeze] Spot ({spot:.2f}) crossed Call Wall ({call_wall:.2f}). "
                        f"CVD Buyer Aggression ({cvd_ratio:.3f}). Placing Market BUY."
                    )
                    self.router.place_order(
                        symbol="NIFTY_FUT",
                        action="BUY",
                        qty=lot_size,
                        order_type="MARKET",
                        price=future_price,
                    )

            # B. Bearish Breakdown: Spot punches BELOW Put Wall AND CVD shows strong selling pressure
            elif not positions and spot < put_wall:
                if cvd_ratio < -0.20 and basis_z_score <= -0.5:
                    print(
                        f"[StateMachine] [Momentum Breakdown] Spot ({spot:.2f}) crossed Put Wall ({put_wall:.2f}). "
                        f"CVD Seller Aggression ({cvd_ratio:.3f}). Placing Market SELL."
                    )
                    self.router.place_order(
                        symbol="NIFTY_FUT",
                        action="SELL",
                        qty=lot_size,
                        order_type="MARKET",
                        price=future_price,
                    )

            # C. Momentum Exits: Close if spot drops back inside GEX wall lines or crosses Zero-Gamma
            elif positions:
                pos = positions[0]
                side = pos.get("side") or pos.get("transactionType")

                if side == "BUY" and spot < zero_gamma:
                    print(f"[StateMachine] [Momentum Exit] Long trailing exit triggered (Spot < ZeroGamma). Closing.")
                    self.router.place_order(
                        symbol="NIFTY_FUT",
                        action="SELL",
                        qty=lot_size,
                        order_type="MARKET",
                        price=future_price,
                    )
                elif side == "SELL" and spot > zero_gamma:
                    print(f"[StateMachine] [Momentum Exit] Short trailing exit triggered (Spot > ZeroGamma). Closing.")
                    self.router.place_order(
                        symbol="NIFTY_FUT",
                        action="BUY",
                        qty=lot_size,
                        order_type="MARKET",
                        price=future_price,
                    )
