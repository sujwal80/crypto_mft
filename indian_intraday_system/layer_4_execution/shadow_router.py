"""Shadow execution router executing simulated trades with flat fees and adverse slippage."""

import uuid
from typing import Dict, List
from indian_intraday_system.config import (
    SHADOW_LIMIT_ADVERSE_SELECTION,
    SHADOW_MARKET_SLIPPAGE_POINTS,
    SHADOW_ROUND_TRIP_FEE,
    round_to_nse_tick,
)
from indian_intraday_system.layer_4_execution.base_router import BaseRouter


class ShadowRouter(BaseRouter):
    """Simulates paper fills with institutional friction levels."""

    def __init__(self, starting_capital: float = 150000.0):
        self.capital = starting_capital
        self.balance = starting_capital
        self.positions: Dict[str, Dict] = {}
        self.active_orders: List[Dict] = []
        self.trade_log: List[Dict] = []
        self.transaction_history: List[Dict] = []

    def get_positions(self) -> List[Dict]:
        return list(self.positions.values())

    def get_funds(self) -> Dict:
        margin_locked = len(self.positions) * 120000.0
        buying_power = max(0.0, self.balance - margin_locked)
        return {
            "starting_capital": self.capital,
            "net_pnl": self.balance - self.capital,
            "balance": self.balance,
            "margin_locked": margin_locked,
            "buying_power": buying_power,
        }

    def place_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        order_type: str = "MARKET",
        price: float = None,
    ) -> Dict:
        action = action.upper()
        order_type = order_type.upper()
        order_id = str(uuid.uuid4())[:8]

        # All input prices are formatted to NSE tick increment (0.05 paise)
        if price is not None:
            price = round_to_nse_tick(price)

        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Limit price is required for LIMIT orders")
            order = {
                "order_id": order_id,
                "symbol": symbol,
                "action": action,
                "qty": qty,
                "order_type": "LIMIT",
                "price": price,
                "status": "OPEN",
            }
            self.active_orders.append(order)
            print(f"[Shadow] Limit {action} Placed: {qty} of {symbol} @ {price:.2f}. ID: {order_id}")
            return order

        elif order_type == "MARKET":
            if price is None:
                raise ValueError("Reference spot price is required for MARKET fills")

            # Market Order Slippage: BUY fills higher, SELL fills lower
            fill_price = (
                price + SHADOW_MARKET_SLIPPAGE_POINTS
                if action == "BUY"
                else price - SHADOW_MARKET_SLIPPAGE_POINTS
            )
            fill_price = round_to_nse_tick(fill_price)

            return self._execute_fill(order_id, symbol, action, qty, fill_price, "MARKET")

        return {"status": "REJECTED", "reason": "Unknown order type"}

    def cancel_order(self, order_id: str) -> bool:
        for order in self.active_orders:
            if order["order_id"] == order_id:
                self.active_orders.remove(order)
                print(f"[Shadow] Cancelled Limit Order: {order_id}")
                return True
        return False

    def emergency_square_off(self) -> Dict:
        """Cancels all pending limit orders and closes open positions immediately."""
        print("[Shadow] EMERGENCY SQUARE OFF ACTIVATED.")
        self.active_orders.clear()
        closed = 0

        for symbol, pos in list(self.positions.items()):
            qty = pos["qty"]
            ref_price = pos["entry_price"]
            exit_action = "SELL" if pos["side"] == "BUY" else "BUY"
            # Apply 1.0 pt emergency slippage penalty
            exit_price = (
                ref_price - 1.0 if exit_action == "SELL" else ref_price + 1.0
            )
            exit_price = round_to_nse_tick(exit_price)

            self._execute_fill(
                str(uuid.uuid4())[:8], symbol, exit_action, qty, exit_price, "MARKET"
            )
            closed += 1

        return {"status": "SUCCESS", "closed_count": closed, "funds": self.get_funds()}

    def process_feed_tick(self, symbol: str, last_price: float):
        """Processes limit orders against the latest tick price applying adverse selection."""
        last_price = round_to_nse_tick(last_price)

        for order in list(self.active_orders):
            if order["symbol"] != symbol:
                continue

            action = order["action"]
            limit_price = order["price"]
            qty = order["qty"]

            # Adverse Selection:
            # Buy limit fills ONLY if price falls BELOW limit_price - adverse_selection
            # Sell limit fills ONLY if price rises ABOVE limit_price + adverse_selection
            is_filled = False
            if action == "BUY" and last_price <= (limit_price - SHADOW_LIMIT_ADVERSE_SELECTION):
                is_filled = True
            elif action == "SELL" and last_price >= (limit_price + SHADOW_LIMIT_ADVERSE_SELECTION):
                is_filled = True

            if is_filled:
                print(
                    f"[Shadow] LIMIT FILL: {action} {qty} @ {limit_price:.2f} (Market Price: {last_price:.2f})"
                )
                self.active_orders.remove(order)
                self._execute_fill(
                    order["order_id"], symbol, action, qty, limit_price, "LIMIT"
                )

    def _execute_fill(
        self,
        order_id: str,
        symbol: str,
        action: str,
        qty: int,
        fill_price: float,
        order_type: str,
    ) -> Dict:
        fill_record = {
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "price": fill_price,
            "order_type": order_type,
        }
        self.transaction_history.append(fill_record)

        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "side": action,
                "qty": qty,
                "entry_price": fill_price,
            }
            print(f"[Shadow] NEW POSITION: {action} {qty} {symbol} @ {fill_price:.2f}")
        else:
            pos = self.positions[symbol]
            if pos["side"] == action:
                # Accumulate position size
                total_qty = pos["qty"] + qty
                pos["entry_price"] = (
                    (pos["entry_price"] * pos["qty"]) + (fill_price * qty)
                ) / total_qty
                pos["qty"] = total_qty
                print(f"[Shadow] ACCUMULATED POSITION: {qty} {symbol} @ {fill_price:.2f}")
            else:
                # Match / Close position
                if qty >= pos["qty"]:
                    closed_qty = pos["qty"]
                    rem_qty = qty - closed_qty

                    if pos["side"] == "BUY":
                        gross_pnl = (fill_price - pos["entry_price"]) * closed_qty
                    else:
                        gross_pnl = (pos["entry_price"] - fill_price) * closed_qty

                    # Subtract Indian Regulatory Friction flat fee: ₹90
                    net_pnl = gross_pnl - SHADOW_ROUND_TRIP_FEE
                    self.balance += net_pnl

                    trade = {
                        "symbol": symbol,
                        "side": pos["side"],
                        "qty": closed_qty,
                        "entry_price": pos["entry_price"],
                        "exit_price": fill_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "fee": SHADOW_ROUND_TRIP_FEE,
                    }
                    self.trade_log.append(trade)

                    print(
                        f"[Shadow] CLOSED POSITION: {symbol}. Gross: ₹{gross_pnl:.2f}. "
                        f"Net (after ₹{SHADOW_ROUND_TRIP_FEE} flat fee): ₹{net_pnl:.2f}"
                    )
                    del self.positions[symbol]

                    if rem_qty > 0:
                        self.positions[symbol] = {
                            "symbol": symbol,
                            "side": action,
                            "qty": rem_qty,
                            "entry_price": fill_price,
                        }
                        print(f"[Shadow] REVERSED POSITION: {action} {rem_qty} {symbol} @ {fill_price:.2f}")
                else:
                    # Partial close
                    pos["qty"] -= qty
                    if pos["side"] == "BUY":
                        gross_pnl = (fill_price - pos["entry_price"]) * qty
                    else:
                        gross_pnl = (pos["entry_price"] - fill_price) * qty

                    pro_rated_fee = (qty / (pos["qty"] + qty)) * SHADOW_ROUND_TRIP_FEE
                    net_pnl = gross_pnl - pro_rated_fee
                    self.balance += net_pnl

                    trade = {
                        "symbol": symbol,
                        "side": pos["side"],
                        "qty": qty,
                        "entry_price": pos["entry_price"],
                        "exit_price": fill_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "fee": pro_rated_fee,
                    }
                    self.trade_log.append(trade)
                    print(f"[Shadow] PARTIAL REDUCTION: {symbol} by {qty}. Net: ₹{net_pnl:.2f}")

        return {
            "status": "FILLED",
            "order_id": order_id,
            "fill_price": fill_price,
            "qty": qty,
        }
