"""Production Dhan HQ API router incorporating pre-trade margin loops and Terminal Stop-Loss orders."""

import time
from typing import Dict, List
from dhanhq import DhanContext, dhanhq
from indian_intraday_system.config import (
    DHAN_ACCESS_TOKEN,
    DHAN_CLIENT_ID,
    round_to_nse_tick,
)
from indian_intraday_system.layer_4_execution.base_router import BaseRouter


class DhanRouter(BaseRouter):
    """Live production interface for Dhan HQ API with statistical margin safety nets."""

    def __init__(self, client_id: str = None, access_token: str = None):
        self.client_id = client_id or DHAN_CLIENT_ID
        self.access_token = access_token or DHAN_ACCESS_TOKEN

        self.is_sandbox = (
            self.client_id == "placeholder_client_id"
            or "placeholder" in self.access_token
        )

        if self.is_sandbox:
            print("[Dhan] Sandbox mode enabled. Using Mock contexts.")
            self.client = None
        else:
            print("[Dhan] Connecting Live API endpoints...")
            context = DhanContext(self.client_id, self.access_token)
            self.client = dhanhq(context)

        self.active_stop_losses: Dict[str, str] = {}

    def get_positions(self) -> List[Dict]:
        if self.is_sandbox:
            return []
        try:
            resp = self.client.get_positions()
            if resp.get("status") == "success":
                return resp.get("data", [])
            return []
        except Exception as e:
            print(f"[Dhan] Error fetching positions: {e}")
            return []

    def get_funds(self) -> Dict:
        if self.is_sandbox:
            return {"buying_power": 150000.0, "balance": 150000.0}
        try:
            resp = self.client.get_fund_limits()
            if resp.get("status") == "success":
                data = resp.get("data", {})
                return {
                    "buying_power": float(data.get("sodLimit", 0.0)),
                    "balance": float(data.get("availabelBalance", 0.0)),
                }
            return {}
        except Exception as e:
            print(f"[Dhan] Error fetching margins: {e}")
            return {}

    def _check_margin_loop(self, required: float = 120000.0) -> bool:
        for attempt in range(3):
            funds = self.get_funds()
            power = funds.get("buying_power", 0.0)
            if power >= required:
                return True
            print(
                f"[Dhan] Check {attempt+1}/3: Insufficient power. "
                f"Available: ₹{power:.2f}, Required: ₹{required:.2f}. Retrying in 5s..."
            )
            time.sleep(5)
        return False

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

        # Ensure price is formatted to valid NSE tick (0.05)
        if price is not None:
            price = round_to_nse_tick(price)

        if not self._check_margin_loop(120000.0):
            print("[Dhan] Order blocked: Pre-trade margin checks failed.")
            return {"status": "REJECTED", "reason": "INSUFFICIENT_MARGIN"}

        security_id = "52"  # Mock security ID lookup
        exchange_segment = "NSE_FNO"

        if self.is_sandbox:
            order_id = f"sb_ord_{int(time.time())}"
            fill_price = price or 22000.0
            print(f"[Dhan] [Sandbox] Filled {action} {qty} {symbol} @ {fill_price:.2f}")

            # Place Sandbox Terminal Stop Loss
            self._place_terminal_stop_loss(
                parent_order_id=order_id,
                security_id=security_id,
                exchange_segment=exchange_segment,
                entry_price=fill_price,
                qty=qty,
                side=action,
            )

            return {
                "status": "FILLED",
                "order_id": order_id,
                "fill_price": fill_price,
                "qty": qty,
            }

        try:
            response = self.client.place_order(
                security_id=security_id,
                exchange_segment=exchange_segment,
                transaction_type=action,
                quantity=qty,
                order_type=order_type,
                product_type="INTRADAY",
                price=price or 0.0,
                trigger_price=0.0,
            )

            if response.get("status") == "success":
                order_id = response["data"]["orderId"]
                time.sleep(0.2)  # short buffer for fill
                details = self.client.get_order_by_id(order_id)
                fill_price = round_to_nse_tick(
                    float(details.get("data", {}).get("price", price or 0.0))
                )

                # Place Live Terminal Stop-Loss to cap risk immediately at the exchange
                self._place_terminal_stop_loss(
                    parent_order_id=order_id,
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    entry_price=fill_price,
                    qty=qty,
                    side=action,
                )

                return {
                    "status": "FILLED",
                    "order_id": order_id,
                    "fill_price": fill_price,
                    "qty": qty,
                }
            else:
                return {"status": "REJECTED", "reason": response.get("remarks")}

        except Exception as e:
            return {"status": "ERROR", "reason": str(e)}

    def _place_terminal_stop_loss(
        self,
        parent_order_id: str,
        security_id: str,
        exchange_segment: str,
        entry_price: float,
        qty: int,
        side: str,
    ):
        """Submits a hard exchange-level Stop-Loss limit order to the matching engine."""
        sl_points = 30.0
        sl_action = "SELL" if side == "BUY" else "BUY"
        trigger_price = (
            entry_price - sl_points if side == "BUY" else entry_price + sl_points
        )
        limit_price = (
            trigger_price - 2.0 if side == "BUY" else trigger_price + 2.0
        )

        # Enforce round NSE tick formats
        trigger_price = round_to_nse_tick(trigger_price)
        limit_price = round_to_nse_tick(limit_price)

        print(
            f"[Dhan] Submitting exchange Stop-Loss: {sl_action} {qty} lots @ "
            f"Trigger: {trigger_price:.2f}, Limit: {limit_price:.2f}"
        )

        if self.is_sandbox:
            self.active_stop_losses[parent_order_id] = f"sb_sl_{parent_order_id}"
            return

        try:
            response = self.client.place_order(
                security_id=security_id,
                exchange_segment=exchange_segment,
                transaction_type=sl_action,
                quantity=qty,
                order_type="STOP_LOSS",
                product_type="INTRADAY",
                price=limit_price,
                trigger_price=trigger_price,
            )
            if response.get("status") == "success":
                sl_order_id = response["data"]["orderId"]
                self.active_stop_losses[parent_order_id] = sl_order_id
            else:
                print(f"[Dhan] CRITICAL: Live Stop-Loss placement failed: {response}")
        except Exception as e:
            print(f"[Dhan] CRITICAL: Error submitting live Stop-Loss: {e}")

    def cancel_order(self, order_id: str) -> bool:
        if self.is_sandbox:
            return True
        try:
            resp = self.client.cancel_order(order_id)
            return resp.get("status") == "success"
        except Exception:
            return False

    def emergency_square_off(self) -> Dict:
        print("[Dhan] EMERGENCY LIQUIDATION SEQUENCE ENGAGED.")
        if self.is_sandbox:
            self.active_stop_losses.clear()
            return {"status": "SUCCESS", "closed": 0}

        # 1. Cancel all pending orders
        try:
            orders_resp = self.client.get_order_list()
            if orders_resp.get("status") == "success":
                for ord in orders_resp.get("data", []):
                    if ord.get("orderStatus") in ("PENDING", "TRG PND"):
                        self.client.cancel_order(ord["orderId"])
        except Exception as e:
            print(f"[Dhan] Error canceling active orders: {e}")

        # 2. Flatten active open positions at worst-case market
        closed = []
        try:
            positions = self.get_positions()
            for pos in positions:
                qty = int(pos.get("netQty", 0))
                if qty != 0:
                    action = "SELL" if qty > 0 else "BUY"
                    resp = self.client.place_order(
                        security_id=pos.get("securityId"),
                        exchange_segment=pos.get("exchangeSegment"),
                        transaction_type=action,
                        quantity=abs(qty),
                        order_type="MARKET",
                        product_type="INTRADAY",
                        price=0.0,
                    )
                    closed.append(resp)
        except Exception as e:
            print(f"[Dhan] Error squaring positions: {e}")

        self.active_stop_losses.clear()
        return {"status": "COMPLETED", "details": closed}
