import asyncio
import logging
import time
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class BinanceExecutionGateway:
    """Interacts with live Binance execution endpoints via CCXT Pro. Supports Paper Trading mode."""
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, paper_trading: bool = True):
        """
        Initializes BinanceExecutionGateway.

        Args:
            api_key: Live exchange key.
            api_secret: Live exchange secret.
            paper_trading: If True, routes orders through high-fidelity latency/fee/slippage simulator.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        self.max_slippage_pct = 0.0010 # Protective collar percentage margin (e.g., 0.10%)
        self.exchange = None
        self.init_exchange()

    def init_exchange(self):
        if not self.paper_trading and self.api_key and self.api_secret:
            try:
                import ccxt.pro as ccxtpro
                self.exchange = ccxtpro.binance({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                logger.info("CCXT Pro Live Binance Exchange Gateway initialized successfully.")
            except ImportError:
                logger.error("ccxt library not found. Cannot initialize live execution gateway.")
            except Exception as e:
                logger.error(f"Failed to initialize CCXT Exchange Gateway: {e}")

    async def send_order(self, order_payload: Dict) -> Dict:
        """Executes order placement via CCXT (live) or SIM (paper), handling unified cash/crypto sizing, fees, and latency."""
        symbol = order_payload.get("symbol")
        action = order_payload.get("action")
        order_type = order_payload.get("type", "limit")
        original_type = order_type
        limit_price = order_payload.get("limit_price", 0.0)
        mid_price = order_payload.get("mid_price", limit_price if limit_price > 0.0 else 65000.0)

        # Unified Slippage Control: Convert market orders into protective collared limit orders
        if order_type == "market":
            if action == "BUY":
                limit_price = mid_price * (1.0 + self.max_slippage_pct)
            else:
                limit_price = mid_price * (1.0 - self.max_slippage_pct)
            order_type = "limit"
            logger.info(f"Slippage Control: Converted market {action} order to protective collared limit at {limit_price:.2f}")

        # Resolve Target Price
        base_price = limit_price if limit_price > 0.0 else mid_price

        # Resolve Sizing (Unified Cash vs Crypto Units)
        if "quantity" in order_payload:
            quantity = float(order_payload["quantity"])
            notional = quantity * base_price
        else:
            notional = float(order_payload.get("notional", 0.0))
            quantity = notional / base_price

        # Simulation Configuration Parameters
        SIM_LATENCY_MS = 100          # 100ms round-trip latency
        SIM_MAKER_FEE_RATE = 0.0002   # 0.02% Maker Fee
        SIM_TAKER_FEE_RATE = 0.0004   # 0.04% Taker Fee
        SIM_SLIPPAGE_RATE = 0.0001    # 0.01% Slippage std dev

        if self.paper_trading or not self.exchange:
            logger.info(f"Gateway routing {order_type.upper()} order to Binance SIM (Simulated Latency: {SIM_LATENCY_MS}ms)...")

            # 1. Simulate Network Latency Delay
            await asyncio.sleep(SIM_LATENCY_MS / 1000.0)

            # 2. Simulate Price Drift (Slippage) during Latency
            drift_direction = 1 if action == "BUY" else -1
            random_slippage = np.random.normal(loc=0.00005, scale=SIM_SLIPPAGE_RATE)
            simulated_slippage_multiplier = 1 + (drift_direction * max(0.0, random_slippage))
            executed_price = base_price * simulated_slippage_multiplier

            # 3. High-Fidelity Execution Starvation Check
            # We ONLY cancel the order due to starvation if it was originally a MARKET order
            if original_type == "market" and limit_price > 0.0:
                if action == "BUY" and executed_price > limit_price:
                    logger.warning(f"Paper Starvation: Price drifted to {executed_price:.2f} past buy collar limit {limit_price:.2f}. Order cancelled.")
                    return {
                        "order_id": f"SIM-{int(time.time() * 1000)}",
                        "symbol": symbol,
                        "action": action,
                        "status": "CANCELLED",
                        "execution_timestamp": int(time.time() * 1e9)
                    }
                elif action == "SELL" and executed_price < limit_price:
                    logger.warning(f"Paper Starvation: Price drifted to {executed_price:.2f} past sell collar limit {limit_price:.2f}. Order cancelled.")
                    return {
                        "order_id": f"SIM-{int(time.time() * 1000)}",
                        "symbol": symbol,
                        "action": action,
                        "status": "CANCELLED",
                        "execution_timestamp": int(time.time() * 1e9)
                    }

            # 4. Dynamic Fee & Fill Sizing
            fee_rate = SIM_MAKER_FEE_RATE if order_type == "limit" else SIM_TAKER_FEE_RATE
            fee_paid = notional * fee_rate

            if action == "BUY":
                executed_qty_cash = notional
                executed_qty_crypto = (notional - fee_paid) / executed_price
            else:
                executed_qty_crypto = quantity
                executed_qty_cash = (quantity * executed_price) - fee_paid

            logger.info(
                f"SIM Fill -> {action} {symbol} filled. "
                f"Target Price: {base_price:.2f} | Executed: {executed_price:.2f} | "
                f"Slippage: {((executed_price - base_price)/base_price):+4%} | "
                f"Fee Paid: ${fee_paid:.4f} ({fee_rate:.2%})"
            )

            return {
                "order_id": f"SIM-{int(time.time() * 1000)}",
                "symbol": symbol,
                "action": action,
                "executed_price": executed_price,
                "executed_qty_cash": executed_qty_cash,
                "executed_qty_crypto": executed_qty_crypto,
                "fee_paid": fee_paid,
                "status": "FILLED",
                "execution_timestamp": int(time.time() * 1e9)
            }
        else:
            logger.warning(f"⚠️ LIVE EXECUTION ROUTING -> Sending {action} {quantity:.6f} {symbol} ({order_type.upper()}) to Binance!")
            try:
                if order_type == "market":
                    order = await self.exchange.create_order(
                        symbol=symbol,
                        type="market",
                        side=action.lower(),
                        amount=quantity
                    )
                else:
                    order = await self.exchange.create_order(
                        symbol=symbol,
                        type="limit",
                        side=action.lower(),
                        amount=quantity,
                        price=limit_price
                    )

                logger.info(f"LIVE ORDER PLACED SUCCESSFULLY -> Order ID: {order['id']}")
                actual_price = order.get("average", limit_price)
                actual_filled_crypto = order.get("filled", quantity)
                actual_filled_cash = actual_filled_crypto * actual_price
                fee_paid = actual_filled_cash * (SIM_MAKER_FEE_RATE if order_type == "limit" else SIM_TAKER_FEE_RATE)

                return {
                    "order_id": order["id"],
                    "symbol": symbol,
                    "action": action,
                    "executed_price": actual_price,
                    "executed_qty_cash": actual_filled_cash - (fee_paid if action == "SELL" else 0.0),
                    "executed_qty_crypto": actual_filled_crypto - (fee_paid / actual_price if action == "BUY" else 0.0),
                    "fee_paid": fee_paid,
                    "status": order.get("status", "FILLED").upper(),
                    "execution_timestamp": int(time.time() * 1e9)
                }
            except Exception as e:
                logger.error(f"CRITICAL LIVE EXECUTION FAILURE: {e}")
                raise e

    async def close(self):
        if self.exchange:
            await self.exchange.close()
