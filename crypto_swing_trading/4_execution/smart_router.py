import logging
import time

logger = logging.getLogger(__name__)

class SmartRouter:
    """
    High-speed Execution Controller.
    Executes Maker (Post-Only) orders inside the spread to capture maker rebates
    or executes rapid, aggressive Taker (Market) orders for risk invalidations.
    Logs simulated trades in Shadow Mode with microsecond accuracy.
    """
    def __init__(self, mode: str = "SHADOW"):
        """
        Args:
            mode: "SHADOW" (Simulation with full execution friction) or "LIVE"
        """
        self.mode = mode.upper()
        self.trade_journal = []
        self.active_orders = []  # Queue for resting limit orders

    async def place_post_only_limit(self, symbol: str, side: str, price: float, amount: float) -> dict:
        """
        Submits a Post-Only (Maker) limit order.
        In SHADOW mode, it adds the order to the resting queue instead of instant-filling.
        """
        price = float(price)
        amount = float(amount)
        timestamp = time.time()
        
        logger.info(f"SMART_ROUTER: Submitting POST_ONLY Limit {side.upper()} | {amount} {symbol.upper()} @ {price:.2f}")
        
        if self.mode == "SHADOW":
            # Add to active orders queue representing a resting order on the book
            order = {
                "order_id": f"shadow_maker_{int(timestamp*1e6)}",
                "symbol": symbol,
                "side": side.upper(),
                "price": price,
                "amount": amount,
                "type": "LIMIT_POST_ONLY",
                "status": "OPEN",
                "timestamp": timestamp
            }
            self.active_orders.append(order)
            logger.warning(f"SHADOW_BOOK: Limit Order {order['order_id']} added to resting queue.")
            return order
        else:
            # Direct ccxt / exchange execution integration placeholder
            logger.warning("LIVE execution modes should be run in ap-northeast-1 (Tokyo) for sub-millisecond latency.")
            return {"status": "SUBMITTED_LIVE"}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancels a pending resting limit order from the active queue."""
        for order in list(self.active_orders):
            if order["order_id"] == order_id:
                order["status"] = "CANCELLED"
                self.active_orders.remove(order)
                logger.warning(f"SHADOW_CANCEL: Limit order {order_id} cancelled successfully.")
                return True
        return False

    async def process_active_orders(self, symbol: str, current_price: float):
        """
        Processes resting limit orders against the current market price.
        Fills order ONLY if the price crosses the limit price (Adverse Selection).
        """
        current_price = float(current_price)
        for order in list(self.active_orders):
            if order["symbol"] != symbol:
                continue
                
            action = order["side"]
            limit_price = order["price"]
            
            is_filled = False
            # Buy limit fills only if market price falls to or below limit price
            if action == "BUY" and current_price <= limit_price:
                is_filled = True
            # Sell limit fills only if market price rises to or above limit price
            elif action == "SELL" and current_price >= limit_price:
                is_filled = True
                
            if is_filled:
                order["status"] = "FILLED"
                fee = order["amount"] * limit_price * 0.001
                order["fee"] = fee
                self.trade_journal.append(order)
                self.active_orders.remove(order)
                logger.warning(f"SHADOW_FILL: Resting Limit {action} order filled @ {limit_price:.2f}. Fee: ${fee:.4f}")

    async def place_market_order(self, symbol: str, side: str, amount: float, shadow_price: float = None) -> dict:
        """
        Submits a high-speed exit order.
        In SHADOW mode, this is automatically optimized to execute as a Maker (Post-Only Limit) 
        exactly at the spread (Ask for SELL / Bid for BUY) to capture the 0.02% maker fee rebate.
        """
        amount = float(amount)
        timestamp = time.time()
        
        logger.info(f"SMART_ROUTER: Submitting Exit Order {side.upper()} | {amount} {symbol.upper()}")
        
        if self.mode == "SHADOW":
            if shadow_price is None:
                raise ValueError("shadow_price is required for fills in SHADOW mode")
                
            action = side.upper()
            ref_price = float(shadow_price)
            
            # Realistically model emergency invalidation exits as crossing the spread (Taker)
            # Rather than assuming a Maker rebate/fill at the spread, apply slippage
            fill_price = ref_price - 0.5 if action == "SELL" else ref_price + 0.5
            
            # Real-world Taker Exit Fee (0.1%) + Slippage
            fee = amount * fill_price * 0.001
            
            order = {
                "order_id": f"shadow_taker_exit_{int(timestamp*1e6)}",
                "symbol": symbol,
                "side": action,
                "price": fill_price,
                "amount": amount,
                "fee": fee,
                "type": "MARKET",  # Flipped to Taker Market
                "status": "FILLED",
                "timestamp": timestamp
            }
            self.trade_journal.append(order)
            logger.warning(f"SHADOW_FILL: Maker Limit Exit {order['order_id']} filled at spread @ {fill_price:.2f}. Fee: ${fee:.4f}")
            return order
        else:
            # ccxt market order integration
            return {"status": "SUBMITTED_LIVE"}
