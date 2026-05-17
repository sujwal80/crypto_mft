import asyncio
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

"""
Real Execution Gateway (CCXT / REST API)
"""
class BinanceExecutionGateway:
    """Interacts with live Binance execution endpoints via CCXT Pro. Supports Paper Trading mode."""
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, paper_trading: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        self.exchange = None
        self._init_exchange()
        
    def _init_exchange(self):
        if not self.paper_trading and self.api_key and self.api_secret:
            try:
                import ccxt.pro as ccxtpro
                self.exchange = ccxtpro.binance({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                logger.info("CCXT Pro Live Binance Exchange Gateway Initialized successfully.")
            except ImportError:
                logger.error("ccxt library not found. Cannot initialize live execution gateway.")
            except Exception as e:
                logger.error(f"Failed to initialize CCXT Exchange Gateway: {e}")

    async def send_order(self, order_payload: Dict) -> Dict:
        """Executes live order placement via CCXT if paper_trading=False, otherwise simulates fills."""
        symbol = order_payload.get("symbol")
        action = order_payload.get("action")
        notional = order_payload.get("notional", 0.0)
        limit_price = order_payload.get("limit_price", 0.0)
        order_type = order_payload.get("type", "limit")
        is_market_order = (order_type == "market") or (limit_price == 0.0)

        if self.paper_trading or not self.exchange:
            logger.info(f"Gateway routing order to Binance (Paper Trading: {self.paper_trading})...")
            await asyncio.sleep(0.01) # Simulating 10ms execution network latency
            return {
                "order_id": f"PAPER_{int(time.time() * 1000)}",
                "symbol": symbol,
                "action": action,
                "executed_price": limit_price,  # 0.0 for market orders in paper mode
                "executed_qty": notional,
                "status": "FILLED",
                "execution_timestamp": int(time.time() * 1e9)
            }
        else:
            logger.warning(f"⚠️ LIVE EXECUTION ROUTING -> Sending {action} {notional} {symbol} to Binance!")
            try:
                if is_market_order:
                    # BUG-04 FIX: Fetch current price for market orders to compute quantity
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_price = ticker.get('last') or ticker.get('bid') or 0.0
                    if current_price <= 0:
                        raise ValueError(f"Cannot compute quantity: invalid market price {current_price} for {symbol}")
                    quantity = notional / current_price
                    order = await self.exchange.create_order(
                        symbol=symbol, type='market', side=action.lower(), amount=quantity
                    )
                else:
                    if limit_price <= 0:
                        raise ValueError(f"Cannot compute quantity: limit_price is {limit_price}")
                    quantity = notional / limit_price
                    order = await self.exchange.create_order(
                        symbol=symbol, type='limit', side=action.lower(),
                        amount=quantity, price=limit_price
                    )

                # BUG-05 FIX: filled is in base asset (BTC); avg_fill_price converts to notional (USD)
                filled_base_qty = order.get('filled', 0.0)
                avg_fill_price = order.get('average') or limit_price
                logger.info(f"LIVE ORDER PLACED SUCCESSFULLY -> Order ID: {order['id']}")
                return {
                    "order_id": order['id'],
                    "symbol": symbol,
                    "action": action,
                    "executed_price": avg_fill_price,
                    "executed_qty": filled_base_qty * avg_fill_price,  # Notional USD value
                    "status": order.get('status', 'FILLED').upper(),
                    "execution_timestamp": int(time.time() * 1e9)
                }
            except Exception as e:
                logger.error(f"CRITICAL LIVE EXECUTION FAILURE: {e}")
                raise e

    async def close(self):
        if self.exchange:
            await self.exchange.close()

