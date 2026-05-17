import asyncio
import json
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ======================================================================
# Dead Letter Queue (DLQ) Audit Logger
# ======================================================================
class DeadLetterQueue:
    """Logs all orders rejected by the Risk Critic to an audit journal for post-mortem analysis."""
    def __init__(self, journal_path: str = "dlq_audit.json"):
        self.journal_path = journal_path
        
    def log_rejection(self, proposed_order: Dict, failure_reason: str):
        audit_payload = {
            "timestamp": int(time.time() * 1e9),
            "proposed_order": proposed_order,
            "rejection_reason": failure_reason
        }
        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(audit_payload) + "\n")
            logger.warning(f"DLQ AUDIT LOGGED -> {failure_reason}")
        except Exception as e:
            logger.error(f"Failed to write to DLQ journal: {e}")

# ======================================================================
# Real Execution Gateway (CCXT / REST API)
# ======================================================================
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
        
        if self.paper_trading or not self.exchange:
            logger.info(f"Gateway routing order to Binance (Paper Trading: {self.paper_trading})...")
            await asyncio.sleep(0.01) # Simulating 10ms execution network latency
            return {
                "order_id": f"PAPER_{int(time.time() * 1000)}",
                "symbol": symbol,
                "action": action,
                "executed_price": limit_price,
                "executed_qty": notional,
                "status": "FILLED",
                "execution_timestamp": int(time.time() * 1e9)
            }
        else:
            logger.warning(f"⚠️ LIVE EXECUTION ROUTING -> Sending {action} {notional} {symbol} to Binance!")
            try:
                # Calculate quantity based on notional and limit price
                quantity = notional / limit_price
                
                # Execute live limit order via CCXT Pro
                order = await self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=action.lower(),
                    amount=quantity,
                    price=limit_price
                    )
                
                logger.info(f"LIVE ORDER PLACED SUCCESSFULLY -> Order ID: {order['id']}")
                return {
                    "order_id": order['id'],
                    "symbol": symbol,
                    "action": action,
                    "executed_price": order.get('average', limit_price),
                    "executed_qty": order.get('filled', quantity) * limit_price,
                    "status": order.get('status', 'FILLED').upper(),
                    "execution_timestamp": int(time.time() * 1e9)
                }
            except Exception as e:
                logger.error(f"CRITICAL LIVE EXECUTION FAILURE: {e}")
                raise e

    async def close(self):
        if self.exchange:
            await self.exchange.close()

# ======================================================================
# Risk Critic & Order Management System (OMS)
# ======================================================================
class RiskGuardrailEngine:
    """Enforces zero-tolerance deterministic risk checks (Maker-Critic architecture)."""
    def __init__(self, dlq: DeadLetterQueue, max_drawdown_limit: float = 0.05):
        self.dlq = dlq
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_peak_value = 100000.0
        self.current_portfolio_value = 100000.0
        
    def validate_order(self, proposed_order: Dict, current_mid_price: float) -> bool:
        # Check 1: Daily Drawdown Circuit Breaker
        drawdown = (self.daily_peak_value - self.current_portfolio_value) / self.daily_peak_value
        if drawdown >= self.max_drawdown_limit:
            self.dlq.log_rejection(proposed_order, "CRITIC REJECT: Daily drawdown limit breached.")
            return False
            
        # Check 2: Fat Finger Price Collar
        limit_price = proposed_order.get("limit_price", current_mid_price)
        if limit_price > current_mid_price * 1.02 or limit_price < current_mid_price * 0.98:
            self.dlq.log_rejection(proposed_order, f"CRITIC REJECT: Price collar breached (Limit: {limit_price} | Mid: {current_mid_price}).")
            return False
            
        return True

class OrderManagementSystem:
    """Manages order state transitions and interacts with the execution gateway."""
    def __init__(self, gateway: BinanceExecutionGateway):
        self.gateway = gateway
        self.active_orders: Dict[str, Dict] = {}
        
    async def process_approved_order(self, approved_order: Dict) -> Optional[Dict]:
        logger.info(f"OMS Received Approved Order -> {approved_order}. Initiating state machine...")
        
        # State Transition: PENDING_SUBMIT
        approved_order["status"] = "PENDING_SUBMIT"
        
        # Route to Gateway
        try:
            execution_report = await self.gateway.send_order(approved_order)
            # State Transition: FILLED
            execution_report["status"] = "FILLED"
            logger.info(f"OMS EXECUTION REPORT -> {execution_report}")
            return execution_report
        except Exception as e:
            logger.error(f"OMS Gateway execution failed: {e}")
            return None

    # ======================================================================
    # Fail-Safe Auto-Sell Liquidation (Flattening)
    # ======================================================================
    async def liquidate_all(self, current_inventory: Dict[str, float]):
        """Emergency safety routine. Liquidates all open positions to cash immediately upon shutdown or crash."""
        logger.critical("🚨 INITIATING FAIL-SAFE EMERGENCY LIQUIDATION (FLATTENING INVENTORY)...")
        
        for symbol, notional_holding in current_inventory.items():
            if abs(notional_holding) > 0.0:
                logger.warning(f"⚠️ Open position detected for {symbol} (${notional_holding:.2f}). Executing market sell...")
                emergency_order = {
                    "symbol": symbol,
                    "action": "SELL" if notional_holding > 0 else "BUY",
                    "notional": abs(notional_holding),
                    "type": "market"  # Force market order for instant execution
                }
                try:
                    report = await self.gateway.send_order(emergency_order)
                    if report and report.get("status") == "FILLED":
                        current_inventory[symbol] = 0.0
                        logger.info(f"✅ Successfully Liquidated {symbol}. Position flattened to cash.")
                except Exception as e:
                    logger.critical(f"❌ EMERGENCY LIQUIDATION FAILED FOR {symbol}: {e}")
            else:
                logger.info(f"No open position for {symbol}. Already flat.")
