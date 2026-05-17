import logging
from typing import Dict, Optional
from .binance_execution_gateway import BinanceExecutionGateway
from core.exceptions import InsufficientFundsException, CriticalExecutionException

logger = logging.getLogger(__name__)

"""
Order Management System (OMS)
"""
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
            
            # Check for critical errors to raise and halt the system
            try:
                import ccxt
                ccxt_available = True
            except ImportError:
                ccxt_available = False
                
            if ccxt_available:
                if isinstance(e, ccxt.InsufficientFunds):
                    raise InsufficientFundsException(f"Insufficient balance on Binance: {e}")
                elif isinstance(e, (ccxt.AuthenticationError, ccxt.PermissionDenied)):
                    raise CriticalExecutionException(f"Invalid API Key, IP Whitelisting, or API permissions: {e}")
            
            # Fallback to substring matching on the string representation of the exception
            err_msg = str(e).lower()
            if "insufficient balance" in err_msg or "insufficient funds" in err_msg:
                raise InsufficientFundsException(f"Insufficient balance on Binance: {e}")
            elif "invalid api-key" in err_msg or "ip, or permissions" in err_msg or "permission" in err_msg:
                raise CriticalExecutionException(f"Invalid API Key, IP Whitelisting, or API permissions: {e}")
                
            return None

    """
    Fail-Safe Auto-Sell Liquidation (Flattening)
    """
    async def liquidate_all(self, current_inventory: Dict[str, float], average_entry_price: Optional[Dict[str, float]] = None, journal_path: Optional[str] = None, realized_pnl: float = 0.0):
        """Emergency safety routine. Liquidates all open positions to cash immediately upon shutdown or crash."""
        logger.critical("🚨 INITIATING FAIL-SAFE EMERGENCY LIQUIDATION (FLATTENING INVENTORY)...")
        
        for symbol, notional_holding in list(current_inventory.items()):
            if abs(notional_holding) > 0.0:
                logger.warning(f"⚠️ Open position detected for {symbol} (${notional_holding:.2f}). Executing market sell...")
                action = "SELL" if notional_holding > 0 else "BUY"
                emergency_order = {
                    "symbol": symbol,
                    "action": action,
                    "notional": abs(notional_holding),
                    "type": "market"  # Force market order for instant execution
                }
                try:
                    report = await self.gateway.send_order(emergency_order)
                    if report and report.get("status") == "FILLED":
                        exec_price = report.get("executed_price", 0.0)
                        executed_qty = report.get("executed_qty", abs(notional_holding))
                        
                        trade_pnl = 0.0
                        if average_entry_price and symbol in average_entry_price:
                            old_avg = average_entry_price[symbol]
                            if old_avg > 0.0 and exec_price > 0.0:
                                qty_closed = executed_qty / exec_price
                                direction = 1 if notional_holding > 0 else -1
                                trade_pnl = (exec_price - old_avg) * qty_closed * direction
                        
                        current_inventory[symbol] = 0.0
                        if average_entry_price and symbol in average_entry_price:
                            average_entry_price[symbol] = 0.0
                            
                        logger.warning(f"✅ Successfully Liquidated {symbol} (Action: {action}, Notional: {executed_qty}, Price: {exec_price}). Position flattened to cash. PnL: ${trade_pnl:.6f}")
                        
                        if journal_path:
                            from datetime import datetime
                            import json
                            cumulative = realized_pnl + trade_pnl
                            journal_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "symbol": symbol,
                                "action": action,
                                "executed_price": exec_price,
                                "executed_notional": executed_qty,
                                "trade_pnl": trade_pnl,
                                "cumulative_pnl": cumulative
                            }
                            try:
                                with open(journal_path, "a") as f:
                                    f.write(json.dumps(journal_entry) + "\n")
                                logger.info(f"Liquidated position journaled to {journal_path}")
                            except Exception as e:
                                logger.error(f"Failed to write liquidation to trades journal: {e}")
                except Exception as e:
                    logger.critical(f"❌ EMERGENCY LIQUIDATION FAILED FOR {symbol}: {e}")
            else:
                logger.info(f"No open position for {symbol}. Already flat.")
