import logging
from typing import Dict, Optional
from execution.execution_gateway import BinanceExecutionGateway

logger = logging.getLogger(__name__)

class OrderManagementSystem:
    """Manages order state transitions and routes execution instructions to the active gateway."""
    def __init__(self, gateway: BinanceExecutionGateway):
        """
        Initializes OrderManagementSystem.

        Args:
            gateway: Active CCXT / Sim Execution Gateway.
        """
        self.gateway = gateway

    async def process_approved_order(self, approved_order: Dict) -> Optional[Dict]:
        """
        Processes an approved order payload through state transitions and sends to gateway.

        Args:
            approved_order: Validated target order payload dictionary.

        Returns:
            Optional[Dict]: High-fidelity execution report from gateway.
        """
        logger.info(f"OMS Received Approved Order -> {approved_order}. Initiating state machine...")
        approved_order["status"] = "PENDING_SUBMIT"
        try:
            execution_report = await self.gateway.send_order(approved_order)
            execution_report["status"] = "FILLED"
            logger.info(f"OMS EXECUTION REPORT -> {execution_report}")
            return execution_report
        except Exception as e:
            logger.error(f"OMS Gateway execution failed: {e}")
            return None

    async def liquidate_all(self, current_inventory: Dict[str, float]):
        """
        Emergency fail-safe liquidation handler. Flattens all inventories instantly to cash.

        Args:
            current_inventory: Reference dictionary containing symbol string keys mapped to float quantities.
        """
        logger.critical("🚨 INITIATING FAIL-SAFE EMERGENCY LIQUIDATION (FLATTENING INVENTORY...)")
        for symbol, crypto_units in current_inventory.items():
            if crypto_units > 0.0:
                logger.warning(f"⚠️ Open position detected for {symbol} ({crypto_units:.6f} units). Executing market sell...")
                emergency_order = {
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": crypto_units,
                    "type": "market"
                }
                try:
                    await self.gateway.send_order(emergency_order)
                    logger.info(f"Emergency liquidation market sell successfully routed for {symbol}.")
                except Exception as e:
                    logger.critical(f"🛑 CRITICAL: EMERGENCY LIQUIDATION ROUTE FAILED FOR {symbol}: {e}!")
