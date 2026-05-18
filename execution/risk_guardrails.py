import logging
from typing import Dict
from execution.dead_letter_queue import DeadLetterQueue

logger = logging.getLogger(__name__)

class RiskGuardrailEngine:
    """
    Enforces zero-tolerance deterministic risk checks (Maker-Critic architecture).

    Ensures no orders breach daily drawdown limits or deviate excessively from the mid-market price.
    """
    def __init__(self, dlq: DeadLetterQueue, max_drawdown_limit: float = 0.05):
        """
        Initializes RiskGuardrailEngine.

        Args:
            dlq: Target Dead Letter Queue to audit blocked orders.
            max_drawdown_limit: Standard capital allocation percentage ceiling (e.g., 0.05 for 5%).
        """
        self.dlq = dlq
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_peak_value = 100000.0
        self.current_portfolio_value = 100000.0

    def validate_order(self, proposed_order: Dict, current_mid_price: float) -> bool:
        """
        Validates whether proposed entry parameters satisfy risk guardrail metrics.

        Args:
            proposed_order: Proposed buy/sell order metadata mapping.
            current_mid_price: Baseline baseline asset valuation.

        Returns:
            bool: True if approved, False if rejected.
        """
        # Check 1: Daily Drawdown Circuit Breaker
        drawdown = (self.daily_peak_value - self.current_portfolio_value) / self.daily_peak_value
        if drawdown >= self.max_drawdown_limit:
            self.dlq.log_rejection(proposed_order, "CRITIC REJECT: Daily drawdown limit breached.")
            return False

        # Check 2: Fat Finger Price Collar (max 2% deviation from current mid-market price)
        limit_price = proposed_order.get("limit_price", current_mid_price)
        if limit_price > current_mid_price * 1.02 or limit_price < current_mid_price * 0.98:
            self.dlq.log_rejection(proposed_order, f"CRITIC REJECT: Price collar breached (Limit: {limit_price} | Mid: {current_mid_price}).")
            return False

        return True
