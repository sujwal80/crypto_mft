import logging
from typing import Dict
from .dead_letter_queue import DeadLetterQueue

logger = logging.getLogger(__name__)

"""
Risk Guardrail Engine (The Critic)
"""
class RiskGuardrailEngine:
    """Enforces zero-tolerance deterministic risk checks (Maker-Critic architecture)."""
    def __init__(self, dlq: DeadLetterQueue, max_drawdown_limit: float = 0.05):
        self.dlq = dlq
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_peak_value = 100000.0
        self.current_portfolio_value = 100000.0
    
    """
    Validate Proposed Order
    """
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
