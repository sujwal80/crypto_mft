import numpy as np
from typing import Dict, Optional

class OrderGenerator:
    """
    Generates bracket entry orders consisting of entry, Take-Profit, and Stop-Loss limits.
    """
    def __init__(self, tp_margin: float = 0.006, sl_margin: float = 0.003):
        """
        Initializes OrderGenerator.
        
        Args:
            tp_margin: Take Profit percentage boundary (default 0.6%).
            sl_margin: Stop Loss percentage boundary (default 0.3%).
        """
        self.tp_margin = tp_margin
        self.sl_margin = sl_margin

    def generate_bracket_order(
        self,
        symbol: str,
        target_weight: float,
        portfolio_value: float,
        bid: float,
        ask: float,
        volatility: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Generates proposed entry bracket order dictionary.
        """
        if abs(target_weight) < 0.05:
            return None

        action = "BUY" if target_weight > 0.0 else "SELL"
        notional = abs(target_weight) * portfolio_value
        
        if notional < 10.0:
            return None

        limit_price = bid if action == "BUY" else ask

        # Dynamic Volatility Scaling (Refinement 2)
        tp_margin = self.tp_margin
        sl_margin = self.sl_margin
        if volatility is not None and volatility > 0.0:
            # Normalize volatility against a 1.5 bps baseline
            scale_factor = volatility / 0.00015
            tp_margin = max(0.0002, self.tp_margin * scale_factor)  # Floor TP at 2 bps
            sl_margin = max(0.0001, self.sl_margin * scale_factor)  # Floor SL at 1 bps

        if action == "BUY":
            take_profit_price = limit_price * (1.0 + tp_margin)
            stop_loss_price = limit_price * (1.0 - sl_margin)
        else: # SELL
            take_profit_price = limit_price * (1.0 - tp_margin)
            stop_loss_price = limit_price * (1.0 + sl_margin)

        return {
            "symbol": symbol,
            "action": action,
            "notional": notional,
            "limit_price": limit_price,
            "type": "limit",
            "bracket": {
                "entry_price": limit_price,
                "take_profit_price": take_profit_price,
                "stop_loss_price": stop_loss_price
            }
        }
