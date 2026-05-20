import numpy as np
from typing import Dict, Optional

class OrderGenerator:
    """
    Generates bracket entry orders consisting of entry, Take-Profit, and Stop-Loss limits.
    """
    def __init__(
        self, 
        tp_margin: float = 0.006, 
        sl_margin: float = 0.003,
        tp_margin_long: Optional[float] = None,
        tp_margin_short: Optional[float] = None,
        sl_margin_long: Optional[float] = None,
        sl_margin_short: Optional[float] = None
    ):
        """
        Initializes OrderGenerator with unified or asymmetric margins.
        """
        self.tp_margin = tp_margin
        self.sl_margin = sl_margin
        self.tp_margin_long = tp_margin_long
        self.tp_margin_short = tp_margin_short
        self.sl_margin_long = sl_margin_long
        self.sl_margin_short = sl_margin_short

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

        # Resolve margin base settings (unified or asymmetric)
        if action == "BUY":
            tp_margin = self.tp_margin_long if self.tp_margin_long is not None else self.tp_margin
            sl_margin = self.sl_margin_long if self.sl_margin_long is not None else self.sl_margin
        else:
            tp_margin = self.tp_margin_short if self.tp_margin_short is not None else self.tp_margin
            sl_margin = self.sl_margin_short if self.sl_margin_short is not None else self.sl_margin

        # Dynamic Volatility Scaling
        if volatility is not None and volatility > 0.0:
            # Normalize volatility against a 1.5 bps baseline
            scale_factor = volatility / 0.00015
            tp_margin = max(0.0002, tp_margin * scale_factor)  # Floor TP at 2 bps
            sl_margin = max(0.0001, sl_margin * scale_factor)  # Floor SL at 1 bps

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
