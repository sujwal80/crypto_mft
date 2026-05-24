import logging

logger = logging.getLogger(__name__)

class RiskGate:
    """
    Quant Execution Risk Gate.
    Enforces mathematical fee-to-profit constraints.
    Blocks entries unless the expected structural bounce/momentum move exceeds
    the round-trip exchange fees, estimated slippage, and required profit margin.
    """
    def __init__(self, 
                 maker_fee_pct: float = 0.0002,   # 0.02% maker fee
                 taker_fee_pct: float = 0.0005,   # 0.05% taker fee
                 expected_slippage_pct: float = 0.0003, # 0.03% expected slippage
                 minimum_net_profit_pct: float = 0.0010): # 0.10% minimum net alpha margin
                 
        self.maker_fee = maker_fee_pct
        self.taker_fee = taker_fee_pct
        self.slippage = expected_slippage_pct
        self.min_net_profit = minimum_net_profit_pct
        
        # Calculates total structural friction:
        # Entry Maker + Exit Taker (conservative) + Slippage + Target Profit
        self.required_edge = self.maker_fee + self.taker_fee + self.slippage + self.min_net_profit

    def evaluate_entry(self, current_price: float, gex_strike: float, expected_move_pct: float) -> bool:
        """
        Evaluates if the trade setup has mathematical positive expectancy.
        
        Args:
            current_price: Current spot price
            gex_strike: Price level of the key options wall
            expected_move_pct: Estimated size of the structural bounce or breakout move
            
        Returns:
            bool: True if the trade satisfies the strict 0.2% fee-to-profit gate constraint.
        """
        distance_to_wall = abs(current_price - gex_strike) / current_price
        
        # The absolute trade target must justify the friction.
        # Expected profit (expected_move_pct) must overpower total transaction costs.
        if expected_move_pct < self.required_edge:
            logger.warning(
                f"RiskGate Lock: Expected move {expected_move_pct * 100:.3f}% "
                f"fails to clear required edge threshold {self.required_edge * 100:.3f}%"
            )
            return False
            
        logger.debug(
            f"RiskGate Open: Expected move {expected_move_pct * 100:.3f}% "
            f"clears hurdle cost {self.required_edge * 100:.3f}%"
        )
        return True
