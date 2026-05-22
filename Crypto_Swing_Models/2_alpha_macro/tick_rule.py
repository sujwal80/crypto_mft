import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class OptionsTickRule:
    """
    Microstructure-based Taker Trade Classifier & Dealer Net Position Estimator.
    Uses the classical Tick Rule (with Zero-Uptick/Zero-Downtick classification) 
    to determine if option trades are buyer-initiated (taker buy) or seller-initiated (taker sell).
    Translates these into estimated Net Dealer Positions (Dealer Net Long/Short) per strike.
    """
    def __init__(self, memory_decay_halflife: int = 100000):
        """
        Args:
            memory_decay_halflife: Halflife in number of trades to decay historical positions 
                                   so older trades don't permanently dominate the profile.
        """
        self.memory_decay = memory_decay_halflife
        # Estimated dealer net position per strike: positive = net long, negative = net short
        self.dealer_positions = defaultdict(float)
        # Last trade price per strike to calculate tick direction
        self.last_prices = defaultdict(float)
        # Last tick direction per strike (1 for uptick, -1 for downtick)
        self.last_directions = defaultdict(int)
        
    def classify_trade(self, strike: float, price: float, amount: float, direction_hint: str = None) -> int:
        """
        Classifies a single option trade and updates the estimated dealer position.
        
        Args:
            strike: Option strike price
            price: Execution price of the option (in BTC or USD)
            amount: Trade contract size (amount of options traded)
            direction_hint: Optional explicit buy/sell direction (e.g., from Exchange API)
                            If provided, 'buy' means retail bought (dealer short), 
                            and 'sell' means retail sold (dealer long).
        
        Returns:
            int: Trade initiator direction: +1 for Retail Taker Buy, -1 for Retail Taker Sell, 0 if unclassifiable.
        """
        trade_direction = 0
        
        if direction_hint is not None:
            # If exchange provides the initiator direction, use it directly
            hint = direction_hint.lower()
            if hint == "buy":
                trade_direction = 1
            elif hint == "sell":
                trade_direction = -1
        else:
            # Apply classical Tick Rule
            last_price = self.last_prices[strike]
            if last_price == 0.0:
                # First trade at this strike, assume uptick/neutral
                trade_direction = 0
            else:
                if price > last_price:
                    trade_direction = 1  # Uptick (Retail Taker Buy)
                elif price < last_price:
                    trade_direction = -1 # Downtick (Retail Taker Sell)
                else:
                    # Unchanged price, use zero-uptick / zero-downtick rule
                    trade_direction = self.last_directions[strike]
                    
        # Record state for future tick direction calculations
        self.last_prices[strike] = price
        if trade_direction != 0:
            self.last_directions[strike] = trade_direction
            
        # Update estimated dealer position
        # Retail Taker Buy (+1) -> Dealer Sell (-amount) -> Dealer Net Short
        # Retail Taker Sell (-1) -> Dealer Buy (+amount) -> Dealer Net Long
        if trade_direction != 0:
            dealer_flow = -float(trade_direction) * float(amount)
            self.dealer_positions[strike] += dealer_flow
            
        return trade_direction

    def process_trades_batch(self, trades: list):
        """
        Processes a batch of option trades.
        
        Each trade dict must contain:
            - 'strike': float
            - 'price': float
            - 'amount': float
            - 'direction': str (optional, e.g. "buy" or "sell")
        """
        for t in trades:
            strike = float(t["strike"])
            price = float(t["price"])
            amount = float(t["amount"])
            direction = t.get("direction")
            self.classify_trade(strike, price, amount, direction)
            
    def get_dealer_multipliers(self, strikes: np.ndarray) -> np.ndarray:
        """
        Returns a NumPy array of multipliers (+1 for Net Long, -1 for Net Short)
        for the given array of strikes to plug directly into the GEX vector engine.
        """
        multipliers = []
        for s in strikes:
            pos = self.dealer_positions[s]
            # If dealer position is negative, dealer is Net Short (multiplier -1.0)
            # If dealer position is positive or zero, dealer is Net Long (multiplier +1.0)
            multipliers.append(-1.0 if pos < 0 else 1.0)
        return np.array(multipliers, dtype=np.float64)

    def decay_positions(self, decay_factor: float = 0.999):
        """
        Exponentially decays all estimated dealer positions to prevent stale history dominance.
        """
        for strike in list(self.dealer_positions.keys()):
            self.dealer_positions[strike] *= decay_factor
            if abs(self.dealer_positions[strike]) < 1e-4:
                # Cleanup negligible positions to save memory
                self.dealer_positions.pop(strike, None)
                self.last_prices.pop(strike, None)
                self.last_directions.pop(strike, None)
