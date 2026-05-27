import logging
import time

logger = logging.getLogger(__name__)

class BasisTracker:
    """
    Continuously tracks the live premium or discount of the high-speed execution venue 
    (Binance Perp) versus the options clearing venue index (Deribit Index).
    Dynamically shifts physical execution targets to ensure option strike walls are matched perfectly.
    """
    def __init__(self):
        self.binance_perp_price = None
        self.deribit_index_price = None
        self.last_perp_update = 0.0
        self.last_index_update = 0.0
        self.basis = 0.0 # Binance_Perp - Deribit_Index

    def update_perp_price(self, price: float):
        self.binance_perp_price = float(price)
        self.last_perp_update = time.time()
        self._recalculate_basis()

    def update_index_price(self, price: float):
        self.deribit_index_price = float(price)
        self.last_index_update = time.time()
        self._recalculate_basis()

    def _recalculate_basis(self):
        now = time.time()
        # Enforce a strict 15-second freshness on both feeds to prevent stale/frozen prices from skewing the basis
        if (self.binance_perp_price is not None and 
            self.deribit_index_price is not None and 
            (now - self.last_perp_update) < 15.0 and 
            (now - self.last_index_update) < 15.0):
            self.basis = self.binance_perp_price - self.deribit_index_price
        else:
            self.basis = 0.0

    def adjust_strike_to_execution_target(self, deribit_strike: float) -> float:
        """
        Dynamically shifts a Deribit-denominated options strike to its exact 
        equivalent execution target on the Binance futures order book.
        
        Example:
            Put Wall at $70,000.
            Binance Perp is at $70,040 while Deribit Index is at $70,000 (+$40 basis).
            Execution buy target on Binance is shifted to $70,040.
        """
        if self.deribit_index_price is None or self.binance_perp_price is None:
            # Fallback if index feeds are not warmed up yet
            return deribit_strike
            
        adjusted_target = deribit_strike + self.basis
        logger.debug(f"Basis Adjustment: Strike {deribit_strike} -> Binance execution target {adjusted_target:.2f} (basis: {self.basis:+.2f})")
        return adjusted_target
