import collections
import numpy as np
from typing import Dict, Optional
from core.schemas import InternalTick

class FeatureStore:
    """Maintains an in-memory Limit Order Book (LOB) and computes rolling statistical features."""
    def __init__(self, window_size: int = 1000):
        self.price_history: Dict[str, collections.deque] = {}
        self.window_size = window_size
        
    def process_tick(self, tick: InternalTick) -> Optional[np.ndarray]:
        if tick.symbol not in self.price_history:
            self.price_history[tick.symbol] = collections.deque(maxlen=self.window_size)
            
        mid_price = (tick.bid + tick.ask) / 2.0
        self.price_history[tick.symbol].append(mid_price)
        
        prices = np.array(self.price_history[tick.symbol])
        if len(prices) < 20:
            return None # Not enough data to compute rolling features
            
        rolling_mean = np.mean(prices)
        rolling_std = np.std(prices)
        z_score = (mid_price - rolling_mean) / (rolling_std + 1e-8)
        spread = tick.ask - tick.bid
        imbalance = (tick.bid_size - tick.ask_size) / (tick.bid_size + tick.ask_size + 1e-8)
        
        return np.array([z_score, spread, imbalance, mid_price])
