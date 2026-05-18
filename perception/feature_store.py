import collections
import numpy as np
from typing import Dict, Optional
from core.schemas import InternalTick

class FeatureStore:
    """Maintains Limit Order Book (LOB) parameters and extracts professional microstructural features."""
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.mid_history: Dict[str, collections.deque] = {}
        self.spread_history: Dict[str, collections.deque] = {}
        self.imbalance_history: Dict[str, collections.deque] = {}

    def process_tick(self, tick: InternalTick) -> Optional[np.ndarray]:
        symbol = tick.symbol

        # Initialize collections dynamically per symbol
        if symbol not in self.mid_history:
            self.mid_history[symbol] = collections.deque(maxlen=self.window_size)
            self.spread_history[symbol] = collections.deque(maxlen=self.window_size)
            self.imbalance_history[symbol] = collections.deque(maxlen=self.window_size)

        mid_price = (tick.bid + tick.ask) / 2.0
        spread = tick.ask - tick.bid

        # Order Book Volume Imbalance
        imbalance = (tick.bid_size - tick.ask_size) / (tick.bid_size + tick.ask_size + 1e-8)

        # Micro-Price: Volumetric Weighted Mid-Price
        # (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        micro_price = (tick.bid * tick.ask_size + tick.ask * tick.bid_size) / (tick.bid_size + tick.ask_size + 1e-8)
        micro_price_drift = micro_price - mid_price

        # Append to rolling histories
        self.mid_history[symbol].append(mid_price)
        self.spread_history[symbol].append(spread)
        self.imbalance_history[symbol].append(imbalance)

        # Ensure we have enough history to compute rolling windows (min 50 ticks)
        if len(self.mid_history[symbol]) < 50:
            return None

        mids = np.array(list(self.mid_history[symbol])[-50:])
        spreads = np.array(list(self.spread_history[symbol])[-50:])
        imbalances = np.array(list(self.imbalance_history[symbol])[-50:])

        # Rolling Volatility
        rolling_vol = np.std(mids)

        # Z-Score of current price relative to rolling 50-tick window
        rolling_mean = np.mean(mids)
        z_score = (mid_price - rolling_mean) / (rolling_vol + 1e-8)

        # Rolling Imbalance Trend (Moving average of OBI)
        rolling_imbalance = np.mean(imbalances)

        # Rolling Spread Mean (to detect expansion/contraction)
        rolling_spread_mean = np.mean(spreads)
        spread_z_score = (spread - rolling_spread_mean) / (np.std(spreads) + 1e-8)

        # Unified Feature Vector (Output matches our ML input requirements)
        return np.array([
            z_score,              # Mean Reversion indicator
            spread_z_score,       # Liquidity / Spread indicator
            rolling_imbalance,    # Volume order flow trend
            micro_price_drift,    # Micro-price prediction signal
            rolling_vol,          # Local volatility / Risk scaling
            mid_price             # Current baseline price
        ])
