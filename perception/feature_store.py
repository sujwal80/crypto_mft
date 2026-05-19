import collections
import numpy as np
from typing import Dict, Optional
from core.schemas import InternalTick

class FeatureStore:
    """Maintains Limit Order Book (LOB) parameters and extracts professional microstructural features."""
    def __init__(self, window_size: int = 1000, lookback: int = 50):
        self.window_size = window_size
        self.lookback = lookback
        
        # Rolling queues
        self.mid_history: Dict[str, collections.deque] = {}
        self.spread_history: Dict[str, collections.deque] = {}
        self.imbalance_history: Dict[str, collections.deque] = {}
        
        # Rolling Welford sliding window states
        self.mid_mean: Dict[str, float] = {}
        self.mid_S: Dict[str, float] = {}
        
        self.spread_mean: Dict[str, float] = {}
        self.spread_S: Dict[str, float] = {}
        
        self.imbalance_mean: Dict[str, float] = {}

    def process_tick(self, tick: InternalTick) -> Optional[np.ndarray]:
        symbol = tick.symbol

        # Initialize collections dynamically per symbol
        if symbol not in self.mid_history:
            self.mid_history[symbol] = collections.deque(maxlen=self.window_size)
            self.spread_history[symbol] = collections.deque(maxlen=self.window_size)
            self.imbalance_history[symbol] = collections.deque(maxlen=self.window_size)
            
            self.mid_mean[symbol] = 0.0
            self.mid_S[symbol] = 0.0
            self.spread_mean[symbol] = 0.0
            self.spread_S[symbol] = 0.0
            self.imbalance_mean[symbol] = 0.0

        mid_price = (tick.bid + tick.ask) / 2.0
        spread = tick.ask - tick.bid
        imbalance = (tick.bid_size - tick.ask_size) / (tick.bid_size + tick.ask_size + 1e-8)

        # Micro-Price: Volumetric Weighted Mid-Price
        micro_price = (tick.bid * tick.ask_size + tick.ask * tick.bid_size) / (tick.bid_size + tick.ask_size + 1e-8)
        micro_price_drift = micro_price - mid_price

        # Append to rolling histories
        self.mid_history[symbol].append(mid_price)
        self.spread_history[symbol].append(spread)
        self.imbalance_history[symbol].append(imbalance)

        history_len = len(self.mid_history[symbol])

        # Ensure we have enough history to compute rolling windows
        if history_len < self.lookback:
            return None

        # 1. Initial Baseline Calculation Phase
        if history_len == self.lookback:
            mids = np.array(list(self.mid_history[symbol])[-self.lookback:])
            self.mid_mean[symbol] = np.mean(mids)
            self.mid_S[symbol] = np.sum((mids - self.mid_mean[symbol])**2)
            
            spreads = np.array(list(self.spread_history[symbol])[-self.lookback:])
            self.spread_mean[symbol] = np.mean(spreads)
            self.spread_S[symbol] = np.sum((spreads - self.spread_mean[symbol])**2)
            
            imbalances = np.array(list(self.imbalance_history[symbol])[-self.lookback:])
            self.imbalance_mean[symbol] = np.mean(imbalances)
            
            rolling_mean = self.mid_mean[symbol]
            rolling_vol = np.sqrt(self.mid_S[symbol] / self.lookback)
            rolling_spread_mean = self.spread_mean[symbol]
            rolling_spread_std = np.sqrt(self.spread_S[symbol] / self.lookback)
            rolling_imbalance = self.imbalance_mean[symbol]

        # 2. Sliding O(1) Recurrence Update Phase
        else:
            # Retrieve old popped element and new incoming element
            x_mid_old = self.mid_history[symbol][-self.lookback - 1]
            x_mid_new = mid_price
            
            x_spread_old = self.spread_history[symbol][-self.lookback - 1]
            x_spread_new = spread
            
            x_imb_old = self.imbalance_history[symbol][-self.lookback - 1]
            x_imb_new = imbalance

            # Recurrence mean & sum-of-squares updates
            mu_mid_old = self.mid_mean[symbol]
            mu_mid_new = mu_mid_old + (x_mid_new - x_mid_old) / self.lookback
            self.mid_mean[symbol] = mu_mid_new
            self.mid_S[symbol] = self.mid_S[symbol] + (x_mid_new - x_mid_old) * (x_mid_new + x_mid_old - mu_mid_old - mu_mid_new)
            
            mu_spread_old = self.spread_mean[symbol]
            mu_spread_new = mu_spread_old + (x_spread_new - x_spread_old) / self.lookback
            self.spread_mean[symbol] = mu_spread_new
            self.spread_S[symbol] = self.spread_S[symbol] + (x_spread_new - x_spread_old) * (x_spread_new + x_spread_old - mu_spread_old - mu_spread_new)
            
            self.imbalance_mean[symbol] = self.imbalance_mean[symbol] + (x_imb_new - x_imb_old) / self.lookback

            rolling_mean = mu_mid_new
            rolling_vol = np.sqrt(max(0.0, self.mid_S[symbol]) / self.lookback)
            rolling_spread_mean = mu_spread_new
            rolling_spread_std = np.sqrt(max(0.0, self.spread_S[symbol]) / self.lookback)
            rolling_imbalance = self.imbalance_mean[symbol]

        # Z-Score of current price relative to rolling window
        z_score = (mid_price - rolling_mean) / (rolling_vol + 1e-8)
        spread_z_score = (spread - rolling_spread_mean) / (rolling_spread_std + 1e-8)

        # Unified Feature Vector (Output matches our ML input requirements)
        return np.array([
            z_score,              # Mean Reversion indicator
            spread_z_score,       # Liquidity / Spread indicator
            rolling_imbalance,    # Volume order flow trend
            micro_price_drift,    # Micro-price prediction signal
            rolling_vol,          # Local volatility / Risk scaling
            mid_price             # Current baseline price
        ])
