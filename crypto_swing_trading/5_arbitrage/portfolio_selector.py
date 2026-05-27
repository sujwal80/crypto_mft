import numpy as np
import logging
from collections import deque
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("HighYieldEngine")

class PortfolioStatArbSelector:
    """
    Vectorized Hurst-Filtered Cointegration Portfolio Selection Engine.
    Ingests multi-coin price matrices, fits dynamic OLS cointegration,
    calculates rescaled range Hurst Exponents, and ranks pairs for capital allocation.
    """
    def __init__(self, candidate_pairs: List[Tuple[str, str]], window: int = 120, hurst_threshold: float = 0.48):
        self.candidate_pairs = candidate_pairs
        self.window = window
        self.hurst_threshold = hurst_threshold
        self.price_histories = {}
        
        # Extract unique assets in universe
        self.unique_assets = set()
        for p1, p2 in candidate_pairs:
            self.unique_assets.add(p1)
            self.unique_assets.add(p2)
            
        # Initialize log price histories
        for asset in self.unique_assets:
            self.price_histories[asset] = deque(maxlen=window)

    def ingest_prices(self, price_tick: Dict[str, float]):
        """
        Ingests price updates for the active coin universe.
        Expects prices in natural scale (e.g., {'BTC': 65000.0, 'ETH': 3400.0})
        """
        for asset, price in price_tick.items():
            if asset in self.price_histories:
                self.price_histories[asset].append(np.log(price))

    def rank_and_select_pairs(self, max_active_pairs: int = 5) -> List[Dict]:
        """
        Vectorized computation of rolling spreads and Hurst exponents across the candidate universe.
        Ranks and returns the top 'max_active_pairs' mean-reverting pairs.
        """
        selected_pairs = []
        
        for p1, p2 in self.candidate_pairs:
            if len(self.price_histories[p1]) < self.window or len(self.price_histories[p2]) < self.window:
                # Ingestion warming up
                continue
                
            y = np.array(self.price_histories[p1])
            x = np.array(self.price_histories[p2])
            
            # 1. OLS Linear regression: y = beta * x + alpha
            beta, alpha = np.polyfit(x, y, 1)
            spread = y - (beta * x + alpha)
            
            # 2. Calculate Hurst Exponent via Rescaled Range (R/S)
            H = self._calculate_hurst(spread)
            
            if H is not None and H < self.hurst_threshold:  # Strict stationarity regime filter
                selected_pairs.append({
                    "pair": (p1, p2),
                    "beta": float(beta),
                    "alpha": float(alpha),
                    "hurst": H,
                    "std_dev": float(np.std(spread))
                })
                
        # Rank pairs by lowest Hurst Exponent (strongest mean reversion)
        selected_pairs = sorted(selected_pairs, key=lambda k: k["hurst"])
        return selected_pairs[:max_active_pairs]

    def _calculate_hurst(self, spread: np.ndarray) -> Optional[float]:
        """
        Dynamic-lag Rescaled Range (R/S) analysis.
        """
        n = len(spread)
        # Ensure lags fit perfectly inside lookback window size
        lags = [lag for lag in [10, 20, 30, 45, 60, 90, 120, 180, 240] if lag <= self.window // 2]
        if len(lags) < 2:
            return None
            
        rs_vals = []
        
        for lag in lags:
            num_periods = n // lag
            periods = spread[:num_periods * lag].reshape((num_periods, lag))
            
            means = np.mean(periods, axis=1, keepdims=True)
            stds = np.std(periods, axis=1) + 1e-8
            demeaned = periods - means
            cum_deviations = np.cumsum(demeaned, axis=1)
            
            ranges = np.max(cum_deviations, axis=1) - np.min(cum_deviations, axis=1)
            rs = np.mean(ranges / stds)
            rs_vals.append(rs)
            
        # Fit log(R/S) vs. log(lag) to extract Hurst exponent (slope)
        H, _ = np.polyfit(np.log(lags), np.log(rs_vals), 1)
        return float(H)
