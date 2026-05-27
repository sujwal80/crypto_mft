"""Tracks premium/discount spread (basis) between Index Futures and Cash Spot Index."""

import collections
import numpy as np


class BasisTracker:
    """Tracks real-time basis spreads and calculates statistical Z-scores.

    Enforces 0DTE decay convergence on Thursdays.
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.spreads = collections.deque(maxlen=window_size)
        self.latest_basis = 0.0

    def add_tick(self, future_price: float, spot_price: float):
        """Calculates and inserts a new basis tick."""
        self.latest_basis = future_price - spot_price
        self.spreads.append(self.latest_basis)

    def get_latest_basis(self) -> float:
        return self.latest_basis

    def get_basis_stats(self) -> dict:
        """Returns rolling mean, standard deviation, and current Z-score of the basis.

        Used to identify speculative price divergence.
        """
        if len(self.spreads) < 5:
            return {"mean": 0.0, "std": 1.0, "z_score": 0.0}

        arr = np.array(self.spreads)
        mean = np.mean(arr)
        std = np.std(arr)
        std = np.maximum(std, 0.5)  # Floor standard dev to prevent divide by zero

        z_score = (self.latest_basis - mean) / std

        return {
            "mean": float(mean),
            "std": float(std),
            "z_score": float(z_score),
        }

    def adjust_threshold_for_expiry(self, base_threshold: float, minutes_to_expiry: float) -> float:
        """Linearly contracts basis anomaly thresholds toward zero on expiry day (0DTE).

        Prevents false speculation blocks during index convergence.
        """
        if minutes_to_expiry <= 0:
            return 0.0

        # If expiry is today and under 375 minutes (normal session length)
        if minutes_to_expiry < 375.0:
            weight = minutes_to_expiry / 375.0
            return base_threshold * weight

        return base_threshold
