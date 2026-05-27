"""Aggregates high-frequency trade ticks to compute Cumulative Volume Delta (CVD)."""


class CVDEngine:
    """Processes raw trade ticks using Quote Rule and Tick Rule to isolate taker aggression."""

    def __init__(self):
        self.last_price = None
        self.last_direction = 1  # Buy: 1, Sell: -1
        self.cvd = 0.0
        self.total_volume = 0.0

    def reset(self):
        """Reset daily session metrics."""
        self.last_price = None
        self.last_direction = 1
        self.cvd = 0.0
        self.total_volume = 0.0

    def process_trade(self, price: float, volume: float, bid: float = None, ask: float = None) -> float:
        """Processes tick details and increments CVD."""
        self.total_volume += volume
        direction = 0

        # Quote Rule
        if bid is not None and ask is not None and bid < ask:
            mid = (bid + ask) / 2.0
            if price >= ask:
                direction = 1
            elif price <= bid:
                direction = -1
            else:
                direction = 1 if price > mid else (-1 if price < mid else self.last_direction)

        # Tick Rule fallback
        else:
            if self.last_price is not None:
                if price > self.last_price:
                    direction = 1
                elif price < self.last_price:
                    direction = -1
                else:
                    direction = self.last_direction
            else:
                direction = 1

        self.last_price = price
        self.last_direction = direction
        self.cvd += direction * volume
        return self.cvd

    def get_cvd(self) -> float:
        return self.cvd

    def get_taker_ratio(self) -> float:
        """Computes taker imbalance ratio between -1.0 (heavy selling) and +1.0 (heavy buying)."""
        if self.total_volume == 0:
            return 0.0
        return self.cvd / self.total_volume
