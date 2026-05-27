"""Replay Engine that simulates intraday trading paths from EOD daily Bhavcopy datasets."""

import random
from typing import Dict, Generator
import numpy as np
import pandas as pd
from indian_intraday_system.backtest.bhavcopy_loader import BhavcopyLoader
from indian_intraday_system.layer_2_macro.gex_mapper import map_gex_levels
from indian_intraday_system.layer_2_macro.vanilla_bs import implied_volatility


class ReplayEngine:
    """FAKE API: Generator yielding 1-minute simulated tick updates using daily EOD data."""

    def __init__(self, loader: BhavcopyLoader = None):
        self.loader = loader or BhavcopyLoader()

    def generate_intraday_replay(
        self, date_str: str, prev_close: float = 22000.0, force_eod_spot: float = None
    ) -> Generator[Dict, None, None]:
        """Loads Bhavcopy for date_str, generates a synthetic 375-minute price path, and yields ticks."""
        df = self.loader.get_bhavcopy(date_str)

        if force_eod_spot is not None:
            eod_spot = force_eod_spot
        else:
            fut_row = df[df["INSTRUMENT"] == "FUTIDX"]
            if not fut_row.empty:
                eod_spot = float(fut_row.iloc[0]["CLOSE"])
            else:
                eod_spot = 22020.0

        opt_df = df[df["INSTRUMENT"] == "OPTIDX"]

        strikes = opt_df["STRIKE_PR"].values
        option_types = np.where(opt_df["OPTION_TYP"] == "CE", "C", "P")
        open_interests = opt_df["OPEN_INT"].values
        closes = opt_df["CLOSE"].values

        expiries_days = np.full_like(strikes, 6.0)

        print(f"[ReplayEngine] Solving EOD Implied Volatilities for {len(strikes)} contracts...")
        ivs = implied_volatility(
            price=closes,
            S=eod_spot,
            K=strikes,
            T=expiries_days / 365.0,
            r=0.07,
            option_type=option_types,
        )
        print("[ReplayEngine] IV solver completed successfully.")

        minutes = 375
        t = np.linspace(0, 1, minutes)

        wiener = np.zeros(minutes)
        for i in range(1, minutes):
            wiener[i] = wiener[i - 1] + random.normalvariate(0, 1.0)

        bridge = wiener - t * wiener[-1]
        path = prev_close + t * (eod_spot - prev_close) + bridge * 5.0

        cvd = 0.0
        total_volume = 0.0

        for minute in range(minutes):
            spot = float(path[minute])
            price_change = spot - path[minute - 1] if minute > 0 else 0.0
            volume = float(random.randint(500, 5000))
            total_volume += volume

            if price_change > 0:
                cvd += volume * 0.8
            elif price_change < 0:
                cvd -= volume * 0.8
            else:
                cvd += random.choice([-1, 1]) * volume * 0.2

            yield {
                "timestamp": f"{date_str}T{9+minute//60:02d}:{minute%60:02d}:00",
                "minute_index": minute,
                "spot": spot,
                "future_price": spot + 10.0,
                "volume": volume,
                "cvd": cvd,
                "total_volume": total_volume,
                "strikes": strikes.tolist(),
                "option_types": option_types.tolist(),
                "open_interests": open_interests.tolist(),
                "ivs": ivs.tolist(),
                "expiries_days": expiries_days.tolist(),
            }
