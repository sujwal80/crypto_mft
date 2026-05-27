"""Calculates aggregate options Gamma Exposure (GEX) per strike and maps critical walls."""

import numpy as np
from indian_intraday_system.config import LOT_SIZE_NIFTY, RISK_FREE_RATE
from indian_intraday_system.layer_2_macro.vanilla_bs import black_scholes_greeks


def calculate_gex(spot, strikes, expiries_days, ivs, open_interests, option_types, lot_size=LOT_SIZE_NIFTY):
    """Calculates options Rupee GEX per contract for a 1% underlying spot move."""
    strikes = np.array(strikes, dtype=float)
    expiries_years = np.array(expiries_days, dtype=float) / 365.0
    ivs = np.array(ivs, dtype=float)
    open_interests = np.array(open_interests, dtype=float)
    option_types = np.array(option_types, dtype=str)

    # BS Gamma
    greeks = black_scholes_greeks(
        S=spot,
        K=strikes,
        T=expiries_years,
        r=RISK_FREE_RATE,
        sigma=ivs,
        option_type=option_types,
    )
    gammas = greeks["gamma"]

    # Call +1, Put -1 (Retail writing puts places dealers short Gamma)
    is_call = (option_types == "C") | (option_types == "call") | (option_types == "CALL")
    directions = np.where(is_call, 1.0, -1.0)

    # Rupee GEX for 1% spot move: OI * Gamma * Spot^2 * LotSize * 0.01
    gex = open_interests * gammas * (spot**2) * lot_size * 0.01 * directions
    return gex


def map_gex_levels(spot, strikes, expiries_days, ivs, open_interests, option_types, lot_size=LOT_SIZE_NIFTY):
    """Aggregates GEX and locates Call Wall, Put Wall, and Zero-Gamma flip level."""
    strikes = np.array(strikes, dtype=float)
    gex = calculate_gex(spot, strikes, expiries_days, ivs, open_interests, option_types, lot_size)

    unique_strikes = np.unique(strikes)
    net_gex_per_strike = []
    call_gex_per_strike = []
    put_gex_per_strike = []

    is_call = (np.array(option_types, dtype=str) == "C") | (
        np.array(option_types, dtype=str) == "call"
    ) | (np.array(option_types, dtype=str) == "CALL")

    for k in unique_strikes:
        mask = strikes == k
        net_gex_per_strike.append(np.sum(gex[mask]))
        call_gex_per_strike.append(np.sum(gex[mask & is_call]))
        put_gex_per_strike.append(np.sum(gex[mask & ~is_call]))

    unique_strikes = np.array(unique_strikes)
    net_gex_per_strike = np.array(net_gex_per_strike)
    call_gex_per_strike = np.array(call_gex_per_strike)
    put_gex_per_strike = np.array(put_gex_per_strike)

    # Put Wall (peak negative Put gamma support)
    put_wall = unique_strikes[np.argmin(put_gex_per_strike)]

    # Call Wall (peak Call GEX resistance)
    call_wall = unique_strikes[np.argmax(call_gex_per_strike)]

    # Zero-Gamma Flip Strike
    zero_gamma = None
    sorted_idx = np.argsort(unique_strikes)
    sorted_strikes = unique_strikes[sorted_idx]
    sorted_gex = net_gex_per_strike[sorted_idx]

    for i in range(len(sorted_gex) - 1):
        if sorted_gex[i] < 0 <= sorted_gex[i + 1]:
            k1, k2 = sorted_strikes[i], sorted_strikes[i + 1]
            g1, g2 = sorted_gex[i], sorted_gex[i + 1]
            zero_gamma = k1 - g1 * (k2 - k1) / (g2 - g1)
            break

    if zero_gamma is None:
        zero_gamma = sorted_strikes[np.argmin(np.abs(sorted_gex))]

    return {
        "spot": spot,
        "zero_gamma": float(zero_gamma),
        "call_wall": float(call_wall),
        "put_wall": float(put_wall),
        "strikes": unique_strikes.tolist(),
        "net_gex": net_gex_per_strike.tolist(),
    }
