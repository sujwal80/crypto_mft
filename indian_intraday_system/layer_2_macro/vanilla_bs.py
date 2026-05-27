"""Vectorized Black-Scholes Greeks calculator and high-precision IV solver."""

import numpy as np
from scipy.stats import norm
from indian_intraday_system.config import MIN_VOLATILITY


def black_scholes_greeks(S, K, T, r, sigma, option_type="C"):
    """Calculates vectorized Option Price, Delta, Gamma, Vega, and Theta."""
    S = np.maximum(np.array(S, dtype=float), 1e-8)
    K = np.maximum(np.array(K, dtype=float), 1e-8)
    T = np.maximum(np.array(T, dtype=float), 1e-8)
    r = np.array(r, dtype=float)
    sigma = np.maximum(np.array(sigma, dtype=float), MIN_VOLATILITY)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if isinstance(option_type, str):
        is_call = option_type.upper() in ("C", "CALL")
    else:
        opt_arr = np.array(option_type, dtype=str)
        is_call = (opt_arr == "C") | (opt_arr == "call") | (opt_arr == "CALL")

    # Option Price
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    price = np.where(is_call, call_price, put_price)

    # Delta
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    delta = np.where(is_call, call_delta, put_delta)

    # Gamma (Same for Call/Put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (Same for Call/Put)
    vega = S * norm.pdf(d1) * np.sqrt(T)

    # Theta
    theta_call = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
        -r * T
    ) * norm.cdf(d2)
    theta_put = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(
        -r * T
    ) * norm.cdf(-d2)
    theta = np.where(is_call, theta_call, theta_put)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def implied_volatility(
    price, S, K, T, r, option_type="C", precision=1e-5, max_iter=100
):
    """Newton-Raphson implied volatility solver with vectorized bisection fallback."""
    price = np.array(price, dtype=float)
    S = np.array(S, dtype=float)
    K = np.array(K, dtype=float)
    T = np.array(T, dtype=float)
    r = np.array(r, dtype=float)

    sigma = np.full_like(price, 0.20)  # Initial 20% guess

    for _ in range(max_iter):
        greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
        diff = greeks["price"] - price
        vega = np.maximum(greeks["vega"], 1e-4)

        converged = np.abs(diff) < precision
        if np.all(converged):
            break

        step = diff / vega
        step = np.clip(step, -0.10, 0.10)
        sigma = np.clip(sigma - step, MIN_VOLATILITY, 5.00)

    # Vectorized Bisection fallback for elements that failed to converge
    failed = np.abs(black_scholes_greeks(S, K, T, r, sigma, option_type)["price"] - price) >= precision
    if np.any(failed):
        low = np.full_like(price, MIN_VOLATILITY)
        high = np.full_like(price, 5.00)
        for _ in range(100):
            mid = (low + high) / 2.0
            mid_price = black_scholes_greeks(S, K, T, r, mid, option_type)["price"]
            diff = mid_price - price

            converged = np.abs(diff) < precision
            if np.all(converged[failed]):
                sigma = np.where(failed, mid, sigma)
                break

            low = np.where(failed & (diff < 0), mid, low)
            high = np.where(failed & (diff >= 0), mid, high)

        sigma = np.where(failed, (low + high) / 2.0, sigma)

    return np.clip(sigma, MIN_VOLATILITY, 5.00)
