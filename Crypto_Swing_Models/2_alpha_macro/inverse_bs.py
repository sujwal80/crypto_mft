import numpy as np
import logging

logger = logging.getLogger(__name__)

class InverseBlackScholes:
    """
    Highly optimized, purely NumPy vectorized Inverse Black-Scholes pricing and Greek engine.
    Specifically designed for coin-margined (inverse) crypto options (e.g. Deribit).
    Calculates standard USD greeks and premium-adjusted coin-margined Greeks natively.
    """
    
    @staticmethod
    def norm_pdf(x):
        """Vectorized standard normal probability density function (PDF)."""
        return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

    @classmethod
    def norm_cdf(cls, x):
        """
        Vectorized standard normal cumulative distribution function (CDF) 
        using Abramowitz & Stegun approximation (max error < 7.5e-8).
        Does not rely on Scipy to maximize speed and eliminate dependency overhead.
        """
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        p = 0.2316419
        a1 = 0.319381530
        a2 = -0.356563782
        a3 = 1.781477937
        a4 = -1.821255978
        a5 = 1.330274429
        
        t = 1.0 / (1.0 + p * abs_x)
        poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
        cdf_abs = 1.0 - cls.norm_pdf(abs_x) * poly
        
        return np.where(x >= 0, cdf_abs, 1.0 - cdf_abs)

    @classmethod
    def calculate_d1_d2(cls, S, K, sigma, t, r, q):
        """
        Calculates Black-Scholes d1 and d2 matrices.
        
        Args:
            S (float or array): Spot price of underlying in USD
            K (float or array): Strike price in USD
            sigma (float or array): Volatility
            t (float or array): Time to expiry in years
            r (float or array): Risk-free rate (USD interest rate)
            q (float or array): Dividend yield (coin interest rate / borrow rate)
        """
        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        
        # Avoid division by zero for expired/near-expired options or zero vol
        t_safe = np.where(t <= 1e-8, 1e-8, t)
        sigma_safe = np.where(sigma <= 1e-8, 1e-8, sigma)
        S_safe = np.where(S <= 0, 1e-8, S)
        K_safe = np.where(K <= 0, 1e-8, K)
        
        sqrt_t = np.sqrt(t_safe)
        d1 = (np.log(S_safe / K_safe) + (r - q + 0.5 * sigma_safe**2) * t_safe) / (sigma_safe * sqrt_t)
        d2 = d1 - sigma_safe * sqrt_t
        return d1, d2

    @classmethod
    def price_option_usd(cls, S, K, sigma, t, r, q, option_type="CALL"):
        """
        Vectorized European option pricing in USD terms.
        """
        d1, d2 = cls.calculate_d1_d2(S, K, sigma, t, r, q)
        
        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        
        if option_type.upper() == "CALL":
            price = S * np.exp(-q * t) * cls.norm_cdf(d1) - K * np.exp(-r * t) * cls.norm_cdf(d2)
        else:
            price = K * np.exp(-r * t) * cls.norm_cdf(-d2) - S * np.exp(-q * t) * cls.norm_cdf(-d1)
            
        return np.where(t <= 0, np.maximum(0.0, S - K) if option_type.upper() == "CALL" else np.maximum(0.0, K - S), price)

    @classmethod
    def price_option_coin(cls, S, K, sigma, t, r, q, option_type="CALL"):
        """
        Vectorized European option pricing in Coin terms (e.g. BTC or ETH).
        Premium_coin = Premium_USD / Spot
        """
        S = np.asarray(S, dtype=np.float64)
        price_usd = cls.price_option_usd(S, K, sigma, t, r, q, option_type)
        S_safe = np.where(S <= 0, 1e-8, S)
        return price_usd / S_safe

    @classmethod
    def calculate_standard_greeks(cls, S, K, sigma, t, r, q, option_type="CALL"):
        """
        Vectorized standard Black-Scholes Greeks in USD terms (linear model).
        """
        d1, d2 = cls.calculate_d1_d2(S, K, sigma, t, r, q)
        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        
        S_safe = np.where(S <= 0, 1e-8, S)
        t_safe = np.where(t <= 1e-8, 1e-8, t)
        sigma_safe = np.where(sigma <= 1e-8, 1e-8, sigma)
        sqrt_t = np.sqrt(t_safe)
        
        pdf_d1 = cls.norm_pdf(d1)
        
        # Delta
        if option_type.upper() == "CALL":
            delta = np.exp(-q * t) * cls.norm_cdf(d1)
        else:
            delta = -np.exp(-q * t) * cls.norm_cdf(-d1)
            
        # Gamma
        gamma = np.exp(-q * t) * pdf_d1 / (S_safe * sigma_safe * sqrt_t)
        
        # Theta
        term1 = -(S_safe * pdf_d1 * sigma_safe * np.exp(-q * t)) / (2.0 * sqrt_t)
        if option_type.upper() == "CALL":
            term2 = q * S_safe * np.exp(-q * t) * cls.norm_cdf(d1) - r * K * np.exp(-r * t) * cls.norm_cdf(d2)
        else:
            term2 = -q * S_safe * np.exp(-q * t) * cls.norm_cdf(-d1) + r * K * np.exp(-r * t) * cls.norm_cdf(-d2)
        theta = term1 + term2
        
        # Vega
        vega = S_safe * np.exp(-q * t) * pdf_d1 * sqrt_t
        
        return {
            "price": cls.price_option_usd(S, K, sigma, t, r, q, option_type),
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega
        }

    @classmethod
    def calculate_coin_margined_greeks(cls, S, K, sigma, t, r, q, option_type="CALL"):
        """
        Vectorized Coin-Margined premium-adjusted Greeks (non-linear model).
        Specifically accounts for coin denomination of payout and premium.
        """
        d1, d2 = cls.calculate_d1_d2(S, K, sigma, t, r, q)
        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        
        S_safe = np.where(S <= 0, 1e-8, S)
        t_safe = np.where(t <= 1e-8, 1e-8, t)
        sigma_safe = np.where(sigma <= 1e-8, 1e-8, sigma)
        sqrt_t = np.sqrt(t_safe)
        
        pdf_d2 = cls.norm_pdf(d2)
        
        # Premium-adjusted Delta in coin:
        # Call Delta = (K/S) * e^(-r*t) * N(d2)
        # Put Delta = -(K/S) * e^(-r*t) * N(-d2)
        if option_type.upper() == "CALL":
            delta_coin = (K / S_safe) * np.exp(-r * t) * cls.norm_cdf(d2)
            gamma_coin = (K / (S_safe**2)) * np.exp(-r * t) * (pdf_d2 / (sigma_safe * sqrt_t) - cls.norm_cdf(d2))
        else:
            delta_coin = -(K / S_safe) * np.exp(-r * t) * cls.norm_cdf(-d2)
            gamma_coin = (K / (S_safe**2)) * np.exp(-r * t) * (pdf_d2 / (sigma_safe * sqrt_t) + cls.norm_cdf(-d2))
            
        return {
            "price_coin": cls.price_option_coin(S, K, sigma, t, r, q, option_type),
            "delta_coin": delta_coin,
            "gamma_coin": gamma_coin
        }
