import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_alpha_macro")))
from inverse_bs import InverseBlackScholes

def test_norm_cdf_precision():
    """Verify the Abramowitz & Stegun CDF approximation matches numpy-based expected values."""
    # Test standard values
    np.testing.assert_allclose(InverseBlackScholes.norm_cdf(0.0), 0.5, atol=1e-7)
    np.testing.assert_allclose(InverseBlackScholes.norm_cdf(1.95996398454), 0.975, atol=1e-5)
    np.testing.assert_allclose(InverseBlackScholes.norm_cdf(-1.95996398454), 0.025, atol=1e-5)
    np.testing.assert_allclose(InverseBlackScholes.norm_cdf(3.0), 0.99865, atol=1e-4)
    np.testing.assert_allclose(InverseBlackScholes.norm_cdf(-3.0), 0.00135, atol=1e-4)

def test_inverse_bs_pricing_symmetry():
    """Verify call and put pricing symmetry and standard vs coin pricing conversion."""
    S = 60000.0
    K = 60000.0
    sigma = 0.50
    t = 30.0 / 365.0
    r = 0.05
    q = 0.02
    
    price_usd_call = InverseBlackScholes.price_option_usd(S, K, sigma, t, r, q, "CALL")
    price_coin_call = InverseBlackScholes.price_option_coin(S, K, sigma, t, r, q, "CALL")
    
    # Premium in coin should be Premium in USD divided by Spot
    np.testing.assert_allclose(price_coin_call, price_usd_call / S, rtol=1e-9)
    
    price_usd_put = InverseBlackScholes.price_option_usd(S, K, sigma, t, r, q, "PUT")
    price_coin_put = InverseBlackScholes.price_option_coin(S, K, sigma, t, r, q, "PUT")
    np.testing.assert_allclose(price_coin_put, price_usd_put / S, rtol=1e-9)
    
    # Put-Call Parity in USD: C - P = S*e^(-q*t) - K*e^(-r*t)
    parity_lhs = price_usd_call - price_usd_put
    parity_rhs = S * np.exp(-q * t) - K * np.exp(-r * t)
    np.testing.assert_allclose(parity_lhs, parity_rhs, rtol=1e-7)

def test_coin_margined_greeks():
    """Verify coin-margined premium-adjusted delta and gamma formulas match analytical expectations."""
    S = np.array([58000.0, 60000.0, 62000.0])
    K = 60000.0
    sigma = 0.40
    t = 7.0 / 365.0
    r = 0.05
    q = 0.00  # 0% coin rate for simplicity
    
    greeks_call = InverseBlackScholes.calculate_coin_margined_greeks(S, K, sigma, t, r, q, "CALL")
    greeks_put = InverseBlackScholes.calculate_coin_margined_greeks(S, K, sigma, t, r, q, "PUT")
    
    # Verify vector sizes match input Spot size
    assert len(greeks_call["delta_coin"]) == 3
    assert len(greeks_call["gamma_coin"]) == 3
    assert len(greeks_put["delta_coin"]) == 3
    assert len(greeks_put["gamma_coin"]) == 3
    
    # Verify premium adjusted call delta: Delta_coin = (K/S) * e^(-r*t) * N(d2)
    d1, d2 = InverseBlackScholes.calculate_d1_d2(S, K, sigma, t, r, q)
    expected_delta_call = (K / S) * np.exp(-r * t) * InverseBlackScholes.norm_cdf(d2)
    np.testing.assert_allclose(greeks_call["delta_coin"], expected_delta_call, rtol=1e-9)
    
    # Verify premium adjusted put delta: Delta_coin = -(K/S) * e^(-r*t) * N(-d2)
    expected_delta_put = -(K / S) * np.exp(-r * t) * InverseBlackScholes.norm_cdf(-d2)
    np.testing.assert_allclose(greeks_put["delta_coin"], expected_delta_put, rtol=1e-9)
    
    # Verify premium-adjusted Gamma relation: Gamma_coin = Gamma_USD - Delta_BTC
    # where Delta_BTC = (Delta_USD - Price_coin) / S
    std_greeks_call = InverseBlackScholes.calculate_standard_greeks(S, K, sigma, t, r, q, "CALL")
    delta_btc = (std_greeks_call["delta"] - greeks_call["price_coin"]) / S
    expected_gamma_call = std_greeks_call["gamma"] - delta_btc
    np.testing.assert_allclose(greeks_call["gamma_coin"], expected_gamma_call, rtol=1e-7)
