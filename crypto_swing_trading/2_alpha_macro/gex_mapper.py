import numpy as np
import sys
import os
import logging

# Insert directory to import peer modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from inverse_bs import InverseBlackScholes

logger = logging.getLogger(__name__)

class GexMapper:
    """
    Quant Macro Mapping Engine.
    Aggregates option chain structures and dealer positions, calculates the complete
    GEX profile, and maps out Zero Gamma, Top Support Put Walls, and Top Squeeze Call Walls.
    Supports standard Black-Scholes and premium-adjusted coin-margined Greek models.
    """
    def __init__(self, model_type="COIN_MARGINED"):
        """
        Args:
            model_type: "STANDARD" (USD linear Greeks) or "COIN_MARGINED" (premium-adjusted Greeks)
        """
        self.model_type = model_type.upper()
        
    def calculate_gex_profile(self, spot_price: float, strikes: np.ndarray, 
                              call_oi: np.ndarray, put_oi: np.ndarray, 
                              sigmas: np.ndarray, t: float, r: float, q: float,
                              dealer_multipliers: np.ndarray = None) -> dict:
        """
        Calculates net dealer GEX values across all strike levels in a highly vectorized step.
        
        Args:
            spot_price: Current spot price of the asset
            strikes: Array of strike prices
            call_oi: Array of Call Open Interest contract counts
            put_oi: Array of Put Open Interest contract counts
            sigmas: Array of volatilities per strike
            t: Time to expiry (years)
            r: Risk-free rate
            q: Dividend yield (coin interest rate)
            dealer_multipliers: Multipliers representing dealer positioning per strike.
                                If None, defaults to short option premium (-1.0).
        """
        if len(strikes) == 0:
            return {}
            
        # Default dealer position: net short option premium (multipliers = -1.0)
        if dealer_multipliers is None:
            dealer_multipliers = -np.ones_like(strikes)
            
        if self.model_type == "COIN_MARGINED":
            # Vectorized premium-adjusted Greeks
            call_greeks = InverseBlackScholes.calculate_coin_margined_greeks(spot_price, strikes, sigmas, t, r, q, "CALL")
            put_greeks = InverseBlackScholes.calculate_coin_margined_greeks(spot_price, strikes, sigmas, t, r, q, "PUT")
            
            gamma_call = call_greeks["gamma_coin"]
            gamma_put = put_greeks["gamma_coin"]
        else:
            # Vectorized standard USD Greeks
            call_greeks = InverseBlackScholes.calculate_standard_greeks(spot_price, strikes, sigmas, t, r, q, "CALL")
            put_greeks = InverseBlackScholes.calculate_standard_greeks(spot_price, strikes, sigmas, t, r, q, "PUT")
            
            gamma_call = call_greeks["gamma"]
            gamma_put = put_greeks["gamma"]
            
        # GEX calculation:
        # GEX = DealerMultiplier * (Call_OI * Gamma_Call + Put_OI * Gamma_Put) * S^2 * 0.01
        gex_values = dealer_multipliers * (call_oi * gamma_call + put_oi * gamma_put) * (spot_price ** 2) * 0.01
        
        return dict(zip(strikes, gex_values))

    def map_structural_hedging(self, gex_profile: dict) -> dict:
        """
        Analyzes GEX profile to find Zero Gamma, Top Support (Put) Walls, and Top Squeeze (Call) Walls.
        
        Returns dict containing:
            - 'zero_gamma': float
            - 'support_walls': list of (strike, gex_value)
            - 'squeeze_walls': list of (strike, gex_value)
        """
        if not gex_profile:
            return {"zero_gamma": None, "support_walls": [], "squeeze_walls": []}
            
        strikes = np.array(list(gex_profile.keys()))
        gex_values = np.array(list(gex_profile.values()))
        
        # 1. Find Zero Gamma Level (where GEX crosses or is closest to zero)
        # Find index where sign changes or absolute GEX value is minimum
        zero_idx = np.argmin(np.abs(gex_values))
        zero_gamma = strikes[zero_idx]
        
        # Look for sign change crossing
        sign_changes = np.diff(np.sign(gex_values))
        if np.any(sign_changes != 0):
            crossing_idx = np.where(sign_changes != 0)[0]
            # Interpolate or choose the strike with sign change closest to zero
            candidates = strikes[crossing_idx]
            zero_gamma = candidates[np.argmin(np.abs(gex_values[crossing_idx]))]

        # 2. Find Top Support Put Walls (Highest POSITIVE GEX levels)
        positive_indices = np.where(gex_values > 0)[0]
        if len(positive_indices) > 0:
            # Sort by GEX value descending
            sorted_pos = positive_indices[np.argsort(-gex_values[positive_indices])]
            support_walls = [(strikes[idx], gex_values[idx]) for idx in sorted_pos[:3]]
        else:
            support_walls = []
            
        # 3. Find Top Squeeze Call Walls (Highest NEGATIVE GEX levels)
        negative_indices = np.where(gex_values < 0)[0]
        if len(negative_indices) > 0:
            # Sort by absolute GEX value descending (most negative)
            sorted_neg = negative_indices[np.argsort(gex_values[negative_indices])]
            squeeze_walls = [(strikes[idx], gex_values[idx]) for idx in sorted_neg[:3]]
        else:
            squeeze_walls = []
            
        return {
            "zero_gamma": zero_gamma,
            "support_walls": support_walls,
            "squeeze_walls": squeeze_walls
        }
