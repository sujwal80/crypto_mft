import numpy as np
import logging
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class GEXAlphaStrategy(BaseAlphaStrategy):
    """
    Live-testable Quantitative Alpha Strategy integrating option Gamma Exposure (GEX),
    Open Interest (OI), and Order Flow Imbalance (OFI) microstructural features.
    
    Dynamically registers option strikes anchored to the asset's initial spot price,
    calculates option Greeks, and scans for reversal/breakout signals at GEX walls.
    """
    def __init__(
        self, 
        expiry_days: float = 7.0, 
        risk_free_rate: float = 0.05,
        gex_threshold: float = 1.2,       # Z-score or distance threshold to wall
        ofi_threshold: float = 0.3,       # OFI trigger threshold
        sensitivity: float = 0.0015
    ):
        self.t = expiry_days / 365.0       # Time to expiry in years
        self.r = risk_free_rate
        self.gex_threshold = gex_threshold
        self.ofi_threshold = ofi_threshold
        self.sensitivity = sensitivity
        
        # Active options chain: Strike -> dict of contract parameters
        self.options_chain = {}
        self.base_price = None

    def _norm_pdf(self, x: float) -> float:
        """Fast standard normal probability density function in pure numpy."""
        return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

    def _calculate_gamma(self, S: float, K: float, sigma: float) -> float:
        """Calculates option Gamma using standard Black-Scholes formulation."""
        if S <= 0 or K <= 0 or sigma <= 0 or self.t <= 0:
            return 0.0
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * self.t) / (sigma * np.sqrt(self.t))
            gamma = self._norm_pdf(d1) / (S * sigma * np.sqrt(self.t))
            return gamma
        except Exception as e:
            logger.error(f"Error calculating Gamma: {e}")
            return 0.0

    def _initialize_options_chain(self, mid_price: float, rolling_vol: float):
        """
        Initializes a live-testable options chain anchored around the first seen spot price.
        Simulates positive GEX call/put walls and negative GEX breakout zones.
        """
        self.base_price = mid_price
        self.options_chain = {}
        
        # Determine strike spacing (approx 0.1% of spot price)
        strike_increment = max(1.0, round(self.base_price * 0.001, 1))
        
        logger.info(f"🎯 Initializing simulated options chain around base price ${self.base_price:.2f} (spacing: ${strike_increment:.2f})")
        
        # Generate strikes from -15 to +15 steps
        for i in range(-15, 16):
            strike = round(self.base_price + i * strike_increment, 1)
            
            # Volatility default to rolling volatility (capped at reasonable limits)
            sigma = max(0.15, min(0.8, rolling_vol / (mid_price + 1e-8)))
            
            # Defaults
            call_oi = 10000.0
            put_oi = 10000.0
            is_dealer_short = False
            
            # 1. Positive GEX Call Wall at +0.4% (Pinning overhead resistance)
            if i == 4:
                call_oi = 150000.0
                put_oi = 5000.0
                logger.info(f"   [Call Wall] Configured at Strike ${strike:.2f} (+0.4%)")
            # 2. Positive GEX Put Wall at -0.4% (Pinning downside support)
            elif i == -4:
                call_oi = 5000.0
                put_oi = 150000.0
                logger.info(f"   [Put Wall] Configured at Strike ${strike:.2f} (-0.4%)")
            # 3. Negative GEX Squeeze Zone at -0.8% (Volatility breakout / short gamma)
            elif i == -8:
                call_oi = 5000.0
                put_oi = 200000.0
                is_dealer_short = True
                logger.info(f"   [Squeeze Zone] Configured at Strike ${strike:.2f} (-0.8%)")
                
            self.options_chain[strike] = {
                "call_oi": call_oi,
                "put_oi": put_oi,
                "sigma": sigma,
                "is_dealer_short": is_dealer_short
            }

    def calculate_gex_profile(self, spot_price: float) -> dict:
        """Calculates net dealer GEX at each active strike level."""
        gex_profile = {}
        for strike, info in self.options_chain.items():
            gamma = self._calculate_gamma(spot_price, strike, info["sigma"])
            
            # Net Gamma Exposure: (Call_OI + Put_OI) * S^2 * Gamma
            # Both call and put contracts contribute positive gamma when the dealer is long.
            # If dealer is net short, flip the sign of their exposure (creates Negative GEX).
            position_multiplier = -1.0 if info["is_dealer_short"] else 1.0
            
            net_gex = position_multiplier * (info["call_oi"] + info["put_oi"]) * gamma * (spot_price ** 2) * 0.01
            gex_profile[strike] = net_gex
        return gex_profile

    def predict(self, features: np.ndarray) -> float:
        """
        Calculates alpha forecast signal.
        
        Args:
            features: [z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price]
        """
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        # 1. Dynamic lazy-load & auto-recenter of options chain on massive price drifts (> 1.5%)
        if not self.options_chain or (self.base_price is not None and abs(mid_price - self.base_price) / self.base_price > 0.015):
            if self.base_price is not None:
                logger.warning(f"🚨 Spot price (${mid_price:.2f}) drifted > 1.5% from options base price (${self.base_price:.2f}). Re-centering strikes...")
            self._initialize_options_chain(mid_price, rolling_vol)

        # 2. Compute GEX profile for all strikes
        gex_profile = self.calculate_gex_profile(mid_price)
        if not gex_profile:
            return 0.0

        #         # 3. Scan all active strikes for proximity and GEX wall triggers
        alpha = 0.0
        for strike, gex in gex_profile.items():
            # Distance in percentage terms
            price_distance_pct = (mid_price - strike) / strike
            
            # Proximity threshold: within 0.3% of the strike price
            if abs(price_distance_pct) <= 0.003:
                # Check if this is a major structural GEX wall (threshold of 100.0 cash GEX)
                if abs(gex) >= 100.0:
                    # CASE A: Positive GEX Wall (Vol pin, expects Mean Reversion)
                    if gex > 0:
                        # Spot approaches Call Wall from below -> expect rejection/short reversal
                        if price_distance_pct < 0 and rolling_imbalance < -self.ofi_threshold:
                            alpha = -self.sensitivity * abs(rolling_imbalance)
                            logger.debug(f"🛡️ [GEX-MR] Spot (${mid_price:.2f}) near Positive GEX Call Wall (${strike:.2f}). OFI Negative ({rolling_imbalance:.2f}) -> Short Signal: {alpha:.4f}")
                            break
                        # Spot approaches Put Wall from above -> expect support/long reversal
                        elif price_distance_pct > 0 and rolling_imbalance > self.ofi_threshold:
                            alpha = self.sensitivity * abs(rolling_imbalance)
                            logger.debug(f"🛡️ [GEX-MR] Spot (${mid_price:.2f}) near Positive GEX Put Wall (${strike:.2f}). OFI Positive ({rolling_imbalance:.2f}) -> Long Signal: {alpha:.4f}")
                            break
                            
                    # CASE B: Negative GEX Wall (Gamma squeeze breakout momentum expected)
                    else:
                        # Spot approaches from below and buyers push hard -> breakout squeeze
                        if price_distance_pct < 0 and rolling_imbalance > self.ofi_threshold:
                            alpha = self.sensitivity * abs(rolling_imbalance) * 1.5
                            logger.debug(f"⚡ [GEX-BO] Spot (${mid_price:.2f}) near Negative GEX Wall (${strike:.2f}). OFI Positive ({rolling_imbalance:.2f}) -> Squeeze Buy: {alpha:.4f}")
                            break
                        # Spot approaches from above and sellers push hard -> momentum cascade
                        elif price_distance_pct > 0 and rolling_imbalance < -self.ofi_threshold:
                            alpha = -self.sensitivity * abs(rolling_imbalance) * 1.5
                            logger.debug(f"⚡ [GEX-BO] Spot (${mid_price:.2f}) near Negative GEX Wall (${strike:.2f}). OFI Negative ({rolling_imbalance:.2f}) -> Breakdown Sell: {alpha:.4f}")
                            break

        # Cap return target to fit optimizer scales
        return float(np.clip(alpha, -0.005, 0.005))
