import numpy as np
import logging
import time
from intelligence.base_strategy import BaseAlphaStrategy

logger = logging.getLogger(__name__)

class GexOiAlphaEngine(BaseAlphaStrategy):
    """
    Institutional-grade, fully vectorized quantitative Option GEX Engine.
    Utilizes NumPy matrix operations to compute Black-Scholes option Gamma and
    net dealer Gamma Exposure (GEX) profiles across the option chain in parallel.
    Features a dual-trigger rate-limiting cache (5,000 ticks or 10 seconds).
    """
    def __init__(
        self, 
        expiry_days: float = 7.0, 
        risk_free_rate: float = 0.05,
        gex_threshold: float = 1.2,
        ofi_threshold: float = 0.3,
        sensitivity: float = 0.0015,
        update_interval: int = 5000,
        time_cache_seconds: float = 10.0
    ):
        self.t = expiry_days / 365.0
        self.r = risk_free_rate
        self.gex_threshold = gex_threshold
        self.ofi_threshold = ofi_threshold
        self.sensitivity = sensitivity
        
        # Cache and update thresholds
        self.update_interval = update_interval
        self.time_cache_seconds = time_cache_seconds
        self.ticks_since_update = 0
        self.last_update_time = None
        
        # Option chain structures (parallel NumPy arrays)
        self.strikes = np.array([], dtype=np.float64)
        self.call_oi = np.array([], dtype=np.float64)
        self.put_oi = np.array([], dtype=np.float64)
        self.sigmas = np.array([], dtype=np.float64)
        self.multipliers = np.array([], dtype=np.float64)
        
        self.options_chain = {}  # Keep dict for backwards compatibility/inspections
        self.base_price = None
        self.cached_gex_profile = None

    def _initialize_options_chain(self, mid_price: float, rolling_vol: float):
        """
        Initializes options chain structures, caching them in parallel NumPy arrays
        for extremely fast, vectorized element-wise pricing operations.
        """
        self.base_price = mid_price
        self.options_chain = {}
        
        # Determine strike spacing (approx 0.1% of spot price)
        strike_increment = max(1.0, round(self.base_price * 0.001, 1))
        
        strikes_list = []
        call_oi_list = []
        put_oi_list = []
        sigmas_list = []
        multipliers_list = []
        
        # Generate strikes from -15 to +15 steps
        for i in range(-15, 16):
            strike = round(self.base_price + i * strike_increment, 1)
            sigma = max(0.15, min(0.8, rolling_vol / (mid_price + 1e-8)))
            
            call_oi = 10000.0
            put_oi = 10000.0
            is_dealer_short = False
            
            # 1. Positive GEX Call Wall at +0.4% (Pinning overhead resistance)
            if i == 4:
                call_oi = 150000.0
                put_oi = 5000.0
            # 2. Positive GEX Put Wall at -0.4% (Pinning downside support)
            elif i == -4:
                call_oi = 5000.0
                put_oi = 150000.0
            # 3. Negative GEX Squeeze Zone at -0.8% (Volatility breakout)
            elif i == -8:
                call_oi = 5000.0
                put_oi = 200000.0
                is_dealer_short = True
                
            multiplier = -1.0 if is_dealer_short else 1.0
            
            strikes_list.append(strike)
            call_oi_list.append(call_oi)
            put_oi_list.append(put_oi)
            sigmas_list.append(sigma)
            multipliers_list.append(multiplier)
            
            # Maintain the standard dictionary API for backwards compatibility
            self.options_chain[strike] = {
                "call_oi": call_oi,
                "put_oi": put_oi,
                "sigma": sigma,
                "is_dealer_short": is_dealer_short
            }
            
        self.strikes = np.array(strikes_list, dtype=np.float64)
        self.call_oi = np.array(call_oi_list, dtype=np.float64)
        self.put_oi = np.array(put_oi_list, dtype=np.float64)
        self.sigmas = np.array(sigmas_list, dtype=np.float64)
        self.multipliers = np.array(multipliers_list, dtype=np.float64)
        
        self.cached_gex_profile = None
        self.ticks_since_update = 0
        self.last_update_time = None

    def calculate_gex_profile(self, spot_price: float) -> dict:
        """
        Calculates the net dealer GEX profile across the entire options chain in parallel.
        Utilizes fully vectorized Black-Scholes options Gamma equations.
        """
        if len(self.strikes) == 0:
            return {}
            
        # Vectorized Black-Scholes d1:
        # d1 = (ln(S/K) + (r + 0.5 * sigma^2) * t) / (sigma * sqrt(t))
        sqrt_t = np.sqrt(self.t)
        d1 = (np.log(spot_price / self.strikes) + (self.r + 0.5 * self.sigmas**2) * self.t) / (self.sigmas * sqrt_t)
        
        # Vectorized standard normal PDF: N'(d1) = exp(-0.5 * d1^2) / sqrt(2 * pi)
        norm_pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
        
        # Vectorized Gamma: Gamma = N'(d1) / (S * sigma * sqrt(t))
        gamma = norm_pdf / (spot_price * self.sigmas * sqrt_t)
        
        # Vectorized GEX: multiplier * (call_oi + put_oi) * Gamma * S^2 * 0.01
        gex_values = self.multipliers * (self.call_oi + self.put_oi) * gamma * (spot_price ** 2) * 0.01
        
        # Return a standard dictionary profile to preserve full compatibility
        return dict(zip(self.strikes, gex_values))

    def predict(self, features: np.ndarray) -> float:
        """
        Calculates the Alpha forecast signal based on proximity to GEX walls.
        Uses the dual-trigger rate-limiting cache to recycle Greeks calculations.
        """
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        # 1. Dynamic recentering of options chain on spot price drift > 1.5%
        if len(self.strikes) == 0 or (self.base_price is not None and abs(mid_price - self.base_price) / self.base_price > 0.015):
            self._initialize_options_chain(mid_price, rolling_vol)

        # 2. Dual-Trigger Cache Verification
        now = time.time()
        self.ticks_since_update += 1
        
        if (self.cached_gex_profile is None or 
            self.ticks_since_update >= self.update_interval or 
            self.last_update_time is None or 
            (now - self.last_update_time) >= self.time_cache_seconds):
            
            self.cached_gex_profile = self.calculate_gex_profile(mid_price)
            self.ticks_since_update = 0
            self.last_update_time = now

        if not self.cached_gex_profile:
            return 0.0

        # 3. Scan GEX walls for trading signals
        alpha = 0.0
        for strike, gex in self.cached_gex_profile.items():
            price_distance_pct = (mid_price - strike) / strike
            
            # Proximity scan within 0.3% of strike
            if abs(price_distance_pct) <= 0.003:
                # Major structural wall filter
                if abs(gex) >= 100.0:
                    # CASE A: Positive GEX Wall (Mean Reversion expected)
                    if gex > 0:
                        if price_distance_pct < 0 and rolling_imbalance < -self.ofi_threshold:
                            alpha = -self.sensitivity * abs(rolling_imbalance)
                            break
                        elif price_distance_pct > 0 and rolling_imbalance > self.ofi_threshold:
                            alpha = self.sensitivity * abs(rolling_imbalance)
                            break
                            
                    # CASE B: Negative GEX Wall (Gamma squeeze breakout expected)
                    else:
                        if price_distance_pct < 0 and rolling_imbalance > self.ofi_threshold:
                            alpha = self.sensitivity * abs(rolling_imbalance) * 1.5
                            break
                        elif price_distance_pct > 0 and rolling_imbalance < -self.ofi_threshold:
                            alpha = -self.sensitivity * abs(rolling_imbalance) * 1.5
                            break

        return float(np.clip(alpha, -0.005, 0.005))
