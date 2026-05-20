import numpy as np
import logging
from intelligence.base_strategy import BaseAlphaStrategy
from intelligence.legacy.gex_oi_alpha import GEXAlphaStrategy
from intelligence.ml_alpha import LightGBMAlpha

logger = logging.getLogger(__name__)

class HybridMLGEXAlpha(BaseAlphaStrategy):
    """
    State-of-the-Art Hybrid Quantitative strategy combining Limit Order Book (LOB) 
    microstructural features with Black-Scholes Options Gamma Exposure (GEX) boundaries,
    orchestrated through a LightGBM supervised learning prediction model.
    """
    def __init__(
        self, 
        model_path: str = "weights.lgb", 
        numpy_path: str = "weights.npy",
        expiry_days: float = 7.0,
        risk_free_rate: float = 0.05,
        ofi_threshold: float = 0.3,
        sensitivity: float = 0.0015
    ):
        self.ofi_threshold = ofi_threshold
        
        # Instantiate dynamic sub-engines
        self.gex_engine = GEXAlphaStrategy(
            expiry_days=expiry_days,
            risk_free_rate=risk_free_rate,
            ofi_threshold=ofi_threshold,
            sensitivity=sensitivity
        )
        self.ml_engine = LightGBMAlpha(model_path=model_path, numpy_path=numpy_path)

    def predict(self, features: np.ndarray) -> float:
        """
        Hybrid prediction pipeline combining GEX boundaries with ML predictions.
        
        Args:
            features: [z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price]
        """
        z_score, spread_z_score, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features

        # 1. Dynamic lazy-load & auto-recenter of options chain on massive price drifts (> 1.5%)
        if not self.gex_engine.options_chain or (self.gex_engine.base_price is not None and abs(mid_price - self.gex_engine.base_price) / self.gex_engine.base_price > 0.015):
            if self.gex_engine.base_price is not None:
                logger.warning(f"🚨 Spot price (${mid_price:.2f}) drifted > 1.5% from options base price (${self.gex_engine.base_price:.2f}). Re-centering strikes...")
            self.gex_engine._initialize_options_chain(mid_price, rolling_vol)
            
        gex_profile = self.gex_engine.calculate_gex_profile(mid_price)
        if not gex_profile:
            return 0.0
        
        # Find GEX walls
        strikes = np.array(list(gex_profile.keys()))
        gex_values = np.array(list(gex_profile.values()))
        
        # Find closest major strikes with significant GEX magnitude (>= 100.0 cash GEX)
        major_strikes = strikes[np.abs(gex_values) >= 100.0]
        major_gex = gex_values[np.abs(gex_values) >= 100.0]
        
        closest_wall_dist = 999.0
        target_gex = 0.0
        closest_strike = 0.0
        
        for strike, gex in zip(major_strikes, major_gex):
            dist = (mid_price - strike) / strike
            if abs(dist) < abs(closest_wall_dist):
                closest_wall_dist = dist
                target_gex = gex
                closest_strike = strike

        # 2. Query Machine Learning prediction
        ml_return_forecast = self.ml_engine.predict(features)

        # 3. Core Hybrid Decision Logic (Regime-Switching Option Contexts)
        alpha = 0.0
        is_near_wall = abs(closest_wall_dist) <= 0.003 # Proximity scan (0.3%)
        
        if is_near_wall:
            # Case A: Positive GEX pinning wall (Mean Reversion / Volatility Pin)
            if target_gex > 0:
                # ML and order flow must agree on reversal direction
                if closest_wall_dist < 0 and ml_return_forecast < 0 and rolling_imbalance < -self.ofi_threshold:
                    # Approaching Call Wall from below, ML predicts downside, OFI confirms passive selling
                    alpha = ml_return_forecast * 1.5 # Amplify high-conviction short
                    logger.debug(f"🎯 [HYBRID-SHORT] Spot near Positive GEX Overhead Wall (${closest_strike:.2f}). ML ({ml_return_forecast:.5f}) and OFI ({rolling_imbalance:.2f}) agree on rejection. Forecast: {alpha:.5f}")
                elif closest_wall_dist > 0 and ml_return_forecast > 0 and rolling_imbalance > self.ofi_threshold:
                    # Approaching Put Wall from above, ML predicts upside, OFI confirms passive buying
                    alpha = ml_return_forecast * 1.5 # Amplify high-conviction long
                    logger.debug(f"🎯 [HYBRID-LONG] Spot near Positive GEX Downside Wall (${closest_strike:.2f}). ML ({ml_return_forecast:.5f}) and OFI ({rolling_imbalance:.2f}) agree on support. Forecast: {alpha:.5f}")
            
            # Case B: Negative GEX squeeze wall (Momentum Breakout / Volatility Accelerator)
            else:
                # ML and order flow must agree on breakout direction
                if closest_wall_dist < 0 and ml_return_forecast > 0.0002 and rolling_imbalance > self.ofi_threshold:
                    # Breaking out upwards above negative GEX resistance
                    alpha = ml_return_forecast * 2.0 # Heavily amplify breakout momentum entry
                    logger.debug(f"🚀 [HYBRID-BREAKOUT-UP] Spot breaking Negative GEX Wall (${closest_strike:.2f}). ML/OFI predict momentum. Forecast: {alpha:.5f}")
                elif closest_wall_dist > 0 and ml_return_forecast < -0.0002 and rolling_imbalance < -self.ofi_threshold:
                    # Breaking down below negative GEX support
                    alpha = ml_return_forecast * 2.0
                    logger.debug(f"🚀 [HYBRID-BREAKOUT-DOWN] Spot breaking Negative GEX Wall (${closest_strike:.2f}). ML/OFI predict downward crash. Forecast: {alpha:.5f}")

        # 4. If not near any option wall, fallback to standard ML statistical forecasts
        if alpha == 0.0:
            alpha = ml_return_forecast

        return float(np.clip(alpha, -0.005, 0.005))
