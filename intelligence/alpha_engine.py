import os
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class AlphaModel:
    """Evaluates mathematical/ML models to generate return forecasts. Supports LightGBM boosters."""
    def __init__(self, model_path: str = "weights.lgb"):
        self.model_path = model_path
        self.booster = None
        self.current_forecast = 0.0 # Holds the current active signal
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                import lightgbm as lgb
                self.booster = lgb.Booster(model_file=self.model_path)
                logger.info(f"Successfully loaded LightGBM booster from {self.model_path}")
            except ImportError:
                logger.warning("LightGBM library not found. Falling back to statistical alpha model.")
            except Exception as e:
                logger.error(f"Failed to load LightGBM model: {e}. Falling back to statistical alpha model.")
        else:
            logger.info(f"Model weights file {self.model_path} not found. Using Regime-Aware Statistical Alpha model.")

    def predict(self, features: np.ndarray) -> float:
        """Predicts expected return forecast. Uses LightGBM if available, otherwise Regime-Aware statistical equations."""
        if self.booster:
            # LightGBM inference on feature vector
            forecast = self.booster.predict([features])[0]
            return float(forecast)
        else:
            # ==========================================================
            # Regime-Aware Statistical Alpha (Fixing Unprofitable Trading)
            # ==========================================================
            z_score, spread, imbalance, mid = features
            
            # 1. Regime Filter (Detecting Trend vs Mean Reversion)
            # In crypto, strong trends crush mean-reversion strategies.
            is_trending = abs(z_score) > 2.0
            
            if is_trending:
                # Momentum Regime: Follow the trend if order book supports it
                if z_score > 2.0 and imbalance > 0.5:
                    self.current_forecast = 0.005 # Strong Buy (5% Allocation)
                elif z_score < -2.0 and imbalance < -0.5:
                    self.current_forecast = -0.005 # Strong Sell (5% Allocation)
            else:
                # Ranging Regime: Mean-revert only when imbalance confirms exhaustion
                if z_score > 1.0 and imbalance < -0.4:
                    self.current_forecast = -0.002 # 2% Allocation
                elif z_score < -1.0 and imbalance > 0.4:
                    self.current_forecast = 0.002 # 2% Allocation
                else:
                    # HOLD THE TREND LOGIC
                    # If we have an active position, DO NOT close it until the Z-Score crosses 0 (Mean Reversion)
                    if self.current_forecast > 0 and z_score < 0.0:
                        self.current_forecast = 0.0 # Exit Long
                    elif self.current_forecast < 0 and z_score > 0.0:
                        self.current_forecast = 0.0 # Exit Short
                    
            return float(self.current_forecast)

class PortfolioOptimizer:
    """Applies Kelly Criterion and Volatility Scaling to determine target weights."""
    def calculate_target_weight(self, forecast: float) -> float:
        if abs(forecast) < 0.001:
            return 0.0 # Filter out weak conviction signals entirely
            
        # Kelly Sizing: Scaled conviction
        base_weight = forecast * 10.0
        return min(max(base_weight, -0.2), 0.2)  # Cap exposure at 20% per asset

class OrderGenerator:
    """Generates rebalancing orders by comparing target weights against current inventory."""
    def generate_order(self, symbol: str, target_weight: float, current_inventory: float, portfolio_value: float, bid: float, ask: float) -> Optional[Dict]:
        target_cash_value = target_weight * portfolio_value
        cash_delta = target_cash_value - current_inventory
        
        # ==========================================================
        # Over-Trading & Fee Protection Fix
        # ==========================================================
        # 1. Rebalance Threshold ($200 dead-band)
        if abs(cash_delta) > 200.0:
            action = "BUY" if cash_delta > 0 else "SELL"
            
            # 2. Maker-Only Limit Pricing (Stop crossing the spread!)
            # Instead of buying at Ask (paying Taker fee), place limit order at Best Bid.
            limit_price = bid if action == "BUY" else ask
            
            return {
                "symbol": symbol,
                "action": action,
                "notional": abs(cash_delta),
                "limit_price": limit_price
            }
        return None
