import numpy as np
import logging
import os
from collections import deque
from typing import Dict, Optional, Tuple
from hmm_regime import GaussianHMMClassifier

logger = logging.getLogger("PairsTrader")

class HurstExponentFilter:
    """
    Vectorized Hurst Exponent calculation via Rescaled Range (R/S) analysis.
    Measures spread mean-reversion (H < 0.5) vs. trending decoupling (H > 0.5).
    """
    def __init__(self, window: int = 360):
        self.window = window
        self.history = deque(maxlen=window)

    def add_sample(self, val: float) -> Optional[float]:
        self.history.append(val)
        if len(self.history) < self.window:
            return None
            
        arr = np.array(self.history)
        n = len(arr)
        
        lags = [lag for lag in [10, 20, 30, 45, 60, 90, 120, 180, 240] if lag <= self.window // 2]
        rs_vals = []
        
        for lag in lags:
            num_periods = n // lag
            periods = arr[:num_periods * lag].reshape((num_periods, lag))
            
            means = np.mean(periods, axis=1, keepdims=True)
            stds = np.std(periods, axis=1) + 1e-8
            demeaned = periods - means
            cum_deviations = np.cumsum(demeaned, axis=1)
            
            ranges = np.max(cum_deviations, axis=1) - np.min(cum_deviations, axis=1)
            rs = np.mean(ranges / stds)
            rs_vals.append(rs)
            
        H, _ = np.polyfit(np.log(lags), np.log(rs_vals), 1)
        return float(H)


class CointegratedPairsTrader:
    """
    Hurst-Filtered Cointegrated Pairs Trading Spread Engine.
    Tracks BTC/ETH cointegration, calculates dynamic OLS hedge ratios,
    and executes delta-neutral swing trades adaptive to statistical regimes.
    """
    def __init__(self, 
                 lookback_window: int = 360,  # Regression window
                 entry_z: float = 2.0,        # Entry boundary
                 exit_z: float = 0.1,        # Profit target reversion
                 stop_loss_z: float = 3.5):   # Risk limit stop
                 
        self.lookback = lookback_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        
        # Core datasets
        self.btc_prices = deque(maxlen=lookback_window)
        self.eth_prices = deque(maxlen=lookback_window)
        self.spread_history = deque(maxlen=lookback_window)
        
        # Mathematical sub-engines
        self.hurst = HurstExponentFilter(window=lookback_window)
        self.hmm = GaussianHMMClassifier()
        
        # Cointegration parameters
        self.beta = 1.0
        self.alpha = 0.0
        self.rolling_mean = 0.0
        self.rolling_std = 1.0
        
        # Position states
        self.in_position = False
        self.position_type = None
        self.entry_regime = None  # Tracks "MEAN_REVERSION" or "TREND_FOLLOWING"
        self.btc_entry_price = 0.0
        self.eth_entry_price = 0.0
        self.entry_tick_counter = 0
        self.max_holding_period = 48  # Dynamic expected holding limit
        
        # Frozen coefficients to prevent Beta Drift
        self.entry_beta = 1.0
        self.entry_alpha = 0.0
        self.entry_mean = 0.0
        self.entry_std = 1.0
        self.entry_btc_qty = 0.0
        self.entry_eth_qty = 0.0
        
        # Cooldown states
        self.global_tick_counter = 0
        self.cooldown_until_tick = 0
        
        # Configurable capital allocation sizing
        self.allocation_fraction = float(os.getenv("ALLOCATION_FRACTION", "0.05"))

    def _record_entry_states(self, btc_price: float, eth_price: float, regime: str, position_type: str, capital: float):
        """
        Modularized helper to freeze coefficients and dynamically calculate
        the Ornstein-Uhlenbeck statistical mean-reversion half-life on entry.
        """
        self.in_position = True
        self.position_type = position_type
        self.entry_regime = regime
        self.btc_entry_price = btc_price
        self.eth_entry_price = eth_price
        
        self.entry_beta = self.beta
        self.entry_alpha = self.alpha
        self.entry_mean = self.rolling_mean
        self.entry_std = self.rolling_std
        self.entry_tick_counter = self.global_tick_counter
        
        allocated_cash = self.allocation_fraction * capital
        self.entry_btc_qty = allocated_cash / btc_price
        self.entry_eth_qty = (self.entry_beta * allocated_cash) / eth_price
        
        # Dynamic OU expected Half-life calculation
        if len(self.spread_history) >= 24:
            spreads = np.array(list(self.spread_history))
            x_ar = spreads[:-1]
            y_ar = spreads[1:]
            a, b = np.polyfit(x_ar, y_ar, 1)
            a_clipped = max(min(a, 0.99), 1e-4)
            lambda_mr = -np.log(a_clipped)
            t_half = 0.69315 / (lambda_mr + 1e-8)
            # Expected max holding is 2x statistical half-life, bounded between 12 and 96 hours
            self.max_holding_period = int(max(min(2.0 * t_half, 96.0), 12.0))
        else:
            self.max_holding_period = 48

    def ingest_prices(self, btc_price: float, eth_price: float) -> Optional[float]:
        self.global_tick_counter += 1
        
        self.btc_prices.append(np.log(btc_price))
        self.eth_prices.append(np.log(eth_price))
        
        if len(self.btc_prices) < self.lookback:
            return None
            
        # Ordinary Least Squares (OLS) hedge estimation
        x = np.array(self.eth_prices)
        y = np.array(self.btc_prices)
        beta, alpha = np.polyfit(x, y, 1)
        self.beta = beta
        self.alpha = alpha
        
        spread = y[-1] - (self.beta * x[-1] + self.alpha)
        self.spread_history.append(spread)
        
        # Calculate rolling metrics of spreads
        self.rolling_mean = np.mean(self.spread_history)
        self.rolling_std = np.std(self.spread_history) + 1e-8
        
        z_score = (spread - self.rolling_mean) / self.rolling_std
        return z_score

    def evaluate_trade_setup(self, btc_price: float, eth_price: float, z_score: float, capital: float = 10000.0) -> Optional[Dict]:
        if z_score is None:
            return None
            
        # Enforce entry cooldown lock
        if not self.in_position and self.global_tick_counter < self.cooldown_until_tick:
            return None
            
        # Hurst Exponent Regime Lockout Gate
        if not self.in_position:
            hurst_val = self.hurst.add_sample(self.spread_history[-1])
            if hurst_val is not None and hurst_val >= 0.48:
                # Spread is trending/decoupling. Block entry!
                return None
                
        # Spread Trend Filter (calculate OLS slope of the last 24 hourly spreads)
        spread_trend = 0.0
        if len(self.spread_history) >= 24:
            y_trend = np.array(list(self.spread_history))[-24:]
            x_trend = np.arange(24)
            slope, _ = np.polyfit(x_trend, y_trend, 1)
            spread_trend = slope
            
        # Dynamic Hidden Markov Model (HMM) Volatility Regime Classification
        active_regime = self.hmm.classify_tick(z_score=z_score, spread_trend=spread_trend)
        
        # State 2 = Decoupling / Liquidation State -> Lock all entries!
        if not self.in_position and active_regime == 2:
            return None
            
        is_ranging = (active_regime == 0)
        is_trending = (active_regime == 1)
                
        exec_cmd = None
        
        if not self.in_position:
            # --- ENTRY DECISIONS ---
            
            # A. MEAN REVERSION REGIME (Sideways range consolidation)
            if is_ranging:
                if z_score >= self.entry_z:
                    self._record_entry_states(btc_price, eth_price, "MEAN_REVERSION", "SHORT_SPREAD", capital)
                    exec_cmd = {
                        "action": "ENTRY",
                        "type": "SHORT_SPREAD",
                        "btc_order": {"side": "SELL", "price": btc_price, "qty": float(self.entry_btc_qty)},
                        "eth_order": {"side": "BUY", "price": eth_price, "qty": float(self.entry_eth_qty)}
                    }
                    
                elif z_score <= -self.entry_z:
                    self._record_entry_states(btc_price, eth_price, "MEAN_REVERSION", "LONG_SPREAD", capital)
                    exec_cmd = {
                        "action": "ENTRY",
                        "type": "LONG_SPREAD",
                        "btc_order": {"side": "BUY", "price": btc_price, "qty": float(self.entry_btc_qty)},
                        "eth_order": {"side": "SELL", "price": eth_price, "qty": float(self.entry_eth_qty)}
                    }
            
            # B. CTA TREND-FOLLOWING REGIME (MOMENTUM BREAKOUT)
            elif is_trending:
                if spread_trend > 0.0002 and z_score >= 1.5:
                    self._record_entry_states(btc_price, eth_price, "TREND_FOLLOWING", "LONG_SPREAD", capital)
                    exec_cmd = {
                        "action": "ENTRY",
                        "type": "LONG_SPREAD",
                        "btc_order": {"side": "BUY", "price": btc_price, "qty": float(self.entry_btc_qty)},
                        "eth_order": {"side": "SELL", "price": eth_price, "qty": float(self.entry_eth_qty)}
                    }
                    
                elif spread_trend < -0.0002 and z_score <= -1.5:
                    self._record_entry_states(btc_price, eth_price, "TREND_FOLLOWING", "SHORT_SPREAD", capital)
                    exec_cmd = {
                        "action": "ENTRY",
                        "type": "SHORT_SPREAD",
                        "btc_order": {"side": "SELL", "price": btc_price, "qty": float(self.entry_btc_qty)},
                        "eth_order": {"side": "BUY", "price": eth_price, "qty": float(self.entry_eth_qty)}
                    }
                
        else:
            # --- ACTIVE POSITION MANAGEMENT (EXIT DECISIONS) ---
            ln_btc = np.log(btc_price)
            ln_eth = np.log(eth_price)
            static_spread = ln_btc - (self.entry_beta * ln_eth + self.entry_alpha)
            static_z = (static_spread - self.entry_mean) / self.entry_std
            
            should_exit = False
            exit_type = None
            
            # A. Check Ornstein-Uhlenbeck Expected Half-Life Position Time-Out!
            holding_duration = self.global_tick_counter - self.entry_tick_counter
            if holding_duration >= self.max_holding_period:
                should_exit = True
                exit_type = "TIME_OUT"
                
            # B. HMM Emergency Liquidation Stop
            else:
                active_regime = self.hmm.classify_tick(z_score=static_z, spread_trend=spread_trend)
                if active_regime == 2:
                    should_exit = True
                    exit_type = "STOP_LOSS"
                    
                # C. MEAN REVERSION EXITS
                elif self.entry_regime == "MEAN_REVERSION":
                    if abs(static_z) >= self.stop_loss_z:
                        should_exit = True
                        exit_type = "STOP_LOSS"
                    elif self.position_type == "SHORT_SPREAD" and static_z <= self.exit_z:
                        should_exit = True
                        exit_type = "TAKE_PROFIT"
                    elif self.position_type == "LONG_SPREAD" and static_z >= -self.exit_z:
                        should_exit = True
                        exit_type = "TAKE_PROFIT"
                        
                # D. TREND-FOLLOWING EXITS (MOMENTUM EXHAUSTION)
                elif self.entry_regime == "TREND_FOLLOWING":
                    is_trend_reversed = (self.position_type == "LONG_SPREAD" and spread_trend < -0.0002) or \
                                        (self.position_type == "SHORT_SPREAD" and spread_trend > 0.0002)
                                        
                    if is_trend_reversed:
                        should_exit = True
                        exit_type = "STOP_LOSS"
                    elif self.position_type == "LONG_SPREAD" and (spread_trend <= 0.0 or static_z <= 0.2):
                        should_exit = True
                        exit_type = "TAKE_PROFIT"
                    elif self.position_type == "SHORT_SPREAD" and (spread_trend >= 0.0 or static_z >= -0.2):
                        should_exit = True
                        exit_type = "TAKE_PROFIT"
                
            if should_exit:
                btc_exit_side = "BUY" if self.position_type == "SHORT_SPREAD" else "SELL"
                eth_exit_side = "SELL" if self.position_type == "SHORT_SPREAD" else "BUY"
                
                exec_cmd = {
                    "action": "EXIT",
                    "type": exit_type,
                    "btc_order": {"side": btc_exit_side, "price": btc_price, "qty": float(self.entry_btc_qty)},
                    "eth_order": {"side": eth_exit_side, "price": eth_price, "qty": float(self.entry_eth_qty)}
                }
                
                # Reset states and set cooldown
                self.in_position = False
                self.position_type = None
                self.entry_regime = None
                self.btc_entry_price = 0.0
                self.eth_entry_price = 0.0
                self.entry_beta = 1.0
                self.entry_alpha = 0.0
                self.entry_mean = 0.0
                self.entry_std = 1.0
                self.entry_btc_qty = 0.0
                self.entry_eth_qty = 0.0
                self.cooldown_until_tick = self.global_tick_counter + 60
                
        return exec_cmd
