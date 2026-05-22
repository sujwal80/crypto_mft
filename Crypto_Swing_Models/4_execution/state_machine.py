import asyncio
import logging
import time
import sys
import os

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_alpha_macro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../3_alpha_micro")))

from welford_stats import WelfordRollingStats
from kalman_filter import MicroPriceKalmanFilter
from cvd_engine import CumulativeVolumeDeltaEngine
from risk_gate import RiskGate
from basis_tracker import BasisTracker
from smart_router import SmartRouter

logger = logging.getLogger(__name__)

class GexMicroStateMachine:
    """
    The master brain of the GEX-Micro State Machine.
    Manages transitions between:
      - STATE 0: Hibernation & Recon (CPU Throttled, waiting for wall proximity)
      - STATE 1: Armed (Proximity within 0.5% of GEX wall, active micro sensors)
      - STATE 2: Execution (Sniper Trigger on Confirmation Matrix)
      - STATE 3: Dynamic Invalidation (Aggressive Early Exit on structural breakdown)
    """
    def __init__(self, symbol: str = "BTCUSDT", mode: str = "SHADOW"):
        self.symbol = symbol
        self.mode = mode
        
        # Core state variable
        self.state = 0  # Starts in State 0: Hibernation
        
        # Instantiate Subsystems
        self.risk_gate = RiskGate()
        self.basis_tracker = BasisTracker()
        self.smart_router = SmartRouter(mode=mode)
        
        # Perception/Micro Sensors (Armed in State 1)
        self.welford = WelfordRollingStats(window_size=1000)
        self.kalman = MicroPriceKalmanFilter(process_noise=1e-5, measurement_noise=1e-2)
        self.cvd = CumulativeVolumeDeltaEngine(rolling_window_ticks=5000)
        
        # Macro GEX Wall target state variables
        self.active_gex_strike = None
        self.active_gex_val = 0.0
        self.adjusted_target_price = None
        
        # Trade holding state variables
        self.in_position = False
        self.entry_order = None
        self.position_side = None # 'LONG' or 'SHORT'
        self.entry_price = 0.0
        
    async def process_market_tick(self, mid_price: float, bid_qty: float, ask_qty: float, 
                                  is_trade: bool = False, trade_price: float = 0.0, 
                                  trade_qty: float = 0.0, is_buyer_maker: bool = False):
        """
        Ingests high-frequency ticks and handles state transitions based on real-time micro-confirmations.
        """
        mid_price = float(mid_price)
        bid_qty = float(bid_qty)
        ask_qty = float(ask_qty)
        
        # 1. Update perp price in basis tracker
        self.basis_tracker.update_perp_price(mid_price)
        
        # 2. Process Micro statistics if we are not throttled (State >= 1)
        l2_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)
        
        # Compute rolling statistics via O(1) Welford
        mean_imbalance, std_imbalance, z_score = self.welford.update(l2_imbalance)
        
        # Compute Kalman-filtered micro-price
        filtered_price = self.kalman.filter_tick(mid_price)
        
        # Process CVD on live trades
        if is_trade:
            self.cvd.process_trade(trade_price, trade_qty, is_buyer_maker)
            
        # Calculate price movement for absorption detection
        price_change = (filtered_price - mid_price) / mid_price
        
        # State Machine Transitions
        if self.state == 0:
            await self._handle_state_0_recon(mid_price)
            
        elif self.state == 1:
            await self._handle_state_1_armed(mid_price, z_score, filtered_price, price_change)
            
        elif self.state == 2:
            await self._handle_state_2_execution(mid_price, z_score, filtered_price)
            
        elif self.state == 3:
            await self._handle_state_3_invalidation(z_score, price_change)

    def update_gex_profile(self, key_strike: float, gex_value: float, deribit_index: float):
        """
        Updates GEX positioning details from the macro mapper subsystem.
        Called by the Orchestrator.
        """
        self.active_gex_strike = key_strike
        self.active_gex_val = gex_value
        self.basis_tracker.update_index_price(deribit_index)
        
        # Pre-calculate adjusted target price
        self.adjusted_target_price = self.basis_tracker.adjust_strike_to_execution_target(key_strike)
        logger.info(f"Master State Machine: Active GEX strike level updated to {key_strike} (Adjusted target: {self.adjusted_target_price:.2f})")

    async def _handle_state_0_recon(self, mid_price: float):
        """
        STATE 0: Hibernation & Recon.
        Monitors proximity to key GEX strikes. Wakes up sensors when within 0.5% of a wall.
        """
        if self.adjusted_target_price is None:
            return
            
        distance_pct = abs(mid_price - self.adjusted_target_price) / mid_price
        
        if distance_pct <= 0.005: # Approaching within 0.5%
            logger.info(f"Transition: STATE 0 -> STATE 1 (Proximity Armed within {distance_pct*100:.3f}%)")
            self.state = 1
            
    async def _handle_state_1_armed(self, mid_price: float, z_score: float, filtered_price: float, price_change: float):
        """
        STATE 1: Armed (Proximity).
        Wakes up micro sensors and tests entry profit viability at the risk gate.
        """
        if self.adjusted_target_price is None:
            self.state = 0
            return
            
        distance_pct = abs(mid_price - self.adjusted_target_price) / mid_price
        
        # If spot drifts away from the wall, return to State 0 to save CPU resources
        if distance_pct > 0.006:
            logger.info(f"Transition: STATE 1 -> STATE 0 (Spot drifted to {distance_pct*100:.3f}%)")
            self.state = 0
            return
            
        # Evaluate entry expectancy at Risk Gate
        # Demand at least 0.2% round-trip expected move
        expected_move = 0.0025 # 0.25% anticipated bounce from major GEX wall
        if not self.risk_gate.evaluate_entry(mid_price, self.adjusted_target_price, expected_move):
            logger.warning("Risk Gate blocked entry, locking transition.")
            return
            
        # If we hit the basis-adjusted strike target, transition to State 2
        # For long entry (approaching Put wall from above) or short entry (approaching Call wall from below)
        if abs(mid_price - self.adjusted_target_price) / mid_price <= 0.0008:
            logger.info("Transition: STATE 1 -> STATE 2 (Adjusted Target Hit, verifying Confirmation Matrix)")
            self.state = 2

    async def _handle_state_2_execution(self, mid_price: float, z_score: float, filtered_price: float):
        """
        STATE 2: Sniper Trigger.
        Requires L2 Z-score and CVD-Kalman absorption confirmation before submitting maker orders.
        """
        # Check Confirmation Matrix for Reversals at positive Put Walls:
        # 1. L2 bid imbalance Z-score must spike > +2.0 (Bids stacked up)
        # 2. CVD engine reports passive absorption (sellers hitting bids but price refuses to budge)
        is_long_confirmed = (z_score >= 2.0) and self.cvd.detect_absorption_divergence(0.0)
        
        # For Short entries at Call Walls (imbalance Z-score < -2.0)
        is_short_confirmed = (z_score <= -2.0) and self.cvd.detect_absorption_divergence(0.0)
        
        if is_long_confirmed:
            logger.info(f"CONFIRMED: Long reversal triggered at {mid_price:.2f}. Submitting maker order.")
            self.entry_order = await self.smart_router.place_post_only_limit(self.symbol, "BUY", mid_price - 0.5, 1.0)
            self.in_position = True
            self.position_side = "LONG"
            self.entry_price = mid_price
            self.state = 3
            
        elif is_short_confirmed:
            logger.info(f"CONFIRMED: Short reversal triggered at {mid_price:.2f}. Submitting maker order.")
            self.entry_order = await self.smart_router.place_post_only_limit(self.symbol, "SELL", mid_price + 0.5, 1.0)
            self.in_position = True
            self.position_side = "SHORT"
            self.entry_price = mid_price
            self.state = 3
            
        else:
            # If confirmation fails, revert to Armed to prevent blind catches
            logger.debug("Matrix confirmation failed, remaining in STATE 2")

    async def _handle_state_3_invalidation(self, z_score: float, price_change: float):
        """
        STATE 3: Dynamic Invalidation & Exit.
        Monitors the trade for structural breakdown (Z-score drops to -3.0, CVD panic sell).
        Fires immediate market orders for early cuts, protecting capital.
        """
        if not self.in_position:
            self.state = 0
            return
            
        # Check invalidation conditions:
        # 1. L2 Bids vanish instantly (Z-score drops to -3.0)
        # 2. Taker selling escalates (aggression ratio collapses)
        should_invalidate = False
        
        if self.position_side == "LONG":
            if z_score <= -3.0 or self.cvd.get_aggression_ratio() < 0.2:
                should_invalidate = True
        else: # SHORT
            if z_score >= 3.0 or self.cvd.get_aggression_ratio() > 5.0:
                should_invalidate = True
                
        if should_invalidate:
            logger.warning(f"INVALIDATION TRIGGERED: Microstructure deteriorated (Z-score: {z_score:.2f}). CUTTING IMMEDIATELY.")
            exit_side = "SELL" if self.position_side == "LONG" else "BUY"
            await self.smart_router.place_market_order(self.symbol, exit_side, 1.0)
            
            # Reset State variables
            self.in_position = False
            self.position_side = None
            self.entry_order = None
            self.state = 0
