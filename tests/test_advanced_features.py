import pytest
import numpy as np
import gc
import os
from typing import List

from core.schemas import InternalTick
from perception.feature_store import FeatureStore
from intelligence.strategy_factory import AlphaStrategyFactory
from intelligence.legacy.micro_trend_alpha import MicroTrendMomentumAlpha
from intelligence.ml_alpha import LightGBMAlpha
from execution.execution_gateway import BinanceExecutionGateway
from intelligence.legacy.gex_oi_alpha import GEXAlphaStrategy

# Dynamically register legacy strategies for factory tests
AlphaStrategyFactory._REGISTRY["MICRO_TREND"] = MicroTrendMomentumAlpha
AlphaStrategyFactory._REGISTRY["GEX"] = GEXAlphaStrategy

# ==============================================================================
# 1. Brute-Force Reference Feature Store (For Mathematical Equivalence validation)
# ==============================================================================
class BruteForceFeatureStore:
    """Slow, O(N) reference implementation to mathematically validate the O(1) optimized version."""
    def __init__(self, window_size: int = 1000, lookback: int = 50):
        self.lookback = lookback
        self.mids = []
        self.spreads = []
        self.imbalances = []

    def process_tick(self, tick: InternalTick):
        mid_price = (tick.bid + tick.ask) / 2.0
        spread = tick.ask - tick.bid
        imbalance = (tick.bid_size - tick.ask_size) / (tick.bid_size + tick.ask_size + 1e-8)
        micro_price = (tick.bid * tick.ask_size + tick.ask * tick.bid_size) / (tick.bid_size + tick.ask_size + 1e-8)
        micro_price_drift = micro_price - mid_price

        self.mids.append(mid_price)
        self.spreads.append(spread)
        self.imbalances.append(imbalance)

        if len(self.mids) < self.lookback:
            return None

        # Get last 50 elements (brute force slice)
        mids_window = np.array(self.mids[-self.lookback:])
        spreads_window = np.array(self.spreads[-self.lookback:])
        imbalances_window = np.array(self.imbalances[-self.lookback:])

        # Slow standard deviation and mean calculations
        rolling_vol = np.std(mids_window)
        rolling_mean = np.mean(mids_window)
        z_score = (mid_price - rolling_mean) / (rolling_vol + 1e-8)

        rolling_imbalance = np.mean(imbalances_window)

        return np.array([
            z_score,
            spread,
            rolling_imbalance,
            micro_price_drift,
            rolling_vol,
            mid_price
        ])

# ==============================================================================
# 2. O(1) Numerical Equivalence Verification Test
# ==============================================================================
def test_feature_store_numerical_equivalence():
    # Generate a deterministic mock tick series of 120 ticks
    np.random.seed(42)
    ticks: List[InternalTick] = []
    base_price = 60000.0
    
    for i in range(120):
        base_price += np.random.normal(0.0, 5.0)
        spread = np.random.uniform(0.5, 2.0)
        tick = InternalTick(
            symbol="BTCUSDT",
            exchange="BINANCE",
            bid=base_price - spread/2.0,
            ask=base_price + spread/2.0,
            bid_size=np.random.uniform(1.0, 5.0),
            ask_size=np.random.uniform(1.0, 5.0),
            timestamp_ns=i * 100_000_000
        )
        ticks.append(tick)

    # Instantiate both Stores
    optimized_store = FeatureStore(window_size=100, lookback=50)
    reference_store = BruteForceFeatureStore(lookback=50)

    # Feed ticks and assert mathematical identical output values
    for idx, tick in enumerate(ticks):
        opt_feats = optimized_store.process_tick(tick)
        ref_feats = reference_store.process_tick(tick)

        if ref_feats is None:
            assert opt_feats is None
        else:
            assert opt_feats is not None
            # Check that z-score, spread z-score, rolling imbalance, vol, mid, etc. match perfectly
            # Float calculations can have minor rounding wicks, so we use 1e-9 tolerance
            np.testing.assert_allclose(opt_feats, ref_feats, rtol=1e-9, atol=1e-9)

# ==============================================================================
# 3. Strategy Factory & Dynamic Kwargs Signature Testing
# ==============================================================================
def test_strategy_factory_dynamic_argument_filtering():
    # Pass redundant/invalid arguments into factory
    # MICRO_TREND strategy does NOT accept model_path or extra args in constructor
    strategy = AlphaStrategyFactory.create_strategy(
        alpha_type="MICRO_TREND",
        threshold=0.45,
        w=0.80,
        model_path="weights.lgb",     # Redundant, should be filtered out safely
        redundant_arg="hello"         # Redundant, should be filtered out safely
    )

    assert isinstance(strategy, MicroTrendMomentumAlpha)
    assert strategy.threshold == 0.45
    assert strategy.w == 0.80

    # Pass ML class weights path to factory
    ml_strategy = AlphaStrategyFactory.create_strategy(
        alpha_type="ML",
        model_path="missing_weights.lgb"
    )
    assert isinstance(ml_strategy, LightGBMAlpha)

# ==============================================================================
# 4. Garbage Collection (GC) State Verification Test
# ==============================================================================
def test_garbage_collection_tuning_states():
    # Check initial GC state
    was_enabled = gc.isenabled()
    
    # Simulate disabling GC
    gc.disable()
    assert gc.isenabled() is False
    
    # Check manual collect runs successfully
    collected = gc.collect()
    assert isinstance(collected, int)
    
    # Re-enable to keep test environment clean
    if was_enabled:
        gc.enable()

# ==============================================================================
# 5. High-Fidelity Short Inventory Simulation Test
# ==============================================================================
@pytest.mark.asyncio
async def test_short_inventory_sizing_fills():
    # Instantiate Gateway in Paper Trading Mode
    gateway = BinanceExecutionGateway(paper_trading=True)
    
    # 1. Simulate Short SELL entry
    # Sell 6,000 notional at limit price of 60,000 (quantity = 0.1)
    order_payload = {
        "symbol": "BTCUSDT",
        "action": "SELL",
        "limit_price": 60000.0,
        "notional": 6000.0,
        "type": "limit"
    }
    
    report = await gateway.send_order(order_payload)
    
    assert report["status"] == "FILLED"
    assert report["action"] == "SELL"
    
    # Check cash and crypto sizing transitions
    executed_qty_crypto = report["executed_qty_crypto"]
    executed_qty_cash = report["executed_qty_cash"]
    fee_paid = report["fee_paid"]
    
    # Short inventory quantity is correctly returned
    assert executed_qty_crypto == 0.1
    # Cash received includes fee deduction: (0.1 * price) - fee
    assert executed_qty_cash == 6000.0 - fee_paid
    
    # 2. Simulate exit BUY (buy back short)
    # Quantity to buy back must be the absolute value of short units (0.1)
    buy_back_payload = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "limit_price": 60000.0,
        "quantity": executed_qty_crypto,
        "type": "limit"
    }
    
    buy_report = await gateway.send_order(buy_back_payload)
    
    assert buy_report["status"] == "FILLED"
    assert buy_report["action"] == "BUY"
    # Buy back crypto units matching entry
    assert buy_report["executed_qty_crypto"] > 0

# ==============================================================================
# 6. Paper Starvation & Protective Collar Testing
# ==============================================================================
@pytest.mark.asyncio
async def test_execution_starvation_protective_collar():
    # Instantiate Gateway with extremely tight collar (0.1 bp)
    gateway = BinanceExecutionGateway(paper_trading=True)
    gateway.max_slippage_pct = 0.000001 # extremely tight protective collar
    
    # Submit a market SELL order
    # Gateway will convert this to limit order at collared price
    order_payload = {
        "symbol": "BTCUSDT",
        "action": "SELL",
        "type": "market",
        "mid_price": 60000.0,
        "quantity": 0.1
    }
    
    report = await gateway.send_order(order_payload)
    
    # Due to price drift during simulated latency, this should starve/cancel!
    assert report["status"] == "CANCELLED"

# ==============================================================================
# 7. OMS Emergency Liquidation (Short & Long) Testing
# ==============================================================================
@pytest.mark.asyncio
async def test_emergency_liquidation_short_positions():
    from execution.oms import OrderManagementSystem
    from unittest.mock import AsyncMock, MagicMock

    gateway_mock = MagicMock()
    gateway_mock.send_order = AsyncMock(return_value={"status": "FILLED"})
    
    oms = OrderManagementSystem(gateway=gateway_mock)
    
    # Inventory holds a SHORT position of 0.5 BTC
    current_inventory = {"BTCUSDT": -0.5}
    
    await oms.liquidate_all(current_inventory)
    
    # Verify that gateway received a BUY market order of size 0.5 to flatten short position
    gateway_mock.send_order.assert_called_once()
    called_payload = gateway_mock.send_order.call_args[0][0]
    assert called_payload["symbol"] == "BTCUSDT"
    assert called_payload["action"] == "BUY"
    assert called_payload["quantity"] == 0.5
    assert called_payload["type"] == "market"

# ==============================================================================
# 8. GEX Options Framework Verification Test
# ==============================================================================
def test_gex_strategy_registration_and_prediction():
    from intelligence.legacy.gex_oi_alpha import GEXAlphaStrategy
    
    # Verify factory creation
    strategy = AlphaStrategyFactory.create_strategy(
        alpha_type="GEX",
        expiry_days=7.0,
        ofi_threshold=0.3
    )
    
    assert isinstance(strategy, GEXAlphaStrategy)
    assert strategy.ofi_threshold == 0.3
    
    # Verify signal computation when spot is flat/far from walls
    features_far = np.array([0.0, 0.0, 0.0, 0.0, 0.001, 60000.0])
    prediction = strategy.predict(features_far)
    assert prediction == 0.0
    
    # Check options chain is generated
    assert len(strategy.options_chain) > 0
    
    # Verify GEX wall rejection/reversal short
    # Base price is 60000.0. Call wall is at +0.4% which is 60240.0.
    # Proximity threshold is 0.3%, so 60234.0 is near the wall.
    # Selling pressure: rolling_imbalance = -0.5 (< -ofi_threshold)
    mid_price_near_wall = 60000.0 * 1.0039 # 60234.0 (near 60240.0 strike)
    features_short = np.array([1.0, 1.0, -0.5, 0.0, 0.001, mid_price_near_wall])
    prediction_short = strategy.predict(features_short)
    assert prediction_short < 0.0
    
    # Verify GEX wall rejection/reversal long
    # Base price is 60000.0. Put wall is at -0.4% which is 59760.0.
    # Proximity threshold is 0.3%, so 59766.0 is near the wall.
    # Buying pressure: rolling_imbalance = 0.5 (> ofi_threshold)
    mid_price_near_put_wall = 60000.0 * 0.9961 # 59766.0 (near 59760.0 strike)
    features_long = np.array([-1.0, -1.0, 0.5, 0.0, 0.001, mid_price_near_put_wall])
    prediction_long = strategy.predict(features_long)
    assert prediction_long > 0.0

# ==============================================================================
# 9. Hybrid ML-GEX Strategy Verification Test
# ==============================================================================
def test_hybrid_strategy_registration_and_prediction():
    from intelligence.hybrid_ml_gex_alpha import HybridMLGEXAlpha
    
    # Verify factory creation
    strategy = AlphaStrategyFactory.create_strategy(
        alpha_type="HYBRID",
        ofi_threshold=0.3
    )
    
    assert isinstance(strategy, HybridMLGEXAlpha)
    assert strategy.ofi_threshold == 0.3
    
    # Verify far-from-wall fallback to ML prediction
    features_far = np.array([0.0, 0.0, 0.0, 0.0, 0.001, 60000.0])
    prediction = strategy.predict(features_far)
    assert isinstance(prediction, float)
    
    # Verify options chain is successfully initialized under the hood
    assert len(strategy.gex_engine.options_chain) > 0


