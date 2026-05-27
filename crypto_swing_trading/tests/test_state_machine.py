import sys
import os
import pytest
import asyncio

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../4_execution")))
from state_machine import GexMicroStateMachine

@pytest.mark.asyncio
async def test_state_machine_lifecycle():
    """Verify the complete 4-state lifecycle: Hibernation -> Armed -> Execution -> Invalidation."""
    sm = GexMicroStateMachine(symbol="BTCUSDT", mode="SHADOW", grace_window_ticks=0, resample_ticks=1)
    
    # 1. Set macro wall targets
    # Strike at 60000, GEX is positive (Put Wall), Index is at 60000
    sm.update_gex_profile(key_strike=60000.0, gex_value=150.0, deribit_index=60000.0)
    
    # State starts in Hibernation (0)
    assert sm.state == 0
    
    # Warm up the Welford rolling statistics with balanced book ticks (imbalance = 0.0)
    for _ in range(50):
        await sm.process_market_tick(mid_price=61000.0, bid_qty=10.0, ask_qty=10.0)
        
    assert sm.state == 0
    
    # 2. Feed price approaching target (within 0.5% -> 60200): Should transition to Armed (1)
    await sm.process_market_tick(mid_price=60200.0, bid_qty=10.0, ask_qty=10.0)
    assert sm.state == 1
    
    # 3. Hit target price (60000): Should transition to Execution verification (2)
    await sm.process_market_tick(mid_price=60000.0, bid_qty=10.0, ask_qty=10.0)
    assert sm.state == 2
    
    # 4. Simulate trade history with heavy taker selling (is_buyer_maker=True) to prime CVD engine
    for _ in range(20):
        await sm.process_market_tick(
            mid_price=60000.0, 
            bid_qty=10.0, 
            ask_qty=10.0, 
            is_trade=True, 
            trade_price=60000.0, 
            trade_qty=10.0, 
            is_buyer_maker=True
        )
        
    # 5. Inject a sudden massive Bid imbalance spike to trigger Z-score > 2.0
    await sm.process_market_tick(
        mid_price=60000.0, 
        bid_qty=500.0, 
        ask_qty=10.0, 
        is_trade=True, 
        trade_price=60000.0, 
        trade_qty=10.0, 
        is_buyer_maker=True
    )
        
    # The state machine confirms triggers, submits limit order, but remains in STATE 2 (OPEN order)
    assert sm.state == 2
    assert sm.entry_order is not None
    assert sm.entry_order["status"] == "OPEN"
    assert sm.entry_order["price"] == 59999.5
    assert sm.in_position == False
    
    # Feed a tick that crosses below the limit price (59999.5) to trigger the fill.
    # Include some buyer trades to show support and keep the CVD aggression ratio healthy (> 0.2).
    await sm.process_market_tick(
        mid_price=59999.0, 
        bid_qty=10.0, 
        ask_qty=10.0,
        is_trade=True,
        trade_price=59999.0,
        trade_qty=100.0,
        is_buyer_maker=False  # Buyer taker (aggressive buying support)
    )
    
    # Now the order should be filled, and we transition to State 3
    assert sm.state == 3
    assert sm.in_position == True
    assert sm.position_side == "LONG"
    assert sm.entry_price == 59999.5
    
    # 6. Inject Invalidation trigger (Retail liquidations / bid vanish):
    # - Bids drop to zero, ask qty spikes to generate massive negative Z-score
    # - Taker selling escalates
    await sm.process_market_tick(mid_price=59900.0, bid_qty=1.0, ask_qty=1000.0)
    
    # Should trigger market order to cut the loss, reset position and return to Hibernation (0)
    assert sm.state == 0
    assert sm.in_position == False

