import os
import json
import pytest
import ccxt
import time
from unittest.mock import MagicMock, AsyncMock

from execution.dead_letter_queue import DeadLetterQueue
from execution.risk_guardrails import RiskGuardrailEngine
from execution.oms import OrderManagementSystem
from execution.execution_gateway import BinanceExecutionGateway

# ==============================================================================
# 1. Dead Letter Queue Tests
# ==============================================================================
def test_dead_letter_queue_audit(tmp_path):
    audit_file = tmp_path / "dlq_audit.json"
    dlq = DeadLetterQueue(audit_path=str(audit_file))
    
    test_payload = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "reason": "FAT_FINGER_COLLAR",
        "price": 100000.0
    }
    
    dlq.write_to_dlq(test_payload)
    
    assert os.path.exists(audit_file)
    with open(audit_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    assert len(records) == 1
    assert records[0]["symbol"] == "BTCUSDT"
    assert records[0]["reason"] == "FAT_FINGER_COLLAR"
    assert "timestamp_ns" in records[0]

# ==============================================================================
# 2. Risk Guardrails Tests
# ==============================================================================
def test_risk_guardrail_valid_order():
    guard = RiskGuardrailEngine(max_drawdown_limit=10.0, fat_finger_collar=0.02)
    
    order = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "price": 60000.0,
        "amount_crypto": 0.1
    }
    
    approved, reason = guard.validate_order(order, mid_price=60000.0)
    assert approved is True
    assert reason == "APPROVED"

def test_risk_guardrail_fat_finger_collar():
    guard = RiskGuardrailEngine(max_drawdown_limit=10.0, fat_finger_collar=0.02)
    
    # 65000 is > 2% from mid price 60000 (which would be 61200 max)
    order = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "price": 65000.0,
        "amount_crypto": 0.1
    }
    
    approved, reason = guard.validate_order(order, mid_price=60000.0)
    assert approved is False
    assert "FAT_FINGER_COLLAR_EXCEEDED" in reason

# ==============================================================================
# 3. Order Management System (OMS) Tests
# ==============================================================================
@pytest.mark.asyncio
async def test_oms_process_approved_order():
    gateway_mock = MagicMock()
    execution_report_mock = {
        "order_id": "SIM-123",
        "symbol": "BTCUSDT",
        "action": "BUY",
        "executed_price": 60000.0,
        "executed_qty_cash": 6000.0,
        "executed_qty_crypto": 0.1,
        "fee_paid": 1.2,
        "status": "FILLED"
    }
    gateway_mock.send_order = AsyncMock(return_value=execution_report_mock)
    
    oms = OrderManagementSystem(gateway=gateway_mock)
    
    order = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "price": 60000.0,
        "notional": 6000.0
    }
    
    res = await oms.process_approved_order(order)
    assert res is not None
    assert res["status"] == "FILLED"
    assert res["order_id"] == "SIM-123"

@pytest.mark.asyncio
async def test_oms_exception_handling():
    gateway_mock = MagicMock()
    # Mock send_order to raise an Exception
    gateway_mock.send_order = AsyncMock(side_effect=Exception("Connection loss"))
    
    oms = OrderManagementSystem(gateway=gateway_mock)
    
    order = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "price": 60000.0,
        "notional": 6000.0
    }
    
    res = await oms.process_approved_order(order)
    assert res is None

# ==============================================================================
# 4. Execution Gateway Paper Simulator Tests
# ==============================================================================
@pytest.mark.asyncio
async def test_execution_gateway_simulation():
    gateway = BinanceExecutionGateway(
        api_key=None,
        api_secret=None,
        paper_trading=True
    )
    
    # Simulate a market BUY order
    fill = await gateway.send_order({
        "symbol": "BTCUSDT",
        "action": "BUY",
        "notional": 6000.0,
        "type": "limit",
        "limit_price": 60000.0
    })
    
    assert fill["status"] == "FILLED"
    assert fill["symbol"] == "BTCUSDT"
    assert fill["action"] == "BUY"
    assert fill["executed_qty_crypto"] > 0
    assert fill["executed_qty_cash"] > 0
    assert "fee_paid" in fill
