import os
import pytest
import asyncio
from execution.dead_letter_queue import DeadLetterQueue
from execution.binance_execution_gateway import BinanceExecutionGateway
from execution.risk_guardrail_engine import RiskGuardrailEngine
from execution.order_management_system import OrderManagementSystem

def test_dead_letter_queue(tmp_path):
    journal_file = tmp_path / "test_dlq.json"
    dlq = DeadLetterQueue(journal_path=str(journal_file))
    
    proposed_order = {"symbol": "BTCUSDT", "action": "BUY", "notional": 1000.0}
    dlq.log_rejection(proposed_order, "Collar limit exceeded")
    
    assert journal_file.exists()
    content = journal_file.read_text()
    assert "Collar limit exceeded" in content
    assert "BTCUSDT" in content

@pytest.mark.asyncio
async def test_binance_execution_gateway_paper():
    gateway = BinanceExecutionGateway(paper_trading=True)
    order_payload = {"symbol": "BTCUSDT", "action": "BUY", "notional": 500.0, "limit_price": 50000.0}
    
    report = await gateway.send_order(order_payload)
    assert report["status"] == "FILLED"
    assert report["symbol"] == "BTCUSDT"
    assert report["executed_price"] == 50000.0
    assert report["executed_qty"] == 500.0
    assert report["order_id"].startswith("PAPER_")
    
    await gateway.close()

def test_risk_guardrail_engine(tmp_path):
    journal_file = tmp_path / "test_dlq.json"
    dlq = DeadLetterQueue(journal_path=str(journal_file))
    
    engine = RiskGuardrailEngine(dlq=dlq, max_drawdown_limit=0.05, initial_portfolio_value=100000.0)
    
    # Test valid order
    proposed_order = {"symbol": "BTCUSDT", "action": "BUY", "notional": 1000.0, "limit_price": 50000.0}
    assert engine.validate_order(proposed_order, 50000.0) is True
    
    # Test price collar reject
    assert engine.validate_order(proposed_order, 40000.0) is False # > 2% deviation
    
    # Test drawdown circuit breaker
    engine.update_portfolio_value(90000.0)  # 10% drawdown from 100k peak
    assert engine.validate_order(proposed_order, 50000.0) is False

@pytest.mark.asyncio
async def test_order_management_system(tmp_path):
    gateway = BinanceExecutionGateway(paper_trading=True)
    oms = OrderManagementSystem(gateway=gateway)
    
    approved_order = {"symbol": "BTCUSDT", "action": "BUY", "notional": 500.0, "limit_price": 50000.0}
    report = await oms.process_approved_order(approved_order)
    
    assert report is not None
    assert report["status"] == "FILLED"
    
    # Test fail-safe liquidate_all with average entry price and PnL calculation
    inventory = {"BTCUSDT": 1000.0}
    avg_price = {"BTCUSDT": 50000.0}
    journal_file = tmp_path / "test_trades_journal.json"
    
    await oms.liquidate_all(inventory, avg_price, str(journal_file))
    
    assert inventory["BTCUSDT"] == 0.0
    assert avg_price["BTCUSDT"] == 0.0
    
    # Check that journal file was written
    assert journal_file.exists()
    content = journal_file.read_text()
    assert "BTCUSDT" in content
    assert "SELL" in content
    
    await gateway.close()

from core.exceptions import InsufficientFundsException, CriticalExecutionException

@pytest.mark.asyncio
async def test_oms_exception_propagation():
    # Mock a gateway that throws insufficient funds
    class MockGatewayInsufficientFunds:
        async def send_order(self, order_payload):
            raise Exception("Account has insufficient balance for requested action.")
            
    oms_funds = OrderManagementSystem(gateway=MockGatewayInsufficientFunds())
    approved_order = {"symbol": "BTCUSDT", "action": "BUY", "notional": 500.0, "limit_price": 50000.0}
    
    with pytest.raises(InsufficientFundsException):
        await oms_funds.process_approved_order(approved_order)
        
    # Mock a gateway that throws invalid API-key
    class MockGatewayPermissionError:
        async def send_order(self, order_payload):
            raise Exception("Invalid API-key, IP, or permissions for action.")
            
    oms_perm = OrderManagementSystem(gateway=MockGatewayPermissionError())
    with pytest.raises(CriticalExecutionException):
        await oms_perm.process_approved_order(approved_order)

