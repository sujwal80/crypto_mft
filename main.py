import asyncio
import logging
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

from core.schemas import InternalTick
from ingestion.binance_adapter import BinanceCryptoAdapter
from perception.feature_store import FeatureStore
from intelligence.alpha_engine import AlphaModel, PortfolioOptimizer, OrderGenerator
from execution.risk_critic import DeadLetterQueue, BinanceExecutionGateway, RiskGuardrailEngine, OrderManagementSystem

# Configure UTF-8 encoding for standard streams to prevent Windows crash on emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MFT_Supervisor")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# Event-Driven Supervisor Loop
# ======================================================================
async def trading_supervisor_loop(
    queue: asyncio.Queue,
    feature_store: FeatureStore,
    alpha_model: AlphaModel,
    optimizer: PortfolioOptimizer,
    order_generator: OrderGenerator,
    risk_critic: RiskGuardrailEngine,
    oms: OrderManagementSystem,
    portfolio_value: float
):
    """Consumes normalized ticks, evaluates AI alpha, sizes positions, validates risk, and executes live trades."""
    logger.info("Trading Supervisor Loop fully wired and operational.")
    current_inventory: Dict[str, float] = {"BTCUSDT": 0.0}  # Track holding cash value
    average_entry_price: Dict[str, float] = {"BTCUSDT": 0.0}
    realized_pnl: float = 0.0
    
    try:
        while True:
            # Phase 1: Ingestion (Get Normalized Tick)
            tick: InternalTick = await queue.get()
            logger.debug(f"Supervisor ingested tick: {tick.symbol} @ {tick.bid}x{tick.ask}")
    
            # Phase 2: Perception (Update LOB & Compute Features)
            features = feature_store.process_tick(tick)
            if features is None:
                queue.task_done()
                continue # Warming up feature windows
                
            logger.debug(f"Phase 2 [Perception] Feature Vector computed: {features}")
            
            # Phase 3: Intelligence (Predict Alpha & Size Position)
            alpha_forecast = alpha_model.predict(features)
            target_weight = optimizer.calculate_target_weight(alpha_forecast)
            logger.debug(f"Phase 3 [Intelligence] Alpha Forecast: {alpha_forecast:.6f} | Target Weight: {target_weight:.2%}")
            
            # Generate Rebalancing Orders
            current_holding = current_inventory.get(tick.symbol, 0.0)
            proposed_order = order_generator.generate_order(
                symbol=tick.symbol,
                target_weight=target_weight,
                current_inventory=current_holding,
                portfolio_value=portfolio_value,
                bid=tick.bid,
                ask=tick.ask
            )
            
            if proposed_order:
                mid_price = (tick.bid + tick.ask) / 2.0
                logger.info(f"Phase 3 [Intelligence] Proposed Order -> {proposed_order}")
                
                # Phase 4: Execution & Risk Guardrails (Maker-Critic)
                is_approved = risk_critic.validate_order(proposed_order, mid_price)
                
                if is_approved:
                    logger.info("Phase 4 [Critic] Order Approved by Guardrails. Sending to OMS...")
                    execution_report = await oms.process_approved_order(proposed_order)
                    
                    if execution_report and execution_report.get("status") == "FILLED":
                        # Update Portfolio Inventory State
                        action = execution_report["action"]
                        notional = execution_report["executed_qty"]
                        exec_price = execution_report.get("executed_price", proposed_order.get("limit_price", 0.0))
                        
                        trade_pnl = 0.0
                        
                        trade_notional = notional if action == "BUY" else -notional
                        old_notional = current_inventory[tick.symbol]
                        new_notional = old_notional + trade_notional
                        old_avg = average_entry_price[tick.symbol]
                        
                        if abs(old_notional) < 0.01:
                            # Opening fresh position
                            average_entry_price[tick.symbol] = exec_price
                            trade_pnl = 0.0
                        elif (old_notional > 0 and trade_notional > 0) or (old_notional < 0 and trade_notional < 0):
                            # Adding to existing position (same sign)
                            average_entry_price[tick.symbol] = ((abs(old_notional) * old_avg) + (notional * exec_price)) / abs(new_notional)
                            trade_pnl = 0.0
                        else:
                            # Reducing or flipping position
                            if notional <= abs(old_notional):
                                # Reducing position
                                qty_closed = notional / exec_price if exec_price > 0 else 0.0
                                direction = 1 if old_notional > 0 else -1
                                trade_pnl = (exec_price - old_avg) * qty_closed * direction
                                # average_entry_price stays the same
                            else:
                                # Flipping position
                        
                                qty_closed = abs(old_notional) / exec_price if exec_price > 0 else 0.0
                                direction = 1 if old_notional > 0 else -1
                                trade_pnl = (exec_price - old_avg) * qty_closed * direction
                                
                                # The remaining notional establishes the new entry price
                                average_entry_price[tick.symbol] = exec_price
                        
                        realized_pnl += trade_pnl
                        current_inventory[tick.symbol] = new_notional
                        
                        # Prevent floating point drift around 0
                        if abs(current_inventory[tick.symbol]) < 0.01:
                            current_inventory[tick.symbol] = 0.0
                            average_entry_price[tick.symbol] = 0.0
                                
                        logger.info(f"Portfolio Inventory Updated -> {current_inventory} | Realized PnL: ${realized_pnl:.2f}")
                        
                        journal_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "symbol": tick.symbol,
                            "action": action,
                            "executed_price": exec_price,
                            "executed_notional": notional,
                            "trade_pnl": trade_pnl,
                            "cumulative_pnl": realized_pnl
                        }
                        journal_path = os.path.join(PROJECT_DIR, "trades_journal.json")
                        try:
                            with open(journal_path, "a") as f:
                                f.write(json.dumps(journal_entry) + "\n")
                        except Exception as e:
                            logger.error(f"Failed to write to trades journal: {e}")
                else:
                    logger.warning("Phase 4 [Critic] Order Rejected. Check DLQ audit journal.")
            queue.task_done()
    finally:
        logger.critical("Supervisor loop exiting. Triggering fail-safe auto-sell liquidation...")
        await oms.liquidate_all(current_inventory)
        
# ======================================================================
# Application Entry Point
# ======================================================================
async def main():
    logger.info("Initializing Enterprise Crypto MFT System - 100% LIVE TRADING MODE...")
    
    # Configuration for BTC/USDT on Binance
    SYMBOL = "BTCUSDT"
    BINANCE_WSS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@bookTicker"
    BINANCE_REST_URL = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit=5"
    
    # Load Live Execution Parameters
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    INITIAL_PORTFOLIO_VALUE = float(os.getenv("PORTFOLIO_CASH_VALUE", "10000.0")) # Default $10k bankroll
    
    # Check if user explicitly enabled Paper Trading via environment variable
    PAPER_TRADING = os.getenv("PAPER_TRADING", "false").lower() == "true"
    
    if not PAPER_TRADING and (not BINANCE_API_KEY or not BINANCE_API_SECRET):
        logger.critical("⚠️ LIVE TRADING ENABLED but BINANCE_API_KEY or BINANCE_API_SECRET is missing! Halting startup.")
        sys.exit(1)
        
    # Initialize Core Async Queue
    message_queue = asyncio.Queue()
    
    # Initialize Subsystems across all Domains
    binance_adapter = BinanceCryptoAdapter(symbol=SYMBOL, wss_url=BINANCE_WSS_URL, rest_url=BINANCE_REST_URL)
    feature_store = FeatureStore(window_size=1000)
    # alpha_model = AlphaModel(model_path="/usr/local/google/home/singhujwal/mft_project/weights.lgb")
    alpha_model_path = os.path.join(PROJECT_DIR, "weights.lgb")
    alpha_model = AlphaModel(model_path=alpha_model_path)
    optimizer = PortfolioOptimizer()
    order_generator = OrderGenerator()
    
    # dlq = DeadLetterQueue(journal_path="/usr/local/google/home/singhujwal/mft_project/dlq_audit.json")
    dlq_path = os.path.join(PROJECT_DIR, "dlq_audit.json")
    dlq = DeadLetterQueue(journal_path=dlq_path)
    risk_critic = RiskGuardrailEngine(dlq=dlq, max_drawdown_limit=0.05)
    
    # Initialize Live Gateway
    gateway = BinanceExecutionGateway(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        paper_trading=PAPER_TRADING
    )
    oms = OrderManagementSystem(gateway=gateway)
    
    # Concurrently run Ingestion Adapter and Supervisor Loop
    try:
        await asyncio.gather(
            binance_adapter.connect_and_stream(message_queue),
            trading_supervisor_loop(
                queue=message_queue,
                feature_store=feature_store,
                alpha_model=alpha_model,
                optimizer=optimizer,
                order_generator=order_generator,
                risk_critic=risk_critic,
                oms=oms,
                portfolio_value=INITIAL_PORTFOLIO_VALUE
            )
        )
    except asyncio.CancelledError:
        logger.info("Orchestrator received cancel signal.")
    finally:
        await binance_adapter.close()
        await gateway.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading system shutdown cleanly by user.")

