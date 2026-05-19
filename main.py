import asyncio
import logging
import os
import sys
import time
import json
import gc
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core.schemas import InternalTick
from ingestion.binance_adapter import BinanceCryptoAdapter
from perception.feature_store import FeatureStore
from intelligence.alpha_engine import AlphaModel
from intelligence.portfolio_optimizer import PortfolioOptimizer
from intelligence.order_generator import OrderGenerator
from execution.dead_letter_queue import DeadLetterQueue
from execution.execution_gateway import BinanceExecutionGateway
from execution.risk_guardrails import RiskGuardrailEngine
from execution.oms import OrderManagementSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MFT_Supervisor")

JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_journal.json")
SUCCESS_JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "success_trade_journal.json")
UNSUCCESS_JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsuccess_trade_journal.json")

def write_to_trade_journal(record: Dict):
    """Appends a structured JSON record of every completed execution and net balance step."""
    try:
        # Write to master journal
        with open(JOURNAL_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
            
        # If it is an EXIT record, route to success/unsuccess journals
        if record.get("action", "").startswith("EXIT_"):
            target_file = SUCCESS_JOURNAL_FILE if record.get("trade_pnl", 0.0) > 0.0 else UNSUCCESS_JOURNAL_FILE
            with open(target_file, "a") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to trade journal: {e}")

# ==========================================
# # Event-Driven Supervisor Loop
# ==========================================

async def trading_supervisor_loop(
    queue: asyncio.Queue,
    feature_store: FeatureStore,
    alpha_model: AlphaModel,
    optimizer: PortfolioOptimizer,
    order_generator: OrderGenerator,
    risk_critic: RiskGuardrailEngine,
    oms: OrderManagementSystem,
    initial_portfolio_value: float,
    current_inventory: Dict[str, float],
    reversal_threshold: Optional[float] = None
):
    """Consumes ticks, monitors active bracket limits (TP/SL/Timeout), and triggers automatic exits."""
    logger.info(f"Bracket-Trading Supervisor Loop fully wired. Active Alpha: [{alpha_model.alpha_type}]")

    # Accounting and State variables
    cash_balance = initial_portfolio_value
    crypto_units = 0.0
    portfolio_value = initial_portfolio_value
    current_inventory[list(current_inventory.keys())[0]] = crypto_units

    # Bracket State Machine
    active_position: Optional[Dict] = None # Tracks entry metadata and limits
    cumulative_pnl = 0.0
    ticks_processed = 0

    while True:
        # Phase 1: Ingestion
        tick: InternalTick = await queue.get()
        ticks_processed += 1
        mid_price = (tick.bid + tick.ask) / 2.0
        current_time = int(time.time())

        # Dynamically update aggregate portfolio equity balance
        portfolio_value = cash_balance + (crypto_units * mid_price)
        risk_critic.current_portfolio_value = portfolio_value
        if hasattr(risk_critic, 'daily_peak_value'):
            if portfolio_value > risk_critic.daily_peak_value:
                risk_critic.daily_peak_value = portfolio_value

        logger.debug(f"Balance: ${portfolio_value:.2f} | Cash: ${cash_balance:.2f} | Crypto: {crypto_units:.6f} units")

        # Initialize loop-local proposed order variable to prevent UnboundLocalError
        proposed_order = None

        # A. Real-Time Perception and Intelligence (Extract features and predict signal on every tick)
        features = feature_store.process_tick(tick)
        alpha_forecast = 0.0
        rolling_vol = 0.0
        if features is not None:
            z_score, spread, rolling_imbalance, micro_price_drift, rolling_vol, mid_price = features
            alpha_forecast = alpha_model.predict(features)

        # ==============================================================
        # # A. ACTIVE BRACKET POSITION MONITORING (EXIT CHANNEL)
        # ==============================================================
        if active_position is not None:
            bracket = active_position["bracket"]
            action = active_position["action"]

            # Check Exit Conditions (Take Profit or Stop Loss)
            is_tp_breached = False
            is_sl_breached = False

            if action == "BUY": # Long Position
                is_tp_breached = mid_price >= bracket["take_profit_price"]
                is_sl_breached = mid_price <= bracket["stop_loss_price"]
            else: # Short Position
                is_tp_breached = mid_price <= bracket["take_profit_price"]
                is_sl_breached = mid_price >= bracket["stop_loss_price"]

            # Refinement: Dynamic Signal-Based Reversal Exit
            is_reversal_breached = False
            if reversal_threshold is not None and features is not None:
                if action == "BUY":
                    is_reversal_breached = alpha_forecast <= -reversal_threshold
                else:
                    is_reversal_breached = alpha_forecast >= reversal_threshold

            if is_tp_breached or is_sl_breached or is_reversal_breached:
                # Resolve Exit Rationale
                exit_reason = "SIGNAL_REVERSAL"
                if is_tp_breached:
                    exit_reason = "TAKE_PROFIT"
                elif is_sl_breached:
                    exit_reason = "STOP_LOSS"
                elif is_reversal_breached:
                    exit_reason = "SIGNAL_REVERSAL"

                logger.warning(f"🚨 BRACKET EXIT TRIGGERED -> Reason: {exit_reason} | Mid: {mid_price:.2f}")

                # Route Emergency Market Sell order to OMS to flatten position instantly
                exit_order = {
                    "symbol": tick.symbol,
                    "action": "SELL" if action == "BUY" else "BUY",
                    "quantity": abs(crypto_units),
                    "type": "market",
                    "mid_price": mid_price,
                    "is_emergency": True
                }

                execution_report = await oms.process_approved_order(exit_order)
                if execution_report and execution_report.get("status") == "FILLED":
                    net_cash = execution_report["executed_qty_cash"]
                    executed_exit_price = execution_report["executed_price"]
                    fee_paid = execution_report.get("fee_paid", 0.0)

                    # Calculate realized trade performance
                    entry_price = bracket["entry_price"]
                    trade_pnl = 0.0
                    if action == "BUY":
                        cash_balance += net_cash
                        crypto_units = 0.0
                        # PNL = Proceeds received - Cash spent initially
                        trade_pnl = net_cash - active_position["notional"]
                    else:
                        cash_balance -= net_cash
                        crypto_units = 0.0
                        trade_pnl = active_position["notional"] - net_cash

                    current_inventory[tick.symbol] = crypto_units
                    portfolio_value = cash_balance
                    cumulative_pnl += trade_pnl

                    logger.info(
                        f"✅ BRACKET POSITION FLATTENED -> P&L realized: ${trade_pnl:+.2f}, "
                        f"Portfolio Balance: ${portfolio_value:.2f} | Fee Paid: -${fee_paid:.4f}"
                    )

                    # Write full execution metrics to JSON lines journal
                    write_to_trade_journal({
                        "timestamp": int(time.time()),
                        "symbol": tick.symbol,
                        "action": "EXIT_" + action,
                        "reason": exit_reason,
                        "entry_price": entry_price,
                        "exit_price": executed_exit_price,
                        "trade_pnl": trade_pnl,
                        "cumulative_pnl": cumulative_pnl,
                        "portfolio_value": portfolio_value,
                        "fee_paid": fee_paid
                    })

                    active_position = None # Clear state

                queue.task_done()
                continue
            else:
                # Active position exists but exit boundaries are not breached. Skip entering new positions.
                queue.task_done()
                continue

        # ==============================================================
        # # B. SIGNAL EVALUATION & BRACKET ENTRY
        # ==============================================================
        if active_position is None and features is not None:
            target_weight = optimizer.calculate_target_weight(alpha_forecast)

            # Attempt to generate a new bracket order
            proposed_order = order_generator.generate_bracket_order(
                symbol=tick.symbol,
                target_weight=target_weight,
                portfolio_value=portfolio_value,
                bid=tick.bid,
                ask=tick.ask,
                volatility=rolling_vol / mid_price
            )

        if proposed_order:
            logger.info(f"🔮 STRATEGY ENTRY PROPOSED -> {proposed_order}")

            # Validate through risk engine
            is_approved = risk_critic.validate_order(proposed_order, mid_price)
            if is_approved:
                logger.info("Phase 4 [Critic] Order Approved. Submitting Limit Entry...")
                execution_report = await oms.process_approved_order(proposed_order)

                if execution_report and execution_report.get("status") == "FILLED":
                    action = execution_report["action"]
                    net_crypto = execution_report["executed_qty_crypto"]
                    net_cash = execution_report["executed_qty_cash"]
                    fee_paid = execution_report.get("fee_paid", 0.0)

                    if action == "BUY":
                        cash_balance -= proposed_order["notional"]
                        crypto_units += net_crypto
                    else:
                        crypto_units -= net_crypto
                        cash_balance += net_cash

                    # Sync orchestrator state
                    current_inventory[tick.symbol] = crypto_units

                    # Initialize Active Bracket tracking
                    active_position = proposed_order
                    active_position["bracket"]["entry_price"] = execution_report["executed_price"]

                    logger.info(
                        f"🚀 BRACKET POSITION INITIALIZED -> Entry: {active_position['bracket']['entry_price']:.2f} | "
                        f"TP Target: {active_position['bracket']['take_profit_price']:.2f} | "
                        f"SL Guard: {active_position['bracket']['stop_loss_price']:.2f}"
                    )

                    # Journal Entry
                    write_to_trade_journal({
                        "timestamp": int(time.time()),
                        "symbol": tick.symbol,
                        "action": "ENTRY_" + action,
                        "reason": "STRATEGY SIGNAL",
                        "entry_price": active_position["bracket"]["entry_price"],
                        "exit_price": 0.0,
                        "trade_pnl": 0.0,
                        "cumulative_pnl": cumulative_pnl,
                        "portfolio_value": portfolio_value,
                        "fee_paid": fee_paid
                    })
            else:
                logger.warning("Phase 4 [Critic] Entry Order Rejected. Checked DLQ journal.")

        # Scheduled manual garbage collection when flat to prevent real-time wicks
        if active_position is None and ticks_processed % 10000 == 0:
            logger.debug(f"Supervisor flat. Triggering scheduled manual Garbage Collection (Ticks: {ticks_processed})...")
            gc.collect()

        queue.task_done()

# ==========================================
# # Application Entry Point (With Fail-Safe Auto-Sell)
# ==========================================
async def main():
    logger.info("Initializing Enterprise Crypto MFT System - 100% LIVE TRADING MODE...")
    
    # Disable automatic garbage collection during real-time trading loop
    gc.disable()
    logger.info("Python automatic Garbage Collection disabled. Delegating to Supervisor scheduler.")

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(PROJECT_DIR, "weights.lgb")
    DLQ_PATH = os.path.join(PROJECT_DIR, "dlq_audit.json")

    SYMBOL = "BTCUSDT"
    BINANCE_WSS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth5@100ms"
    BINANCE_REST_URL = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit=5"

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    INITIAL_PORTFOLIO_VALUE = float(os.getenv("PORTFOLIO_CASH_VALUE", "10000.0"))
    PAPER_TRADING = os.getenv("PAPER_TRADING", "False").lower() == "true"

    # CLI Selection parameter: default to KALMAN math filter
    ALPHA_MODEL_TYPE = os.getenv("ALPHA_MODEL_TYPE", "KALMAN")

    if not PAPER_TRADING and (not BINANCE_API_KEY or not BINANCE_API_SECRET):
        logger.critical("⚠️ LIVE TRADING ENABLED but BINANCE_API_KEY or BINANCE_API_SECRET is missing! Halting startup.")
        sys.exit(1)

    message_queue = asyncio.Queue()

    binance_adapter = BinanceCryptoAdapter(symbol=SYMBOL, wss_url=BINANCE_WSS_URL, rest_url=BINANCE_REST_URL)
    feature_store = FeatureStore(window_size=1000)

    # Dynamically configure active forecast model selection (ML, OU, KALMAN, OFI)
    alpha_model = AlphaModel(model_path=WEIGHTS_PATH, alpha_type=ALPHA_MODEL_TYPE)
    optimizer = PortfolioOptimizer()
    TP_MARGIN = float(os.getenv("TP_MARGIN", "0.0005"))
    SL_MARGIN = float(os.getenv("SL_MARGIN", "0.0003"))
    REVERSAL_THRESHOLD = os.getenv("REVERSAL_THRESHOLD")
    REVERSAL_THRESHOLD = float(REVERSAL_THRESHOLD) if REVERSAL_THRESHOLD is not None else None
    
    order_generator = OrderGenerator(tp_margin=TP_MARGIN, sl_margin=SL_MARGIN)

    dlq = DeadLetterQueue(journal_path=DLQ_PATH)
    risk_critic = RiskGuardrailEngine(dlq=dlq, max_drawdown_limit=0.05)
    gateway = BinanceExecutionGateway(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, paper_trading=PAPER_TRADING)
    oms = OrderManagementSystem(gateway=gateway)

    current_inventory: Dict[str, float] = {SYMBOL: 0.0}

    # Initialize clean trade journal files for the run session
    for path in [JOURNAL_FILE, SUCCESS_JOURNAL_FILE, UNSUCCESS_JOURNAL_FILE]:
        try:
            with open(path, "w") as f:
                f.write("") # Overwrite with clean journal
        except Exception:
            pass

    try:
        logger.info(f"Launching supervisor with Alpha Model: [{ALPHA_MODEL_TYPE}]...")
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
                initial_portfolio_value=INITIAL_PORTFOLIO_VALUE,
                current_inventory=current_inventory,
                reversal_threshold=REVERSAL_THRESHOLD
            )
        )
    except asyncio.CancelledError:
        logger.info("Orchestrator received cancel signal (SIGINT/SIGTERM). Initiating fail-safe shutdown...")
    except Exception as e:
        logger.critical(f"CRITICAL UNHANDLED EXCEPTION IN SUPERVISOR: {e}. Initiating emergency liquidation...")
    finally:
        logger.info("Shutting down ingestion adapter and severing WebSocket connections...")
        await binance_adapter.close()

        logger.warning("Checking inventory state for required emergency liquidation...")
        await oms.liquidate_all(current_inventory)

        logger.info("Closing CCXT gateway sessions cleanly...")
        await gateway.close()
        logger.info("Trading system shutdown complete. All positions flattened.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected by OS. Asyncio event loop terminating cleanly.")
