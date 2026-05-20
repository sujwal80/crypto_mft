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
load_dotenv(override=True)

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

JOURNAL_FILE = None

def write_to_trade_journal(record: Dict):
    """Appends a structured JSON record of every completed execution and net balance step to the master ledger."""
    try:
        with open(JOURNAL_FILE, "a") as f:
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

                    # Write full execution metrics to JSON lines journal including active strategy model key
                    write_to_trade_journal({
                        "timestamp": int(time.time()),
                        "model": alpha_model.alpha_type,
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

                    # Journal Entry including active strategy model key
                    write_to_trade_journal({
                        "timestamp": int(time.time()),
                        "model": alpha_model.alpha_type,
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
    CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")

    # 1. Check for dynamic JSON configuration file
    config_data = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config_data = json.load(f)
            logger.info(f"✅ Dynamically loaded configuration from JSON file: {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Failed to parse JSON configuration {CONFIG_PATH}: {e}. Falling back to environment...")

    # Helper to resolve configuration settings (JSON -> Env -> Default fallback)
    def get_config(section: str, key: str, env_key: str, default_val):
        if section in config_data and key in config_data[section]:
            val = config_data[section][key]
            if val is not None:
                return val
        env_val = os.getenv(env_key)
        if env_val is not None and str(env_val).strip() != "":
            return env_val
        return default_val

    # 2. Map trading setup configuration and load API credentials strictly from environment variables
    SYMBOL = get_config("trading_setup", "symbol", "SYMBOL", "BTCUSDT")
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    
    INITIAL_PORTFOLIO_VALUE = float(get_config("trading_setup", "portfolio_cash_value", "PORTFOLIO_CASH_VALUE", "10000.0"))
    
    paper_trading_val = get_config("trading_setup", "paper_trading", "PAPER_TRADING", "True")
    if isinstance(paper_trading_val, bool):
        PAPER_TRADING = paper_trading_val
    else:
        PAPER_TRADING = str(paper_trading_val).lower() == "true"

    ALPHA_MODEL_TYPE = get_config("trading_setup", "alpha_model_type", "ALPHA_MODEL_TYPE", "MICRO_TREND").upper()

    # Dynamically bind strategy-specific journal filenames inside their own folder
    global JOURNAL_FILE
    journals_dir = os.path.join(PROJECT_DIR, "journals")
    os.makedirs(journals_dir, exist_ok=True)
    
    JOURNAL_FILE = os.path.join(journals_dir, f"trade_journal_{ALPHA_MODEL_TYPE}.json")

    BINANCE_WSS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth5@100ms"
    BINANCE_REST_URL = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit=5"

    if not PAPER_TRADING and (not BINANCE_API_KEY or not BINANCE_API_SECRET):
        logger.critical("⚠️ LIVE TRADING ENABLED but BINANCE_API_KEY or BINANCE_API_SECRET is missing! Halting startup.")
        sys.exit(1)

    message_queue = asyncio.Queue()

    binance_adapter = BinanceCryptoAdapter(symbol=SYMBOL, wss_url=BINANCE_WSS_URL, rest_url=BINANCE_REST_URL)
    feature_store = FeatureStore(window_size=1000)

    # 3. Resolved strategy-level parameter presets (static fallbacks)
    HARDCODED_PRESETS = {
        "MICRO_TREND": {
            "tp_margin": 0.0120,
            "sl_margin": 0.0045,
            "threshold": 0.45,
            "reversal_threshold": None
        },
        "ML": {
            "tp_margin": 0.0060,
            "sl_margin": 0.0030,
            "threshold": 0.0,
            "reversal_threshold": None
        },
        "GEX": {
            "tp_margin": 0.0180,
            "sl_margin": 0.0060,
            "threshold": 0.3,
            "reversal_threshold": None
        },
        "HYBRID": {
            "tp_margin": 0.0180,
            "sl_margin": 0.0060,
            "threshold": 0.3,
            "reversal_threshold": None
        }
    }

    fallback_presets = HARDCODED_PRESETS.get(ALPHA_MODEL_TYPE, HARDCODED_PRESETS["MICRO_TREND"])

    # Helper to resolve dynamic parameter overrides (Env -> JSON strategies block -> Static Preset fallback)
    def resolve_param(json_key: str, env_key: str, preset_val):
        # A. Check explicit environment variables
        env_val = os.getenv(env_key)
        if env_val is not None and env_val.strip() != "":
            try:
                return float(env_val)
            except ValueError:
                logger.warning(f"Could not parse {env_key}='{env_val}' as float. Falling back to JSON configuration...")
        
        # B. Check structured JSON strategy settings
        if "strategies" in config_data and ALPHA_MODEL_TYPE in config_data["strategies"]:
            strategy_block = config_data["strategies"][ALPHA_MODEL_TYPE]
            if json_key in strategy_block and strategy_block[json_key] is not None:
                return float(strategy_block[json_key])
                
        # C. Fallback to default static presets
        return preset_val

    # Resolve pipeline parameters
    THRESHOLD = resolve_param("threshold", "THRESHOLD", fallback_presets["threshold"])
    TP_MARGIN = resolve_param("tp_margin", "TP_MARGIN", fallback_presets["tp_margin"])
    SL_MARGIN = resolve_param("sl_margin", "SL_MARGIN", fallback_presets["sl_margin"])
    
    # Resolve reversal threshold
    REVERSAL_THRESHOLD = fallback_presets["reversal_threshold"]
    if "strategies" in config_data and ALPHA_MODEL_TYPE in config_data["strategies"]:
        strategy_block = config_data["strategies"][ALPHA_MODEL_TYPE]
        if "reversal_threshold" in strategy_block and strategy_block["reversal_threshold"] is not None:
            REVERSAL_THRESHOLD = float(strategy_block["reversal_threshold"])
    else:
        env_reversal = os.getenv("REVERSAL_THRESHOLD")
        if env_reversal is not None and env_reversal.strip() != "":
            try:
                REVERSAL_THRESHOLD = float(env_reversal)
            except ValueError:
                pass

    # Instantiate AlphaModel and other pipeline modules
    alpha_kwargs = {"threshold": THRESHOLD}
    alpha_model = AlphaModel(model_path=WEIGHTS_PATH, alpha_type=ALPHA_MODEL_TYPE, **alpha_kwargs)
    optimizer = PortfolioOptimizer()
    
    logger.info("==========================================================")
    logger.info("🎯 LOADED PARAMETERS FROM CONFIGURATION:")
    logger.info(f"  - ALPHA_MODEL_TYPE  : {ALPHA_MODEL_TYPE}")
    logger.info(f"  - THRESHOLD         : {THRESHOLD}")
    logger.info(f"  - TP_MARGIN         : {TP_MARGIN * 100:.4f}% ({TP_MARGIN})")
    logger.info(f"  - SL_MARGIN         : {SL_MARGIN * 100:.4f}% ({SL_MARGIN})")
    logger.info(f"  - REVERSAL_THRESHOLD: {f'{REVERSAL_THRESHOLD * 100:.4f}% ({REVERSAL_THRESHOLD})' if REVERSAL_THRESHOLD is not None else 'None (Disabled)'}")
    logger.info(f"  - PAPER_TRADING     : {PAPER_TRADING}")
    logger.info(f"  - INITIAL_BALANCE   : ${INITIAL_PORTFOLIO_VALUE:.2f}")
    logger.info("==========================================================")

    order_generator = OrderGenerator(tp_margin=TP_MARGIN, sl_margin=SL_MARGIN)

    dlq = DeadLetterQueue(journal_path=DLQ_PATH)
    risk_critic = RiskGuardrailEngine(dlq=dlq, max_drawdown_limit=0.05)
    gateway = BinanceExecutionGateway(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, paper_trading=PAPER_TRADING)
    oms = OrderManagementSystem(gateway=gateway)

    current_inventory: Dict[str, float] = {SYMBOL: 0.0}

    # Initialize clean trade journal file for the run session
    try:
        with open(JOURNAL_FILE, "w") as f:
            f.write("") # Overwrite with clean journal
    except Exception as e:
        logger.warning(f"Failed to clear master trade journal: {e}")

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
