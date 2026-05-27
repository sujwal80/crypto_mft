"""Configuration parameters and core constants for indian_intraday_system."""

import os
from datetime import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# System Execution Mode
# ==============================================================================
USE_SHADOW_MODE = os.getenv("BROKER_ROUTER_CLASS", "SHADOW_ROUTER") == "SHADOW_ROUTER"

# ==============================================================================
# API Credentials
# ==============================================================================
# TrueData credentials
TRUEDATA_USERNAME = os.getenv("TRUEDATA_USERNAME", "placeholder_user")
TRUEDATA_PASSWORD = os.getenv("TRUEDATA_PASSWORD", "placeholder_pass")
TRUEDATA_WS_URL = os.getenv(
    "TRUEDATA_WS_URL", "wss://api.truedata.in/v3/websocket"
)

# Dhan HQ credentials
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "placeholder_client_id")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "placeholder_token")
DHAN_API_BASE_URL = "https://api.dhan.co/v2"

# ==============================================================================
# Market Mechanics & Math Engine Constants
# ==============================================================================
RISK_FREE_RATE = 0.07      # 7% (Indian 10Y Government Bond Yield)
LOT_SIZE_NIFTY = 25        # standard NSE contract size for NIFTY Index Options/Futures
LOT_SIZE_BANKNIFTY = 15    # standard contract size for BANKNIFTY
MIN_VOLATILITY = 0.05      # Volatility floor (5%) to avoid Black-Scholes division by zero

# ==============================================================================
# Time Regimes (IST Timezone)
# ==============================================================================
TIMEZONE = "Asia/Kolkata"
REGIME_LOCK_START = time(9, 15)     # 9:15 AM: Reconnaissance (Settling pre-open, calc walls)
REGIME_ARM_START = time(9, 45)      # 9:45 AM: Mean-Reversion Phase Active
REGIME_MOMENTUM_START = time(13, 30) # 1:30 PM: Momentum Breakout (0DTE Squeeze) Active
REGIME_KILL_START = time(15, 15)    # 3:15 PM: Hard Liquidation Kill-Switch
REGIME_MARKET_CLOSE = time(15, 30)

# ==============================================================================
# Dynamic Strike Window (ATM Options Bandwidth Protection)
# ==============================================================================
DYNAMIC_OPTION_STRIKE_COUNT = 5     # ATM +/- 5 strikes (Total of 11 options strikes monitored)

# ==============================================================================
# Shadow Broker Brutal Realism Models
# ==============================================================================
SHADOW_ROUND_TRIP_FEE = 90.0         # Flat ₹90 transactional tax per lot round-trip
SHADOW_MARKET_SLIPPAGE_POINTS = 0.5  # Market slippage penalty (points per side)
SHADOW_LIMIT_ADVERSE_SELECTION = 0.5 # Price must punch through limit by 0.5 points to fill

# Heartbeat Checks
HEARTBEAT_TIMEOUT_MS = 3000          # 3.0 seconds data silence triggers emergency shutdown

# ==============================================================================
# NSE Tick Size Formatting Utility
# ==============================================================================
def round_to_nse_tick(price: float) -> float:
    """Formats any calculated float price to the exact NSE tick size of 0.05 paise.

    Prevents exchange API rejections.
    """
    return round(float(price) * 20) / 20
