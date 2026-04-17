"""Configuration for the futures ORB trading system.

Adapts the proven equity ORB strategy (SPY/QQQ) to micro futures (MES/MNQ).
Live trading uses Interactive Brokers (TWS/Gateway) for real futures execution.
Backtesting uses SPY/QQQ cached CSV data as proxy (unchanged).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API credentials (Alpaca - used for backtesting data only) ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# --- Interactive Brokers connection (live/paper trading) ---
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))  # 7497=TWS paper, 4002=Gateway paper
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))
IB_TIMEOUT = int(os.getenv("IB_TIMEOUT", "30"))  # Connection timeout seconds
IB_READONLY = False  # Must be False for order submission

# --- Futures instruments ---
FUTURES_SYMBOLS = ["MES", "MNQ", "MGC"]
PROXY_SYMBOLS = ["SPY", "QQQ"]  # Used for data fetching (MES/MNQ only)

# --- Data paths ---
DATA_DIR = "data"
CACHE_DIR = "data/cache"
# Also check the equity system's cache for shared data
EQUITY_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "spy-trader", ".claude", "worktrees", "flamboyant-lewin", "data", "cache"
)

# --- Backtest parameters ---
INITIAL_CAPITAL = 100_000.0

# --- Futures cost model ---
# Commission: $0.62 round trip per contract (exchange + clearing + NFA)
# Slippage: 1 tick per side (MES: $1.25/tick, MNQ: $0.50/tick)
# These are already defined in contracts.py per instrument

# --- Risk management (futures-specific) ---
MAX_RISK_PER_TRADE_PCT = 0.01    # 1% of account per trade ($1,000 on $100k)
MAX_DAILY_LOSS_PCT = 0.05         # 5% daily loss limit ($5,000 on $100k)
CIRCUIT_BREAKER_PCT = 0.03        # At 3% daily loss, reduce size by 50%
MAX_CONTRACTS_MES = 30            # Hard cap on MES contracts per trade
MAX_CONTRACTS_MNQ = 20            # Hard cap on MNQ contracts per trade
MAX_CONTRACTS_MGC = 5             # Hard cap on MGC contracts per trade (conservative)
MARGIN_SAFETY_MULTIPLE = 2.0      # Require 2x margin available before entering
MAX_CONCURRENT_POSITIONS = 3      # MES + MNQ + MGC max

# --- Risk backstop (hardcoded safety net, not primary exits) ---
STOP_LOSS_PCT = 0.03              # 3% hard stop backstop
TAKE_PROFIT_PCT = 0.03            # 3% hard TP backstop

# --- Trading schedule (US/Eastern) ---
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
TRADE_START = "09:45"   # After opening range forms
TRADE_END = "15:30"     # No new entries after this
FORCE_CLOSE = "15:50"   # Force close all positions

# --- ORB strategy profiles per futures instrument ---
# Same parameters that work on the equity version

ORB_SHARED_DEFAULTS = {
    "range_minutes": 15,
    "target_multiple": 1.5,
    "min_range_pct": 0.001,
    "max_range_pct": 0.008,
    "min_atr_percentile": 25,
    "min_breakout_volume": 1.2,
    "last_entry_minute": 900,   # 15:00 ET
    "stale_exit_bars": 0,
    "min_gap_pct": 0.0,
}

SYMBOL_PROFILES = {
    "MES": {
        # Maps to SPY: use gap filter (skip small-gap days)
        "min_gap_pct": 0.3,
    },
    "MNQ": {
        # Maps to QQQ: use stale exit (cut underwater positions after 90 bars)
        "stale_exit_bars": 90,
    },
}

# --- VWAP Reversion parameters (validated: locked OOS Sharpe 4.08/4.31) ---
VWAP_REVERSION_DEFAULTS = {
    "z_entry": 1.0,
    "max_hold": 90,
    "min_volume": 10,
}

# --- Overnight strategy schedule ---
OVERNIGHT_TRADE_START = "20:00"   # Begin scanning at 8 PM ET
OVERNIGHT_TRADE_END = "02:00"     # No new entries after 2 AM ET
OVERNIGHT_FORCE_CLOSE = "09:25"   # Must be flat before cash ORB starts

# --- Overnight Reversion parameters (validated in Phase 5) ---
OVERNIGHT_REVERSION_DEFAULTS = {
    "z_threshold": 1.5,
    "max_hold_bars": 90,
    "min_volume": 10,
    "warmup_bars": 20,
}

# --- Discord alerts ---
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_TRADE", "")

# --- Email alerts ---
EMAIL_TO = os.getenv("EMAIL_TO", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

# --- Logging ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "futures")
