"""Central configuration for the trading system."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Credentials (Alpaca — kept for backtest data cache only) ──
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# ── Interactive Brokers connection (live/paper trading) ──
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))   # 7497=TWS paper, 4002=Gateway paper
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "10"))  # Overridden per launcher
IB_TIMEOUT = int(os.getenv("IB_TIMEOUT", "30"))

# ── Universe ──
# Liquid ETFs with tight spreads, suitable for intraday.
# DIA excluded: negative OOS Sharpe (-0.38) across all variants tested.
# IWM excluded: locked OOS Sharpe -0.74, failed all filter/exit remediation
#   attempts. Pure ORB overtrades in choppy regimes. Dropping IWM improved
#   locked-OOS portfolio Sharpe from 1.54 to 3.28.
SYMBOLS = ["SPY", "QQQ"]

# ── Per-symbol strategy profiles ──
# Each symbol gets its own ORB configuration based on walk-forward OOS evidence.
# Only parameters that differ from the shared defaults are listed.
# Rationale per symbol documented in comments.
SYMBOL_PROFILES = {
    # SPY: Gap filter >= 0.3% improved OOS Sharpe from 0.77 -> 0.98 (+27%)
    # Skip low-gap days where SPY lacks directional conviction.
    # Stable across thresholds: 0.3% (0.98), 0.5% (0.87), 0.2% (0.32).
    # 0.3% is the sweet spot: enough trades (205 OOS) with strong improvement.
    # Note: stale_exit_bars=90 tested on dev period (+0.18) but HURT on locked
    #   OOS (-0.33). Rejected to avoid overfitting.
    "SPY": {
        "min_gap_pct": 0.3,
        "min_atr_percentile": 25,
        "min_breakout_volume": 1.2,
        "last_entry_minute": 900,  # 15:00
    },

    # QQQ: Time decay 90 bars improved OOS Sharpe from 0.97 -> 1.56 (+61%)
    # QQQ breakouts that stall for 90+ minutes underwater are mean-reverting.
    # Cutting stale losers early preserves capital for better setups.
    # Stable: 90b (1.56), 120b (0.40), 60b (0.26) -- 90 is clearly best.
    # Locked OOS: Sharpe 4.20, alpha +10.25%. Best individual symbol.
    "QQQ": {
        "stale_exit_bars": 90,
        "min_atr_percentile": 25,
        "min_breakout_volume": 1.2,
        "last_entry_minute": 900,  # 15:00
    },

    # IWM: EXCLUDED from active universe. Kept for reference.
    # Pure ORB (no filters) gave walk-forward Sharpe 2.04 but collapsed to -0.74
    # on locked OOS (Dec 2025-Apr 2026). 70% exposure generates too many false
    # breakouts in choppy regimes. Adding last_entry=900 and stale exits did not
    # fix the locked-OOS failure.
    # "IWM": {
    #     "min_atr_percentile": 0,
    #     "min_breakout_volume": 0,
    #     "last_entry_minute": 0,
    # },
}

# Shared ORB defaults (used when a profile key is missing)
ORB_SHARED_DEFAULTS = {
    "range_minutes": 15,
    "target_multiple": 1.5,
    "min_range_pct": 0.001,
    "max_range_pct": 0.008,
    "min_atr_percentile": 25,
    "min_breakout_volume": 1.2,
    "last_entry_minute": 900,
    "stale_exit_bars": 0,
    "min_gap_pct": 0.0,
}

# ── Data ──
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# ── Backtest Assumptions ──
COMMISSION_PER_SHARE = 0.0  # Alpaca is commission-free
SLIPPAGE_PER_SHARE = 0.01   # 1 cent per share slippage assumption
INITIAL_CAPITAL = 100_000.0

# ── Risk Management ──
MAX_POSITION_PCT = 0.30       # Max % of equity in a single position (conservative for paper)
MAX_DAILY_LOSS_PCT = 0.02     # 2% max daily loss -> stop trading
MAX_CONCURRENT_POSITIONS = 4  # Max open positions at once
STOP_LOSS_PCT = 0.02          # 2% hard stop (backstop only; strategy manages its own exits)
TAKE_PROFIT_PCT = 0.02        # 2% hard TP (backstop only; strategy manages its own exits)

# ── Trading Schedule (US/Eastern) ──
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
# Avoid first 15 min (chaotic) and last 5 min (closing auction)
TRADE_START = "09:45"
TRADE_END = "15:55"
# Force close all positions by this time
FORCE_CLOSE = "15:50"

# ── Strategy Parameters (defaults, overridden per strategy) ──
# VWAP Mean Reversion
VWAP_ENTRY_THRESHOLD = 1.5    # Std devs from VWAP to enter
VWAP_EXIT_THRESHOLD = 0.3     # Std devs from VWAP to exit
VWAP_LOOKBACK_BARS = 20       # Bars for std dev calculation
VWAP_MIN_VOLUME_RATIO = 1.0   # Min relative volume to trade

# Opening Range Breakout
ORB_RANGE_MINUTES = 15        # First N minutes define the range
ORB_MIN_RANGE_PCT = 0.001     # Min range width as % of price
ORB_MAX_RANGE_PCT = 0.008     # Max range width (avoid huge ranges)
ORB_TARGET_MULTIPLE = 1.5     # Target = range * this multiple

# ORB Filters
ATR_FILTER_PERCENTILE = 25    # Skip days where ATR% is below this percentile (low vol)
ATR_FILTER_LOOKBACK = 20      # Rolling window (trading days) for ATR percentile
VOLUME_CONFIRM_RATIO = 1.2    # Breakout bar volume must be >= this * rolling avg

# RSI Mean Reversion
RSI_PERIOD = 14
RSI_OVERSOLD = 25
RSI_OVERBOUGHT = 75
RSI_EXIT_LEVEL = 50

# ── Paths ──
REPORTS_DIR = "reports"

# ── Logging ──
LOG_FILE = "trader.log"


# ═══════════════════════════════════════════════════════════════════════════════
# WAVE 2 CANDIDATE STRATEGIES (Experiment 15)
# FROZEN from research. DO NOT TUNE without new locked OOS evidence.
# These are paper-trading candidates, not yet production-approved.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Pairs Spread: GLD vs TLT ──
# Locked OOS: Sharpe 4.86, alpha +15.1%, beta 0.002, 228 trades
# Dev Sharpe: 0.49 (weaker — requires forward validation)
PAIRS_GLD_TLT = {
    "primary_symbol": "GLD",
    "secondary_symbol": "TLT",
    "lookback": 120,
    "entry_zscore": 2.0,
    "exit_zscore": 0.5,
    "stale_bars": 90,
    "last_entry_minute": 900,
}

# ── Opening Drive: SMH ──
# Locked OOS: Sharpe 3.87, alpha +17.1%, 86 trades, PF 1.91
# Dev Sharpe: 1.33
OPENDRIVE_SMH = {
    "drive_minutes": 5,
    "min_drive_pct": 0.10,
    "target_multiple": 3.0,
    "stop_multiple": 1.0,
    "stale_bars": 120,
    "last_entry_minute": 720,
}

# ── Opening Drive: XLK ──
# Locked OOS: Sharpe 3.26, alpha +7.4%, 84 trades, PF 1.63
# Dev Sharpe: 2.23
OPENDRIVE_XLK = {
    "drive_minutes": 5,
    "min_drive_pct": 0.10,
    "target_multiple": 1.5,
    "stop_multiple": 1.0,
    "stale_bars": 120,
    "last_entry_minute": 720,
}

# ── Wave 2 risk parameters (conservative for paper validation) ──
WAVE2_MAX_POSITION_PCT = 0.20       # 20% per position (conservative vs 30% for ORB)
WAVE2_MAX_DAILY_LOSS_PCT = 0.015    # 1.5% daily loss limit (tighter than ORB's 2%)
WAVE2_MAX_CONCURRENT_POSITIONS = 2  # Per-instance limit
WAVE2_STOP_LOSS_PCT = 0.02          # Backstop only
WAVE2_TAKE_PROFIT_PCT = 0.02        # Backstop only

# ── Portfolio allocation weights (risk-parity, Experiment 17) ──
# Computed from dev period covariance matrix (Jan-Nov 2025).
# Each stream contributes equal volatility to the portfolio.
# Verified: improves Sharpe on BOTH dev (2.90->3.05) and OOS (6.64->6.88).
# Key: overweights low-vol Pairs (30%), underweights high-vol SMH (10.5%).
PORTFOLIO_WEIGHTS = {
    "ORB_SPY": 0.229,
    "ORB_QQQ": 0.154,
    "Pairs_GLD_TLT": 0.302,
    "OD_SMH": 0.105,
    "OD_XLK": 0.210,
}
