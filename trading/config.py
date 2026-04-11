"""Central configuration for the trading system."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Credentials ──
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# ── Universe ──
# Liquid ETFs with tight spreads, suitable for intraday
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

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
STOP_LOSS_PCT = 0.003         # 0.3% hard stop loss
TAKE_PROFIT_PCT = 0.006       # 0.6% take profit (2:1 R:R target)

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
