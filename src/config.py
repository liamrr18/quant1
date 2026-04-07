import os
from dotenv import load_dotenv

load_dotenv()

# --- API credentials ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")

# --- Symbol ---
SYMBOL = "SPY"

# --- Gap fade strategy parameters ---
MAX_GAP_PCT = 0.20          # only trade gaps smaller than this (absolute %)
ENTRY_DELAY_MINUTES = 5     # wait 5 min after open to confirm gap rejection
SLIPPAGE_BPS = 2            # assumed slippage per side in basis points

# --- Risk management ---
RISK_PER_TRADE_PCT = 1.0    # max % of equity to risk per trade
MAX_POSITION_PCT = 10.0     # max % of equity in a single position
MAX_DAILY_LOSS_PCT = 2.0    # stop trading if daily loss exceeds this %
STOP_LOSS_PCT = 0.30        # hard stop loss % from entry

# --- Regime filters ---
USE_TREND_FILTER = True     # only short when price < 20-day MA
TREND_MA_PERIOD = 20

# --- Schedule (all times US/Eastern) ---
SCAN_TIME = "09:35"         # check gap signal after first 5-min bar
EXIT_TIME = "12:00"         # close position at noon
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

# --- Paths ---
DATA_DIR = "data"
LOG_FILE = "spy_trader.log"
