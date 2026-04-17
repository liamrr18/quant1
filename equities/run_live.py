#!/usr/bin/env python3
"""Start the live trading bot.

Usage:
    python run_live.py                          # Paper trade with per-symbol ORB profiles
    python run_live.py --dry-run                # Scan signals only, no orders
    python run_live.py --symbols SPY            # Trade only SPY
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set IB clientId=10 BEFORE broker module loads
import trading.config as _cfg
_cfg.IB_CLIENT_ID = 10

from trading.config import SYMBOLS, SYMBOL_PROFILES, ORB_SHARED_DEFAULTS
from trading.strategies.orb import ORBBreakout
from trading.live.trader import LiveTrader

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def make_orb_for_symbol(symbol: str) -> ORBBreakout:
    """Create an ORB strategy with symbol-specific parameters."""
    params = dict(ORB_SHARED_DEFAULTS)
    if symbol in SYMBOL_PROFILES:
        params.update(SYMBOL_PROFILES[symbol])
    return ORBBreakout(**params)


def setup_logging():
    """Configure logging to both console and file."""
    os.makedirs(LOG_DIR, exist_ok=True)

    from datetime import datetime
    import pytz
    date_str = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, date_str, "trader.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return log_file


def main():
    parser = argparse.ArgumentParser(description="ORB Live Trading Bot (SPY+QQQ)")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        help="Symbols to trade (default: from config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan signals only, no actual orders")
    args = parser.parse_args()

    log_file = setup_logging()
    log = logging.getLogger(__name__)

    # Build per-symbol strategies from validated profiles
    strategies = {sym: make_orb_for_symbol(sym) for sym in args.symbols}

    log.info("=" * 70)
    log.info("ORB LIVE TRADER")
    log.info("  Symbols: %s", args.symbols)
    log.info("  Mode:    %s", "DRY RUN (no orders)" if args.dry_run else "PAPER TRADING")
    log.info("  Log:     %s", log_file)
    for sym, strat in strategies.items():
        log.info("  %s profile: %s", sym, strat.get_params())
    log.info("=" * 70)

    trader = LiveTrader(
        strategies=strategies,
        symbols=args.symbols,
        dry_run=args.dry_run,
    )
    trader.run()


if __name__ == "__main__":
    main()
