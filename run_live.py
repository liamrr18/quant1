#!/usr/bin/env python3
"""Start the live trading bot.

Usage:
    python run_live.py                          # Paper trade with default strategy
    python run_live.py --dry-run                # Scan signals only, no orders
    python run_live.py --strategy vwap          # Use VWAP reversion strategy
    python run_live.py --strategy orb           # Use ORB strategy
    python run_live.py --symbols SPY            # Trade only SPY
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.config import SYMBOLS, LOG_FILE
from trading.strategies.vwap_reversion import VWAPReversion
from trading.strategies.orb import ORBBreakout
from trading.strategies.rsi_reversion import RSIReversion
from trading.live.trader import LiveTrader


STRATEGIES = {
    "orb": lambda: ORBBreakout(range_minutes=15, target_multiple=1.5),
    "orb_filtered": lambda: ORBBreakout(
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
    ),
    "orb30": lambda: ORBBreakout(range_minutes=30, target_multiple=1.0),
    "vwap": lambda: VWAPReversion(entry_std=1.5, exit_std=0.3),
    "rsi": lambda: RSIReversion(rsi_period=14, oversold=25, overbought=75),
}


def main():
    parser = argparse.ArgumentParser(description="Live Trading Bot")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()),
                        default="orb_filtered", help="Strategy to run")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan signals only, no actual orders")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE),
        ],
    )

    strategy = STRATEGIES[args.strategy]()
    log = logging.getLogger(__name__)

    log.info("Strategy: %s (%s)", strategy.name, strategy.get_params())
    log.info("Symbols: %s", args.symbols)
    log.info("Mode: %s", "DRY RUN" if args.dry_run else "PAPER TRADING")

    if not args.dry_run:
        from trading.config import ALPACA_PAPER
        if not ALPACA_PAPER:
            log.error("ALPACA_PAPER is not set to true! Set ALPACA_PAPER=true in .env for paper trading.")
            log.error("Refusing to start in live mode. Change .env and restart.")
            sys.exit(1)

    trader = LiveTrader(
        strategy=strategy,
        symbols=args.symbols,
        dry_run=args.dry_run,
    )
    trader.run()


if __name__ == "__main__":
    main()
