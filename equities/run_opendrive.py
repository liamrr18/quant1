#!/usr/bin/env python3
"""Launch SMH+XLK Opening Drive paper trading.

Runs the OpeningDrive strategy on SMH (target 3.0x) and XLK (target 1.5x).
Separate from the SPY+QQQ ORB baseline — logs to logs/opendrive/.

Usage:
    python run_opendrive.py                     # Both SMH and XLK
    python run_opendrive.py --symbols SMH       # SMH only
    python run_opendrive.py --symbols XLK       # XLK only
    python run_opendrive.py --dry-run           # Signal scanning only
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set IB clientId=11 BEFORE broker module loads
import trading.config as _cfg
_cfg.IB_CLIENT_ID = 11

from trading.config import (
    OPENDRIVE_SMH, OPENDRIVE_XLK,
    WAVE2_MAX_POSITION_PCT, WAVE2_MAX_DAILY_LOSS_PCT,
    WAVE2_MAX_CONCURRENT_POSITIONS,
)
from trading.strategies.opening_drive import OpeningDrive
from trading.live.trader import LiveTrader

ET = pytz.timezone("America/New_York")


def main():
    parser = argparse.ArgumentParser(description="SMH+XLK Opening Drive Paper Trader")
    parser.add_argument("--symbols", nargs="+", default=["SMH", "XLK"],
                        choices=["SMH", "XLK"], help="Symbols to trade (default: both)")
    parser.add_argument("--dry-run", action="store_true", help="Signal scanning only, no orders")
    args = parser.parse_args()

    # ── Log directory ──
    log_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "opendrive")
    today = datetime.now(ET).strftime("%Y-%m-%d")
    log_dir = os.path.join(log_base, today)
    os.makedirs(log_dir, exist_ok=True)

    # ── Logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "trader.log")),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    # ── Per-symbol strategies (frozen config) ──
    CONFIGS = {
        "SMH": OPENDRIVE_SMH,
        "XLK": OPENDRIVE_XLK,
    }

    strategies = {}
    for sym in args.symbols:
        cfg = CONFIGS[sym]
        strategies[sym] = OpeningDrive(**cfg)

    log.info("=" * 70)
    log.info("OPENING DRIVE PAPER TRADER")
    log.info("  Symbols: %s", args.symbols)
    log.info("  Mode:    %s", "DRY RUN" if args.dry_run else "PAPER TRADING")
    log.info("  Log dir: %s", log_dir)
    for sym, strat in strategies.items():
        log.info("  %s: %s %s", sym, strat.name, strat.get_params())
    log.info("  FROZEN CONFIG — DO NOT TUNE")
    log.info("=" * 70)

    # ── Launch ──
    trader = LiveTrader(
        strategies=strategies,
        symbols=args.symbols,
        dry_run=args.dry_run,
        log_base_dir=log_base,
    )
    trader.run()


if __name__ == "__main__":
    main()
