#!/usr/bin/env python3
"""Launch GLD/TLT pairs spread paper trading.

Runs the PairsSpread strategy on GLD (primary) vs TLT (secondary).
Separate from the SPY+QQQ ORB baseline — logs to logs/pairs/.

Usage:
    python run_pairs.py            # Paper trading (live orders)
    python run_pairs.py --dry-run  # Signal scanning only (no orders)
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set IB clientId=12 BEFORE broker module loads
import trading.config as _cfg
_cfg.IB_CLIENT_ID = 12

from trading.config import (
    PAIRS_GLD_TLT,
    WAVE2_MAX_POSITION_PCT, WAVE2_MAX_DAILY_LOSS_PCT,
    WAVE2_MAX_CONCURRENT_POSITIONS,
)
from trading.strategies.pairs_spread import PairsSpread
from trading.live.pairs_trader import PairsLiveTrader

ET = pytz.timezone("America/New_York")


def main():
    parser = argparse.ArgumentParser(description="GLD/TLT Pairs Spread Paper Trader")
    parser.add_argument("--dry-run", action="store_true", help="Signal scanning only, no orders")
    args = parser.parse_args()

    # ── Log directory ──
    log_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "pairs")
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

    # ── Strategy (frozen config) ──
    cfg = PAIRS_GLD_TLT
    strategy = PairsSpread(
        lookback=cfg["lookback"],
        entry_zscore=cfg["entry_zscore"],
        exit_zscore=cfg["exit_zscore"],
        stale_bars=cfg["stale_bars"],
        last_entry_minute=cfg["last_entry_minute"],
    )

    log.info("=" * 70)
    log.info("PAIRS SPREAD PAPER TRADER")
    log.info("  Primary:   %s", cfg["primary_symbol"])
    log.info("  Secondary: %s", cfg["secondary_symbol"])
    log.info("  Mode:      %s", "DRY RUN" if args.dry_run else "PAPER TRADING")
    log.info("  Log dir:   %s", log_dir)
    log.info("  Params:    %s", strategy.get_params())
    log.info("  FROZEN CONFIG — DO NOT TUNE")
    log.info("=" * 70)

    # ── Launch ──
    trader = PairsLiveTrader(
        strategy=strategy,
        primary_symbol=cfg["primary_symbol"],
        secondary_symbol=cfg["secondary_symbol"],
        dry_run=args.dry_run,
        log_base_dir=log_base,
    )
    trader.run()


if __name__ == "__main__":
    main()
