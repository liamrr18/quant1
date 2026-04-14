#!/usr/bin/env python3
"""Pull and cache historical minute data from Alpaca.

Usage:
    python pull_data.py                    # Pull default symbols
    python pull_data.py --symbols SPY QQQ  # Pull specific symbols
    python pull_data.py --months 6         # Pull 6 months
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta

import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.data.provider import get_minute_bars
from trading.config import SYMBOLS

ET = pytz.timezone("America/New_York")


def main():
    parser = argparse.ArgumentParser(description="Pull historical data")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--months", type=int, default=15,
                        help="Months of history to pull")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    end = datetime(2026, 4, 4, tzinfo=ET)
    start = end - timedelta(days=args.months * 30)

    for sym in args.symbols:
        log.info("Pulling %s: %s to %s", sym, start.date(), end.date())
        df = get_minute_bars(sym, start, end, use_cache=True)
        log.info("  Got %d bars covering %d trading days",
                 len(df), df["dt"].dt.date.nunique())

    log.info("Done. Data cached in data/cache/")


if __name__ == "__main__":
    main()
