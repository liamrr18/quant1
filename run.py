#!/usr/bin/env python3
"""
SPY Gap Fade Trader

Runs the gap fade strategy daily:
- Scans for small gap-up setups at 9:35 ET
- Shorts SPY if gap < 0.20% and first 5-min bar rejects the gap
- Covers at noon ET or on stop loss

Usage:
    python run.py              # Run once (today's session)
    python run.py --loop       # Run continuously, trading each market day
    python run.py --dry-run    # Scan for signal without placing orders
"""

import argparse
import logging
import sys
import time
from datetime import datetime

import pytz

from src.config import SYMBOL, LOG_FILE

ET = pytz.timezone("America/New_York")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE),
        ],
    )


def run_once(dry_run: bool = False):
    log = logging.getLogger("run")
    today = datetime.now(ET).strftime("%Y-%m-%d")

    if dry_run:
        from src.scanner import scan
        log.info("=== DRY RUN for %s ===", today)
        signal = scan(SYMBOL, today)
        log.info("Signal: %s", signal.signal)
        log.info("Reason: %s", signal.reason)
        log.info("Gap: %.3f%%", signal.gap_pct)
        log.info("Prev close: %.2f", signal.prev_close)
        log.info("Open: %.2f", signal.open_price)
        log.info("5m close: %.2f", signal.close_5m)
        return

    from src.trader import run_daily
    run_daily()


def run_loop():
    log = logging.getLogger("run")
    log.info("Starting continuous trading loop")

    while True:
        now = datetime.now(ET)

        # Only run on weekdays
        if now.weekday() >= 5:
            log.info("Weekend. Sleeping until Monday.")
            # Sleep until Monday 9:00 ET
            hours_until_monday = (7 - now.weekday()) * 24 - now.hour + 9
            time.sleep(max(hours_until_monday * 3600, 3600))
            continue

        # If before 9:30, wait
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            wait_secs = (market_open - now).total_seconds()
            log.info("Pre-market. Waiting %.0f seconds for open.", wait_secs)
            time.sleep(min(wait_secs, 300))
            continue

        # If after 13:00, done for the day
        if now.hour >= 13:
            log.info("Past noon exit window. Sleeping until tomorrow 9:00 ET.")
            tomorrow_9am = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if now.hour >= 9:
                tomorrow_9am = tomorrow_9am.replace(day=now.day + 1)
            sleep_secs = (tomorrow_9am - now).total_seconds()
            time.sleep(max(sleep_secs, 3600))
            continue

        # Run today's session
        try:
            run_once(dry_run=False)
        except Exception:
            log.exception("Error in daily run")

        # Wait until tomorrow
        log.info("Session complete. Sleeping until next trading day.")
        time.sleep(3600 * 20)  # ~20 hours


def main():
    parser = argparse.ArgumentParser(description="SPY Gap Fade Trader")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, no orders")
    args = parser.parse_args()

    setup_logging()

    if args.loop:
        run_loop()
    else:
        run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
