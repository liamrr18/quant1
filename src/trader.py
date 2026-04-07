"""Main trading orchestrator. Runs the gap fade strategy on a daily schedule."""

import logging
import time
from datetime import datetime

import pytz

from src.config import SYMBOL, EXIT_TIME, MAX_DAILY_LOSS_PCT, STOP_LOSS_PCT
from src import broker, scanner, risk
from src.data import fetch_recent_minutes, get_bar_at_time

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def wait_until(target_time_str: str):
    """Block until the target time (HH:MM ET) today."""
    while True:
        now = now_et()
        target = now.replace(
            hour=int(target_time_str.split(":")[0]),
            minute=int(target_time_str.split(":")[1]),
            second=0,
            microsecond=0,
        )
        if now >= target:
            return
        remaining = (target - now).total_seconds()
        log.info("Waiting %.0f seconds until %s ET...", remaining, target_time_str)
        time.sleep(min(remaining, 30))


def is_market_day() -> bool:
    """Check if today is a trading day by looking for an open bar."""
    today = now_et().strftime("%Y-%m-%d")
    try:
        minutes = fetch_recent_minutes(SYMBOL, days=1)
        bar = get_bar_at_time(minutes, today, "09:30")
        return bar is not None
    except Exception as e:
        log.warning("Could not check market day: %s", e)
        return False


def run_daily():
    """Execute the full daily trading cycle.

    1. Wait for 9:35 ET
    2. Scan for gap fade signal
    3. If signal fires, enter short
    4. Monitor stop loss until noon
    5. Close position at noon
    """
    today = now_et().strftime("%Y-%m-%d")
    log.info("=== Trading day: %s ===", today)

    # Step 1: Wait for scan time (9:35 ET)
    wait_until("09:35")

    # Check we're not already in a position
    existing = broker.get_position(SYMBOL)
    if existing:
        log.warning("Already have a position in %s: %s", SYMBOL, existing)
        return

    # Step 2: Scan for signal
    signal = scanner.scan(SYMBOL, today)
    log.info("Signal: %s | %s", signal.signal, signal.reason)

    if signal.signal != "short":
        log.info("No trade today. Reason: %s", signal.reason)
        return

    # Step 3: Calculate position size and enter
    equity = broker.get_account_equity()
    shares = risk.calculate_shares(equity, signal.close_5m)

    if shares == 0:
        log.warning("Position sizing returned 0 shares. Skipping.")
        return

    # Check daily loss limit
    daily_pnl_pct = 0.0  # TODO: track across trades if running multiple days
    if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
        log.warning("Daily loss limit reached (%.2f%%). No more trades.", daily_pnl_pct)
        return

    order_id = broker.submit_short(SYMBOL, shares)
    entry_price = signal.close_5m  # approximate entry price
    stop_price = entry_price * (1 + STOP_LOSS_PCT / 100.0)

    log.info(
        "ENTERED SHORT: %d shares @ ~%.2f | stop=%.2f | exit at %s ET",
        shares, entry_price, stop_price, EXIT_TIME,
    )

    # Step 4: Monitor until exit time
    exit_hour, exit_minute = map(int, EXIT_TIME.split(":"))

    while True:
        now = now_et()
        if now.hour > exit_hour or (now.hour == exit_hour and now.minute >= exit_minute):
            break

        # Check stop loss every 30 seconds
        pos = broker.get_position(SYMBOL)
        if pos is None:
            log.info("Position closed externally. Done.")
            return

        current_price = abs(pos["market_value"]) / abs(pos["qty"]) if pos["qty"] != 0 else entry_price
        if risk.check_stop_loss(entry_price, current_price):
            log.warning("STOP LOSS triggered at %.2f. Closing position.", current_price)
            broker.close_position(SYMBOL)
            return

        time.sleep(30)

    # Step 5: Close at exit time
    log.info("Exit time reached (%s ET). Closing position.", EXIT_TIME)
    pos = broker.get_position(SYMBOL)
    if pos:
        broker.close_position(SYMBOL)
        log.info("Position closed. P&L: $%.2f (%.2f%%)", pos["unrealized_pl"], pos["unrealized_plpc"] * 100)
    else:
        log.info("No position to close.")

    log.info("=== Day complete ===")
