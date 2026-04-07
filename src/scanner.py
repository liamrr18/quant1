"""Gap fade signal scanner. Detects qualifying gap setups on SPY."""

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import MAX_GAP_PCT, USE_TREND_FILTER, TREND_MA_PERIOD
from src.data import fetch_daily_bars, fetch_recent_minutes, get_previous_close, get_bar_at_time

log = logging.getLogger(__name__)


@dataclass
class GapSignal:
    date: str
    prev_close: float
    open_price: float
    close_5m: float
    gap_pct: float
    first5_confirms: bool
    trend_ok: bool
    signal: str  # "short", "none"
    reason: str


def compute_trend_filter(symbol: str) -> bool:
    """Check if price is below its 20-day moving average (bearish regime)."""
    daily = fetch_daily_bars(symbol, lookback_days=TREND_MA_PERIOD + 5)
    if len(daily) < TREND_MA_PERIOD:
        log.warning("Not enough daily bars for trend filter (%d < %d)", len(daily), TREND_MA_PERIOD)
        return False

    closes = daily["close"].values
    ma = closes[-TREND_MA_PERIOD:].mean()
    last_close = closes[-1]

    below_ma = last_close < ma
    log.info("Trend filter: last_close=%.2f, MA%d=%.2f, below=%s", last_close, TREND_MA_PERIOD, ma, below_ma)
    return below_ma


def scan(symbol: str, today_str: str) -> GapSignal:
    """Scan for a gap fade signal on the given date.

    Call this after 9:35 ET when the first 5-minute bar has closed.

    Returns a GapSignal with signal="short" if all conditions are met.
    """
    log.info("Scanning %s for gap fade on %s", symbol, today_str)

    minute_df = fetch_recent_minutes(symbol, days=3)

    # Get yesterday's close
    prev_close = get_previous_close(minute_df, today_str)
    if prev_close is None:
        return GapSignal(today_str, 0, 0, 0, 0, False, False, "none", "No previous close found")

    # Get today's open (9:30 bar)
    open_bar = get_bar_at_time(minute_df, today_str, "09:30")
    if open_bar is None:
        return GapSignal(today_str, prev_close, 0, 0, 0, False, False, "none", "No open bar found")
    open_price = open_bar["open"]

    # Get 5-minute bar close (9:34 bar close)
    bar_5m = get_bar_at_time(minute_df, today_str, "09:34")
    if bar_5m is None:
        return GapSignal(today_str, prev_close, open_price, 0, 0, False, False, "none", "No 5m bar found")
    close_5m = bar_5m["close"]

    # Calculate gap
    gap_pct = (open_price / prev_close - 1.0) * 100.0

    # Check: must be a gap UP
    if gap_pct <= 0:
        return GapSignal(today_str, prev_close, open_price, close_5m, gap_pct, False, False,
                         "none", f"Gap is down ({gap_pct:.3f}%), need up")

    # Check: gap must be small (< MAX_GAP_PCT)
    if abs(gap_pct) >= MAX_GAP_PCT:
        return GapSignal(today_str, prev_close, open_price, close_5m, gap_pct, False, False,
                         "none", f"Gap too large ({gap_pct:.3f}% >= {MAX_GAP_PCT}%)")

    # Check: first 5-min bar must NOT confirm the gap (price falls back)
    move_5m = (close_5m / open_price - 1.0) * 100.0
    first5_confirms = move_5m > 0  # confirms = continued up
    if first5_confirms:
        return GapSignal(today_str, prev_close, open_price, close_5m, gap_pct, True, False,
                         "none", f"First 5m confirms gap (5m ret: +{move_5m:.3f}%)")

    # Check: trend filter
    trend_ok = True
    if USE_TREND_FILTER:
        trend_ok = compute_trend_filter(symbol)
        if not trend_ok:
            return GapSignal(today_str, prev_close, open_price, close_5m, gap_pct, False, False,
                             "none", "Trend filter: price above 20-day MA")

    log.info("SIGNAL: Short %s | gap=%.3f%% | 5m_ret=%.3f%% | trend_ok=%s",
             symbol, gap_pct, move_5m, trend_ok)

    return GapSignal(
        date=today_str,
        prev_close=prev_close,
        open_price=open_price,
        close_5m=close_5m,
        gap_pct=gap_pct,
        first5_confirms=False,
        trend_ok=trend_ok,
        signal="short",
        reason=f"Gap up {gap_pct:.3f}%, first 5m rejected, trend OK",
    )
