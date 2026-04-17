"""Opening Range Breakout Strategy.

Identical to the equity version. The ORB strategy is instrument-agnostic -
it generates signals based on price action relative to the opening range.
The backtest engine handles the futures-specific P&L conversion, position
sizing, and cost modeling.

Core idea: The first N minutes of trading establish a range. A breakout
above/below that range with volume can signal the day's direction.

Entry: price breaks above OR high (long) or below OR low (short)
Exit: target (range * multiplier), stop (opposite side of range), or EOD

Filters:
  - Range width must be reasonable (not too narrow, not too wide)
  - ATR volatility regime: skip low-vol days (ATR percentile filter)
  - Volume confirmation: breakout bar must have above-average volume
  - Gap filter (MES/SPY): skip days where overnight gap < 0.3%
  - Stale exit (MNQ/QQQ): exit underwater positions after 90 bars
"""

import logging

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy

log = logging.getLogger(__name__)


class ORBBreakout(Strategy):
    name = "orb_breakout"

    def __init__(self, range_minutes: int = 15, target_multiple: float = 1.5,
                 min_range_pct: float = 0.001, max_range_pct: float = 0.008,
                 min_atr_percentile: float = 0.0,
                 min_breakout_volume: float = 0.0,
                 last_entry_minute: int = 0,
                 cooldown_bars: int = 0,
                 stale_exit_bars: int = 0,
                 min_gap_pct: float = 0.0,
                 breakeven_trigger: float = 0.0,
                 trail_trigger: float = 0.0,
                 trail_offset: float = 0.5):
        self.range_minutes = range_minutes
        self.target_multiple = target_multiple
        self.min_range_pct = min_range_pct
        self.max_range_pct = max_range_pct
        self.min_atr_percentile = min_atr_percentile
        self.min_breakout_volume = min_breakout_volume
        self.last_entry_minute = last_entry_minute
        self.cooldown_bars = cooldown_bars
        self.stale_exit_bars = stale_exit_bars
        self.min_gap_pct = min_gap_pct
        self.breakeven_trigger = breakeven_trigger
        self.trail_trigger = trail_trigger
        self.trail_offset = trail_offset

    def get_params(self) -> dict:
        d = {
            "range_minutes": self.range_minutes,
            "target_multiple": self.target_multiple,
            "min_range_pct": self.min_range_pct,
            "max_range_pct": self.max_range_pct,
        }
        if self.min_atr_percentile > 0:
            d["min_atr_pctl"] = self.min_atr_percentile
        if self.min_breakout_volume > 0:
            d["min_bo_vol"] = self.min_breakout_volume
        if self.last_entry_minute > 0:
            d["last_entry"] = self.last_entry_minute
        if self.cooldown_bars > 0:
            d["cooldown"] = self.cooldown_bars
        if self.stale_exit_bars > 0:
            d["stale_exit"] = self.stale_exit_bars
        if self.min_gap_pct > 0:
            d["min_gap"] = self.min_gap_pct
        if self.breakeven_trigger > 0:
            d["be_trigger"] = self.breakeven_trigger
        if self.trail_trigger > 0:
            d["trail_at"] = self.trail_trigger
            d["trail_off"] = self.trail_offset
        return d

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "or_high" not in df.columns or "or_low" not in df.columns:
            return df

        has_atr_filter = self.min_atr_percentile > 0 and "atr_percentile" in df.columns
        has_vol_filter = self.min_breakout_volume > 0 and "rel_volume" in df.columns
        has_late_cutoff = self.last_entry_minute > 0
        has_cooldown = self.cooldown_bars > 0
        has_stale_exit = self.stale_exit_bars > 0
        has_gap_filter = self.min_gap_pct > 0 and "gap_pct" in df.columns
        has_breakeven = self.breakeven_trigger > 0
        has_trail = self.trail_trigger > 0

        or_end = 9 * 60 + 30 + self.range_minutes
        signals = np.zeros(len(df))
        position = 0
        entry_price = 0.0
        entry_bar = 0
        or_high = 0.0
        or_low = 0.0
        target = 0.0
        stop = 0.0
        range_width = 0.0
        best_price = 0.0
        current_date = None
        day_skipped = False
        bars_since_exit = 999

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                or_high = row["or_high"] if pd.notna(row["or_high"]) else 0
                or_low = row["or_low"] if pd.notna(row["or_low"]) else 0

                if has_atr_filter:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_percentile:
                        day_skipped = True

                if has_gap_filter and not day_skipped:
                    gap = row.get("gap_pct", 0)
                    if pd.notna(gap) and abs(gap) < self.min_gap_pct:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            if row["minute_of_day"] < or_end:
                signals[i] = 0
                continue

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            if or_high <= 0 or or_low <= 0:
                signals[i] = 0
                continue

            range_width = or_high - or_low
            range_pct = range_width / or_low

            if range_pct < self.min_range_pct or range_pct > self.max_range_pct:
                signals[i] = 0
                position = 0
                continue

            if position == 0:
                bars_since_exit += 1

                if has_cooldown and bars_since_exit < self.cooldown_bars:
                    signals[i] = 0
                    continue

                if has_late_cutoff and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                if has_vol_filter:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_breakout_volume:
                        signals[i] = 0
                        continue

                if row["close"] > or_high:
                    position = 1
                    entry_price = row["close"]
                    entry_bar = i
                    target = entry_price + range_width * self.target_multiple
                    stop = or_low
                    best_price = row["close"]
                elif row["close"] < or_low:
                    position = -1
                    entry_price = row["close"]
                    entry_bar = i
                    target = entry_price - range_width * self.target_multiple
                    stop = or_high
                    best_price = row["close"]

            elif position == 1:
                if row["close"] > best_price:
                    best_price = row["close"]

                if has_trail and range_width > 0:
                    profit = best_price - entry_price
                    if profit >= self.trail_trigger * range_width:
                        trail_stop = best_price - self.trail_offset * range_width
                        if trail_stop > stop:
                            stop = trail_stop
                elif has_breakeven and range_width > 0:
                    profit = best_price - entry_price
                    if profit >= self.breakeven_trigger * range_width:
                        if entry_price > stop:
                            stop = entry_price

                if row["close"] >= target or row["close"] <= stop:
                    position = 0
                    bars_since_exit = 0
                elif has_stale_exit and (i - entry_bar) >= self.stale_exit_bars and row["close"] <= entry_price:
                    position = 0
                    bars_since_exit = 0

            elif position == -1:
                if row["close"] < best_price:
                    best_price = row["close"]

                if has_trail and range_width > 0:
                    profit = entry_price - best_price
                    if profit >= self.trail_trigger * range_width:
                        trail_stop = best_price + self.trail_offset * range_width
                        if trail_stop < stop:
                            stop = trail_stop
                elif has_breakeven and range_width > 0:
                    profit = entry_price - best_price
                    if profit >= self.breakeven_trigger * range_width:
                        if entry_price < stop:
                            stop = entry_price

                if row["close"] <= target or row["close"] >= stop:
                    position = 0
                    bars_since_exit = 0
                elif has_stale_exit and (i - entry_bar) >= self.stale_exit_bars and row["close"] >= entry_price:
                    position = 0
                    bars_since_exit = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df
