"""Opening Range Breakout Strategy.

Core idea: The first N minutes of trading establish a range. A breakout
above/below that range with volume can signal the day's direction.

Entry: price breaks above OR high (long) or below OR low (short)
Exit: target (range * multiplier), stop (opposite side of range), or EOD

Filters:
  - Range width must be reasonable (not too narrow, not too wide)
  - ATR volatility regime: skip low-vol days (ATR percentile filter)
  - Volume confirmation: breakout bar must have above-average volume
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
                 min_breakout_volume: float = 0.0):
        self.range_minutes = range_minutes
        self.target_multiple = target_multiple
        self.min_range_pct = min_range_pct
        self.max_range_pct = max_range_pct
        self.min_atr_percentile = min_atr_percentile
        self.min_breakout_volume = min_breakout_volume

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
        return d

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "or_high" not in df.columns or "or_low" not in df.columns:
            return df

        has_atr_filter = self.min_atr_percentile > 0 and "atr_percentile" in df.columns
        has_vol_filter = self.min_breakout_volume > 0 and "rel_volume" in df.columns

        or_end = 9 * 60 + 30 + self.range_minutes
        signals = np.zeros(len(df))
        position = 0
        entry_price = 0.0
        or_high = 0.0
        or_low = 0.0
        target = 0.0
        stop = 0.0
        current_date = None
        day_skipped = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            # Reset on new day
            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                or_high = row["or_high"] if pd.notna(row["or_high"]) else 0
                or_low = row["or_low"] if pd.notna(row["or_low"]) else 0

                # ATR volatility regime filter: skip entire day if vol too low
                if has_atr_filter:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_percentile:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            # Skip bars within the opening range period
            if row["minute_of_day"] < or_end:
                signals[i] = 0
                continue

            # Don't trade after 15:30
            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            # Check range validity
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
                # Volume confirmation: require above-average volume on breakout bar
                if has_vol_filter:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_breakout_volume:
                        signals[i] = 0
                        continue

                # Breakout long
                if row["close"] > or_high:
                    position = 1
                    entry_price = row["close"]
                    target = entry_price + range_width * self.target_multiple
                    stop = or_low
                # Breakout short
                elif row["close"] < or_low:
                    position = -1
                    entry_price = row["close"]
                    target = entry_price - range_width * self.target_multiple
                    stop = or_high

            elif position == 1:
                # Exit long: target or stop
                if row["close"] >= target or row["close"] <= stop:
                    position = 0

            elif position == -1:
                # Exit short: target or stop
                if row["close"] <= target or row["close"] >= stop:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df
