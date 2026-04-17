"""Opening Drive Continuation Strategy.

Trades in the direction of the first N-minute price move after market open.
The hypothesis is that the initial drive captures overnight information flow
and institutional positioning that persists through the session.

Structurally different from ORB: trades DIRECTION of initial move, not
breakout from a range. Enters immediately after drive window, not waiting
for a range breakout. One entry per day.

Research provenance (Experiment 15):
  SMH variant (target_multiple=3.0):
    - Dev walk-forward Sharpe: 1.33
    - Locked OOS Sharpe: 3.87, Sortino: 15.05, alpha +17.1%, beta -0.026
    - OOS return: +5.83%, MaxDD -1.41%, 86 trades, PF 1.91
    - Portfolio with ORB: Sharpe 4.72 (delta +1.38)

  XLK variant (target_multiple=1.5):
    - Dev walk-forward Sharpe: 2.23
    - Locked OOS Sharpe: 3.26, Sortino: 9.29, alpha +7.4%, beta 0.006
    - OOS return: +2.55%, MaxDD -0.53%, 84 trades, PF 1.63
    - Portfolio with ORB: Sharpe 4.09 (delta +0.75)

CAVEAT: Trade counts are low (39-86 in ~84 OOS days). Statistical
confidence is limited. Requires forward validation via paper trading.

FROZEN CONFIGURATION — DO NOT TUNE without new locked OOS evidence.
"""

import logging

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy

log = logging.getLogger(__name__)


class OpeningDrive(Strategy):
    """Trade in the direction of the first N-minute move after open."""

    name = "opening_drive"

    def __init__(
        self,
        drive_minutes: int = 5,
        min_drive_pct: float = 0.10,
        target_multiple: float = 2.0,
        stop_multiple: float = 1.0,
        require_gap_align: bool = False,
        min_gap_pct: float = 0.0,
        stale_bars: int = 120,
        last_entry_minute: int = 720,
        min_atr_pctl: float = 0,
    ):
        self.drive_minutes = drive_minutes
        self.min_drive_pct = min_drive_pct
        self.target_multiple = target_multiple
        self.stop_multiple = stop_multiple
        self.require_gap_align = require_gap_align
        self.min_gap_pct = min_gap_pct
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl

    def get_params(self) -> dict:
        d = {
            "drive_minutes": self.drive_minutes,
            "min_drive_pct": self.min_drive_pct,
            "target_multiple": self.target_multiple,
            "stop_multiple": self.stop_multiple,
            "stale_bars": self.stale_bars,
        }
        if self.require_gap_align:
            d["gap_align"] = True
            d["min_gap_pct"] = self.min_gap_pct
        if self.min_atr_pctl > 0:
            d["min_atr_pctl"] = self.min_atr_pctl
        return d

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on the first N-minute price drive.

        Logic:
        1. Track the maximum % move from day's open during the first
           `drive_minutes` minutes (09:30 + drive_minutes).
        2. If the drive exceeds `min_drive_pct`, enter in that direction
           on the first bar after the drive window.
        3. One entry per day. Target/stop based on drive magnitude.
        4. Exit on target, stop, stale timeout, or EOD (15:30).
        """
        df = df.copy()
        df["signal"] = 0

        drive_end_minute = 9 * 60 + 30 + self.drive_minutes
        has_atr_filter = self.min_atr_pctl > 0 and "atr_percentile" in df.columns

        signals = np.zeros(len(df))
        position = 0
        entry_price = 0.0
        entry_bar = 0
        target = 0.0
        stop = 0.0
        current_date = None
        day_skipped = False
        day_open = 0.0
        drive_direction = 0
        drive_size = 0.0
        has_entered = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            # ── New day reset ──
            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                drive_direction = 0
                drive_size = 0.0
                has_entered = False
                day_open = row["open"]

                # ATR volatility regime filter
                if has_atr_filter:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl:
                        day_skipped = True

                # Gap alignment pre-check
                if self.require_gap_align and "gap_pct" in df.columns:
                    gap = row.get("gap_pct", 0)
                    if pd.isna(gap) or abs(gap) < self.min_gap_pct:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            # Force flat after 15:30
            if row["minute_of_day"] > 930:
                position = 0
                signals[i] = 0
                continue

            # ── During drive window: track the move ──
            if row["minute_of_day"] <= drive_end_minute:
                if day_open > 0:
                    move_pct = (row["close"] - day_open) / day_open * 100
                    if abs(move_pct) > abs(drive_size):
                        drive_size = move_pct
                signals[i] = 0
                continue

            # ── First bar after drive window: determine direction ──
            if drive_direction == 0 and not has_entered:
                if drive_size >= self.min_drive_pct:
                    drive_direction = 1  # Bullish drive
                elif drive_size <= -self.min_drive_pct:
                    drive_direction = -1  # Bearish drive
                else:
                    day_skipped = True  # Drive too small, skip day
                    signals[i] = 0
                    continue

                # Gap alignment check
                if self.require_gap_align and "gap_pct" in df.columns:
                    gap = row.get("gap_pct", 0)
                    if pd.notna(gap):
                        if (drive_direction > 0 and gap < 0) or (drive_direction < 0 and gap > 0):
                            day_skipped = True
                            signals[i] = 0
                            continue

            # ── Entry (once per day) ──
            if position == 0 and not has_entered and drive_direction != 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                has_entered = True
                position = drive_direction
                entry_price = row["close"]
                entry_bar = i
                abs_drive = abs(drive_size / 100 * entry_price)

                if position == 1:
                    target = entry_price + abs_drive * self.target_multiple
                    stop = entry_price - abs_drive * self.stop_multiple
                else:
                    target = entry_price - abs_drive * self.target_multiple
                    stop = entry_price + abs_drive * self.stop_multiple

            # ── Exit logic ──
            elif position == 1:
                if row["close"] >= target or row["close"] <= stop:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            elif position == -1:
                if row["close"] <= target or row["close"] >= stop:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df
