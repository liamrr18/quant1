"""Overnight Mean Reversion strategy for MNQ.

Fades extended moves from session VWAP during dead hours (8 PM - 2 AM ET).
Enters when price z-score vs VWAP exceeds threshold, exits on reversion
to VWAP or timeout. All positions flat by 9:25 AM.

Validated in Phase 5: walk-forward OOS Sharpe 3.53, 6/6 windows positive,
robust parameter space, low correlation with cash ORB.
"""

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy


class OvernightReversion(Strategy):
    """Mean reversion during overnight dead hours."""

    name = "overnight_reversion"

    def __init__(self, z_threshold: float = 1.5, max_hold_bars: int = 90,
                 min_volume: int = 10, warmup_bars: int = 20):
        self.z_threshold = z_threshold
        self.max_hold_bars = max_hold_bars
        self.min_volume = min_volume
        self.warmup_bars = warmup_bars

    def get_params(self) -> dict:
        return {
            "z_threshold": self.z_threshold,
            "max_hold_bars": self.max_hold_bars,
            "min_volume": self.min_volume,
            "warmup_bars": self.warmup_bars,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean-reversion signals for overnight session.

        Trading window: 8 PM to 2 AM ET (dead hours with mean-reverting behavior).
        Force flat by 9:25 AM if still holding.
        """
        df = df.copy()
        n = len(df)
        signals = np.zeros(n)

        hour = df["dt"].dt.hour.values
        minute = df["dt"].dt.minute.values
        minute_of_day = hour * 60 + minute
        close = df["close"].values
        volume = df["volume"].values
        date = df["dt"].dt.date.values

        # Track state across bars
        position = 0
        entry_bar = 0

        # Running VWAP for overnight window
        cum_vol = 0.0
        cum_vp = 0.0
        deviations = []
        current_date = None

        for i in range(n):
            h = hour[i]
            mod = minute_of_day[i]

            # Reset VWAP at start of overnight window (8 PM)
            if h == 20 and minute[i] == 0:
                cum_vol = 0.0
                cum_vp = 0.0
                deviations = []

            # In trading window: 8 PM to 2 AM
            in_window = (h >= 20) or (h < 2)

            # Force close window: anything after 2 AM until 9:25 AM
            force_close_zone = (2 <= h < 9) or (h == 9 and minute[i] < 25)

            if in_window:
                # Update VWAP
                v = max(volume[i], 1)
                cum_vol += v
                cum_vp += close[i] * v
                vwap = cum_vp / cum_vol if cum_vol > 0 else close[i]

                dev = close[i] - vwap
                deviations.append(dev)

                # Compute z-score
                if len(deviations) >= self.warmup_bars:
                    std = np.std(deviations[-self.warmup_bars:])
                    z = dev / std if std > 0 else 0
                else:
                    z = 0

                if position == 0:
                    if volume[i] >= self.min_volume and len(deviations) >= self.warmup_bars:
                        if z > self.z_threshold:
                            position = -1  # Short: price extended above VWAP
                            entry_bar = i
                        elif z < -self.z_threshold:
                            position = 1  # Long: price extended below VWAP
                            entry_bar = i
                else:
                    bars_held = i - entry_bar

                    # Exit: reversion to VWAP
                    reverted = (position == 1 and z >= 0) or (position == -1 and z <= 0)
                    timed_out = bars_held >= self.max_hold_bars

                    if reverted or timed_out:
                        position = 0

            elif force_close_zone and position != 0:
                # Force close outside trading window
                position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df
