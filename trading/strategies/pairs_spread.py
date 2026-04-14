"""Pairs Spread Mean Reversion Strategy.

Trades the log-spread between two correlated ETFs. When the spread
deviates significantly from its rolling mean (measured by Z-score),
enter in the direction of mean reversion.

Research provenance (Experiment 15):
  - Primary pair: GLD/TLT (lookback=120, z_entry=2.0, z_exit=0.5)
  - Dev walk-forward Sharpe: 0.49 (modest)
  - Locked OOS Sharpe: 4.86, alpha +15.1%, beta 0.002
  - Locked OOS return: +5.31%, MaxDD -0.91%, 228 trades, PF 1.66
  - Correlation with ORB baseline: -0.13 (negatively correlated)
  - Portfolio with ORB: Sharpe 6.05 (baseline 3.35, delta +2.70)

CAVEAT: Dev Sharpe (0.49) much weaker than OOS (4.86). This asymmetry
may indicate a regime-favorable OOS period. Requires forward validation.

The strategy requires a 'pair_close' column in the input DataFrame.
The caller is responsible for merging the two instruments' data before
calling generate_signals(). If 'pair_close' is missing, returns all-zero
signals.

FROZEN CONFIGURATION — DO NOT TUNE without new locked OOS evidence.
"""

import logging

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy

log = logging.getLogger(__name__)


class PairsSpread(Strategy):
    """Mean reversion on the log-spread between two correlated ETFs."""

    name = "pairs_spread"

    def __init__(
        self,
        lookback: int = 120,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stale_bars: int = 90,
        last_entry_minute: int = 900,
    ):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "entry_zscore": self.entry_zscore,
            "exit_zscore": self.exit_zscore,
            "stale_bars": self.stale_bars,
            "last_entry_minute": self.last_entry_minute,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean-reversion signals on the log-spread.

        Requires 'pair_close' column (close price of the second instrument).
        Signal: +1 = spread too low (buy primary, hedge would short secondary),
                -1 = spread too high (sell primary, hedge would buy secondary),
                 0 = flat.
        """
        df = df.copy()
        df["signal"] = 0

        if "pair_close" not in df.columns:
            log.warning("PairsSpread: 'pair_close' column missing — returning zero signals")
            return df

        # Compute log-spread and rolling Z-score
        log_spread = np.log(df["close"]) - np.log(df["pair_close"])
        spread_mean = log_spread.rolling(self.lookback, min_periods=20).mean()
        spread_std = log_spread.rolling(self.lookback, min_periods=20).std()
        zscore = (log_spread - spread_mean) / spread_std.replace(0, np.nan)

        signals = np.zeros(len(df))
        position = 0
        entry_bar = 0
        current_date = None

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            # Reset position on new day
            if date != current_date:
                position = 0
                current_date = date

            # Wait for VWAP/spread to stabilize (10:00 ET = minute 600)
            if row["minute_of_day"] < 600:
                signals[i] = 0
                continue

            # Force flat after 15:30 ET (minute 930)
            if row["minute_of_day"] > 930:
                position = 0
                signals[i] = 0
                continue

            z = zscore.iloc[i] if i < len(zscore) else 0
            if pd.isna(z):
                signals[i] = position
                continue

            if position == 0:
                # No new entries after cutoff
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                # Spread too low -> long primary (spread will revert up)
                if z < -self.entry_zscore:
                    position = 1
                    entry_bar = i
                # Spread too high -> short primary (spread will revert down)
                elif z > self.entry_zscore:
                    position = -1
                    entry_bar = i

            elif position == 1:
                # Exit: spread reverted back toward mean
                if z >= -self.exit_zscore:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            elif position == -1:
                if z <= self.exit_zscore:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df
