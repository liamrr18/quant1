"""VWAP Mean Reversion Strategy.

Core idea: When price deviates far from intraday VWAP, fade the move.
VWAP acts as a gravity center for institutional flow. Large deviations
tend to revert, especially during the midday session when momentum fades.

Entry: price crosses below VWAP - threshold*std (long) or above + threshold*std (short)
Exit: price returns near VWAP (within exit_threshold), or stop/target hit, or EOD
Filter: only trade 09:45-15:30, require minimum volume
"""

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy


class VWAPReversion(Strategy):
    name = "vwap_reversion"

    def __init__(self, entry_std: float = 1.5, exit_std: float = 0.3,
                 min_volume_ratio: float = 1.0, lookback: int = 20):
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.min_volume_ratio = min_volume_ratio
        self.lookback = lookback

    def get_params(self) -> dict:
        return {
            "entry_std": self.entry_std,
            "exit_std": self.exit_std,
            "min_volume_ratio": self.min_volume_ratio,
            "lookback": self.lookback,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        # Require VWAP deviation and volume data
        if "vwap_dev" not in df.columns or "rel_volume" not in df.columns:
            return df

        # Time filter: 09:45 to 15:30
        time_ok = (df["minute_of_day"] >= 9 * 60 + 45) & (df["minute_of_day"] <= 15 * 60 + 30)

        # Volume filter
        vol_ok = df["rel_volume"] >= self.min_volume_ratio

        # Entry signals
        long_entry = (df["vwap_dev"] < -self.entry_std) & time_ok & vol_ok
        short_entry = (df["vwap_dev"] > self.entry_std) & time_ok & vol_ok

        # Build position signal: hold until exit condition
        position = 0
        signals = np.zeros(len(df))

        for i in range(len(df)):
            if position == 0:
                if long_entry.iloc[i]:
                    position = 1
                elif short_entry.iloc[i]:
                    position = -1
            elif position == 1:
                # Exit long: VWAP dev returns to near zero, or time filter fails
                if df["vwap_dev"].iloc[i] > -self.exit_std or not time_ok.iloc[i]:
                    position = 0
            elif position == -1:
                # Exit short: VWAP dev returns to near zero, or time filter fails
                if df["vwap_dev"].iloc[i] < self.exit_std or not time_ok.iloc[i]:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df
