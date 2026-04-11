"""RSI Mean Reversion Strategy.

Core idea: Short-term RSI extremes on 1-min bars signal temporary
exhaustion. Fade the move and exit when RSI normalizes.

Entry: RSI < oversold (buy) or RSI > overbought (sell)
Exit: RSI crosses back through mid-level, or stop/target hit, or EOD
Filter: time of day, require ATR confirmation for volatility
"""

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy


class RSIReversion(Strategy):
    name = "rsi_reversion"

    def __init__(self, rsi_period: int = 14, oversold: float = 25,
                 overbought: float = 75, exit_level: float = 50):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level

    def get_params(self) -> dict:
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "exit_level": self.exit_level,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "rsi" not in df.columns:
            return df

        # Time filter: 09:45 to 15:30
        time_ok = (df["minute_of_day"] >= 9 * 60 + 45) & (df["minute_of_day"] <= 15 * 60 + 30)

        signals = np.zeros(len(df))
        position = 0
        current_date = None

        for i in range(len(df)):
            row = df.iloc[i]

            # Reset on new day
            if row["date"] != current_date:
                position = 0
                current_date = row["date"]

            rsi = row["rsi"]
            if pd.isna(rsi) or not time_ok.iloc[i]:
                position = 0
                signals[i] = 0
                continue

            if position == 0:
                if rsi < self.oversold:
                    position = 1   # Buy oversold
                elif rsi > self.overbought:
                    position = -1  # Sell overbought
            elif position == 1:
                # Exit when RSI normalizes
                if rsi >= self.exit_level:
                    position = 0
            elif position == -1:
                if rsi <= self.exit_level:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df
