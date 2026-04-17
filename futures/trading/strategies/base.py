"""Base strategy interface.

Identical to the equity version - the strategy abstraction is
instrument-agnostic. Signal values (1, -1, 0) mean the same thing
whether we're trading shares or contracts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Trade:
    """A completed trade."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    contracts: int
    pnl: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop", "target", "eod"


class Strategy(ABC):
    """Base class for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'signal' column to DataFrame.

        Signal values:
            1  = go long
           -1  = go short
            0  = no signal / flat

        The signal indicates desired position at that bar.
        The backtest engine handles execution on the NEXT bar.
        """
        ...

    def get_params(self) -> dict:
        """Return strategy parameters for logging."""
        return {}
