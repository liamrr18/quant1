"""Backtesting engine with realistic execution assumptions.

Key assumptions:
- Execution on NEXT bar open after signal (no lookahead)
- Slippage applied to both entry and exit
- Commission applied per share
- Position sizing based on available capital
- Daily P&L tracking for drawdown calculation
- No fractional shares
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy, Trade
from trading.config import (
    INITIAL_CAPITAL, COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE,
    MAX_POSITION_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest output."""
    strategy_name: str
    symbol: str
    params: dict
    trades: list[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_pct: float
    num_trades: int
    avg_bars_held: float
    exposure_pct: float  # % of time in a position
    start_date: str
    end_date: str


def run_backtest(df: pd.DataFrame, strategy: Strategy, symbol: str,
                 initial_capital: float = INITIAL_CAPITAL,
                 stop_loss_pct: float = STOP_LOSS_PCT,
                 take_profit_pct: float = TAKE_PROFIT_PCT,
                 position_pct: float = MAX_POSITION_PCT) -> BacktestResult:
    """Run a single backtest.

    The signal column should already be in df (from strategy.generate_signals).
    Execution happens on the NEXT bar's open after a signal change.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must have 'signal' column")

    df = df.copy().reset_index(drop=True)
    n = len(df)

    # State tracking
    capital = initial_capital
    position = 0        # current shares held (positive=long, negative=short)
    entry_price = 0.0
    entry_time = None
    entry_bar = 0
    trades: list[Trade] = []
    equity = np.full(n, initial_capital)
    bars_in_position = 0

    prev_signal = 0
    current_date = None

    for i in range(1, n):  # Start at 1 since we look at previous signal
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        prev_signal = int(prev_row["signal"])
        exec_price = float(row["open"])  # Execute at this bar's open
        date = row["date"]

        # Force flat at end of day (15:50+)
        eod_close = row["minute_of_day"] >= 15 * 60 + 50

        # Force flat on new day
        new_day = date != current_date
        if new_day:
            current_date = date

        # ── Check exits first ──
        if position != 0:
            exit_reason = None

            # EOD or new day force close
            if eod_close or new_day:
                exit_reason = "eod"

            # Stop loss
            elif position > 0 and exec_price <= entry_price * (1 - stop_loss_pct):
                exit_reason = "stop"
            elif position < 0 and exec_price >= entry_price * (1 + stop_loss_pct):
                exit_reason = "stop"

            # Take profit
            elif position > 0 and exec_price >= entry_price * (1 + take_profit_pct):
                exit_reason = "target"
            elif position < 0 and exec_price <= entry_price * (1 - take_profit_pct):
                exit_reason = "target"

            # Signal says go flat or reverse
            elif prev_signal == 0 and position != 0:
                exit_reason = "signal"
            elif prev_signal != 0 and np.sign(prev_signal) != np.sign(position):
                exit_reason = "signal"

            if exit_reason:
                # Apply slippage to exit
                if position > 0:
                    fill_price = exec_price - SLIPPAGE_PER_SHARE
                else:
                    fill_price = exec_price + SLIPPAGE_PER_SHARE

                shares = abs(position)
                cost = shares * COMMISSION_PER_SHARE

                if position > 0:
                    pnl = (fill_price - entry_price) * shares - cost
                else:
                    pnl = (entry_price - fill_price) * shares - cost

                pnl_pct = pnl / (entry_price * shares) if entry_price > 0 else 0
                capital += pnl

                trades.append(Trade(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=row["dt"],
                    direction="long" if position > 0 else "short",
                    entry_price=entry_price,
                    exit_price=fill_price,
                    shares=shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                ))

                position = 0
                entry_price = 0.0

        # ── Check entries ──
        if position == 0 and not eod_close and prev_signal != 0:
            # Don't enter on first bar of day (gap risk handled by signal)
            if not new_day:
                max_notional = capital * position_pct
                shares = int(max_notional / exec_price)
                if shares > 0:
                    # Apply slippage to entry
                    if prev_signal > 0:
                        entry_price = exec_price + SLIPPAGE_PER_SHARE
                    else:
                        entry_price = exec_price - SLIPPAGE_PER_SHARE

                    entry_cost = shares * COMMISSION_PER_SHARE
                    capital -= entry_cost

                    position = shares if prev_signal > 0 else -shares
                    entry_time = row["dt"]
                    entry_bar = i

        # Track equity
        if position != 0:
            mark = float(row["close"])
            unrealized = (mark - entry_price) * position
            equity[i] = capital + abs(position) * entry_price + unrealized
            bars_in_position += 1
        else:
            equity[i] = capital

    # Close any remaining position at last bar
    if position != 0:
        last = df.iloc[-1]
        fill = float(last["close"]) - SLIPPAGE_PER_SHARE * np.sign(position)
        shares = abs(position)
        if position > 0:
            pnl = (fill - entry_price) * shares
        else:
            pnl = (entry_price - fill) * shares
        capital += pnl
        trades.append(Trade(
            symbol=symbol, entry_time=entry_time, exit_time=last["dt"],
            direction="long" if position > 0 else "short",
            entry_price=entry_price, exit_price=fill, shares=shares,
            pnl=pnl, pnl_pct=pnl / (entry_price * shares) if entry_price > 0 else 0,
            exit_reason="eod",
        ))
        equity[-1] = capital

    # Compute metrics
    equity_series = pd.Series(equity, index=df["dt"])
    metrics = compute_metrics(trades, equity_series, initial_capital, n, bars_in_position)

    return BacktestResult(
        strategy_name=strategy.name,
        symbol=symbol,
        params=strategy.get_params(),
        trades=trades,
        equity_curve=equity_series,
        daily_returns=metrics["daily_returns"],
        total_return=metrics["total_return"],
        annual_return=metrics["annual_return"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        win_rate=metrics["win_rate"],
        profit_factor=metrics["profit_factor"],
        avg_trade_pct=metrics["avg_trade_pct"],
        num_trades=metrics["num_trades"],
        avg_bars_held=metrics["avg_bars_held"],
        exposure_pct=metrics["exposure_pct"],
        start_date=str(df["date"].iloc[0]),
        end_date=str(df["date"].iloc[-1]),
    )


def compute_metrics(trades: list[Trade], equity: pd.Series,
                    initial_capital: float, total_bars: int,
                    bars_in_position: int) -> dict:
    """Compute performance metrics from trade list and equity curve."""
    num_trades = len(trades)
    final_equity = float(equity.iloc[-1])
    total_return = (final_equity / initial_capital) - 1

    # Daily returns for Sharpe
    daily_equity = equity.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    # Annualized return (252 trading days)
    n_days = max(len(daily_equity), 1)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 1 else 0

    # Sharpe ratio (annualized, risk-free = 0)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Trade-level metrics
    if num_trades > 0:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / num_trades

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss

        avg_trade_pct = np.mean([t.pnl_pct for t in trades]) * 100

        # Average holding time in bars
        avg_bars_held = np.mean([
            max(1, (t.exit_time - t.entry_time).total_seconds() / 60)
            for t in trades
        ])
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade_pct = 0
        avg_bars_held = 0

    exposure_pct = (bars_in_position / total_bars * 100) if total_bars > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pct": avg_trade_pct,
        "num_trades": num_trades,
        "avg_bars_held": avg_bars_held,
        "exposure_pct": exposure_pct,
        "daily_returns": daily_returns,
    }
