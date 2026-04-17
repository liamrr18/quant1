"""Walk-forward validation for strategy robustness testing.

Splits data into rolling train/test windows. The strategy is evaluated
on each test window independently, preventing in-sample overfitting
from inflating results.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from trading.strategies.base import Strategy
from trading.backtest.engine import run_backtest, BacktestResult

log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""
    strategy_name: str
    symbol: str
    params: dict
    num_windows: int
    oos_results: list[BacktestResult]
    # Aggregated OOS metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_pct: float
    total_trades: int
    avg_trades_per_window: float
    exposure_pct: float


def walk_forward(df: pd.DataFrame, strategy: Strategy, symbol: str,
                 train_days: int = 60, test_days: int = 20,
                 step_days: int = 20) -> WalkForwardResult:
    """Run walk-forward validation.

    Args:
        df: Full dataset with features and signals already computed
        strategy: Strategy instance
        symbol: Ticker symbol
        train_days: Number of trading days for in-sample training window
        test_days: Number of trading days for out-of-sample test window
        step_days: Number of days to step forward between windows
    """
    dates = sorted(df["date"].unique())
    n_dates = len(dates)

    if n_dates < train_days + test_days:
        raise ValueError(f"Not enough data: {n_dates} days < {train_days + test_days} required")

    oos_results = []
    window_start = 0

    while window_start + train_days + test_days <= n_dates:
        train_end = window_start + train_days
        test_end = min(train_end + test_days, n_dates)

        train_dates = set(dates[window_start:train_end])
        test_dates = set(dates[train_end:test_end])

        test_df = df[df["date"].isin(test_dates)].copy()

        if len(test_df) < 100:  # Need minimum bars
            window_start += step_days
            continue

        # Generate signals on test data
        # (Strategy sees current bar features only, no future data)
        test_df = strategy.generate_signals(test_df)

        result = run_backtest(test_df, strategy, symbol)
        oos_results.append(result)

        log.info("Window %d: %s to %s | trades=%d, return=%.2f%%, sharpe=%.2f",
                 len(oos_results),
                 min(test_dates), max(test_dates),
                 result.num_trades, result.total_return * 100, result.sharpe_ratio)

        window_start += step_days

    if not oos_results:
        raise ValueError("No valid walk-forward windows")

    # Aggregate OOS metrics
    all_trades = []
    for r in oos_results:
        all_trades.extend(r.trades)

    total_trades = len(all_trades)
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    # Chain equity curves
    total_return = 1.0
    for r in oos_results:
        total_return *= (1 + r.total_return)
    total_return -= 1

    avg_sharpe = sum(r.sharpe_ratio for r in oos_results) / len(oos_results)
    worst_dd = min(r.max_drawdown for r in oos_results)
    avg_trade_pct = (sum(t.pnl_pct for t in all_trades) / total_trades * 100) if total_trades > 0 else 0
    avg_exposure = sum(r.exposure_pct for r in oos_results) / len(oos_results)

    return WalkForwardResult(
        strategy_name=strategy.name,
        symbol=symbol,
        params=strategy.get_params(),
        num_windows=len(oos_results),
        oos_results=oos_results,
        total_return=total_return,
        sharpe_ratio=avg_sharpe,
        max_drawdown=worst_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pct=avg_trade_pct,
        total_trades=total_trades,
        avg_trades_per_window=total_trades / len(oos_results),
        exposure_pct=avg_exposure,
    )


def format_results(result: WalkForwardResult) -> str:
    """Format walk-forward results as a readable string."""
    lines = [
        f"\n{'='*60}",
        f"Walk-Forward: {result.strategy_name} on {result.symbol}",
        f"{'='*60}",
        f"Params:             {result.params}",
        f"OOS Windows:        {result.num_windows}",
        f"Total OOS Trades:   {result.total_trades}",
        f"Avg Trades/Window:  {result.avg_trades_per_window:.1f}",
        f"",
        f"OOS Total Return:   {result.total_return*100:+.2f}%",
        f"OOS Avg Sharpe:     {result.sharpe_ratio:.2f}",
        f"OOS Max Drawdown:   {result.max_drawdown*100:.2f}%",
        f"OOS Win Rate:       {result.win_rate*100:.1f}%",
        f"OOS Profit Factor:  {result.profit_factor:.2f}",
        f"OOS Avg Trade:      {result.avg_trade_pct:+.3f}%",
        f"OOS Exposure:       {result.exposure_pct:.1f}%",
        f"{'='*60}",
    ]
    return "\n".join(lines)
