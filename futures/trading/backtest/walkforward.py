"""Walk-forward validation for futures strategy robustness testing.

Same framework as the equity version (60/20/20 rolling windows).
The strategy is evaluated on each test window independently,
preventing in-sample overfitting from inflating results.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy
from trading.backtest.engine import run_backtest, BacktestResult
from trading.data.contracts import FuturesContract, CONTRACTS

log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results for futures."""
    strategy_name: str
    symbol: str
    params: dict
    num_windows: int
    oos_results: list[BacktestResult]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_dollars: float
    avg_trade_pct: float
    total_trades: int
    avg_trades_per_window: float
    exposure_pct: float
    total_pnl: float
    total_costs: float


def walk_forward(df: pd.DataFrame, strategy: Strategy, futures_symbol: str,
                 train_days: int = 60, test_days: int = 20,
                 step_days: int = 20, contract: FuturesContract = None,
                 initial_capital: float = None) -> WalkForwardResult:
    """Run walk-forward validation on futures data.

    Args:
        df: Full dataset with features computed
        strategy: Strategy instance
        futures_symbol: Futures symbol (MES, MNQ)
        train_days: Trading days for in-sample training window
        test_days: Trading days for out-of-sample test window
        step_days: Days to step forward between windows
        contract: Futures contract spec (auto-resolved if None)
        initial_capital: Starting capital (default from config)
    """
    if contract is None:
        contract = CONTRACTS[futures_symbol]

    from trading.config import INITIAL_CAPITAL
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL

    dates = sorted(df["date"].unique())
    n_dates = len(dates)

    if n_dates < train_days + test_days:
        raise ValueError(f"Not enough data: {n_dates} days < {train_days + test_days} required")

    oos_results = []
    window_start = 0

    while window_start + train_days + test_days <= n_dates:
        train_end = window_start + train_days
        test_end = min(train_end + test_days, n_dates)

        test_dates = set(dates[train_end:test_end])
        test_df = df[df["date"].isin(test_dates)].copy()

        if len(test_df) < 100:
            window_start += step_days
            continue

        test_df = strategy.generate_signals(test_df)
        result = run_backtest(test_df, strategy, futures_symbol,
                              initial_capital=initial_capital,
                              contract=contract)
        oos_results.append(result)

        log.info("Window %d: %s to %s | trades=%d, pnl=$%.0f, return=%.2f%%",
                 len(oos_results),
                 min(test_dates), max(test_dates),
                 result.num_trades, result.total_pnl,
                 result.total_return * 100)

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

    total_pnl = sum(t.pnl for t in all_trades)
    total_costs = sum(r.total_costs for r in oos_results)

    avg_trade_dollars = (total_pnl / total_trades) if total_trades > 0 else 0
    avg_trade_pct = (sum(t.pnl_pct for t in all_trades) / total_trades * 100) if total_trades > 0 else 0
    avg_exposure = sum(r.exposure_pct for r in oos_results) / len(oos_results)

    return WalkForwardResult(
        strategy_name=strategy.name,
        symbol=futures_symbol,
        params=strategy.get_params(),
        num_windows=len(oos_results),
        oos_results=oos_results,
        total_return=total_return,
        sharpe_ratio=avg_sharpe,
        max_drawdown=worst_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_dollars=avg_trade_dollars,
        avg_trade_pct=avg_trade_pct,
        total_trades=total_trades,
        avg_trades_per_window=total_trades / len(oos_results),
        exposure_pct=avg_exposure,
        total_pnl=total_pnl,
        total_costs=total_costs,
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
        f"OOS Total P&L:      ${result.total_pnl:+,.0f}",
        f"OOS Avg Sharpe:     {result.sharpe_ratio:.2f}",
        f"OOS Max Drawdown:   {result.max_drawdown*100:.2f}%",
        f"OOS Win Rate:       {result.win_rate*100:.1f}%",
        f"OOS Profit Factor:  {result.profit_factor:.2f}",
        f"OOS Avg Trade:      ${result.avg_trade_dollars:+,.0f} ({result.avg_trade_pct:+.3f}%)",
        f"OOS Exposure:       {result.exposure_pct:.1f}%",
        f"Total Costs:        ${result.total_costs:,.0f}",
        f"{'='*60}",
    ]
    return "\n".join(lines)
