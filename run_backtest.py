#!/usr/bin/env python3
"""Run a backtest and generate a performance report.

Usage:
    python run_backtest.py                                  # Default: ORB on all symbols
    python run_backtest.py --strategy orb --symbols SPY QQQ
    python run_backtest.py --strategy orb_filtered          # With ATR + volume filters
    python run_backtest.py --walkforward                    # Walk-forward validation
    python run_backtest.py --report                         # Generate HTML report
    python run_backtest.py --compare                        # Before/after comparison report
"""

import argparse
import logging
import sys
import os
from datetime import datetime

import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.vwap_reversion import VWAPReversion
from trading.strategies.orb import ORBBreakout
from trading.strategies.rsi_reversion import RSIReversion
from trading.backtest.engine import run_backtest
from trading.backtest.walkforward import walk_forward, format_results
from trading.config import SYMBOLS

ET = pytz.timezone("America/New_York")

STRATEGIES = {
    "orb": lambda: ORBBreakout(range_minutes=15, target_multiple=1.5),
    "orb_filtered": lambda: ORBBreakout(
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
    ),
    "orb_tight": lambda: ORBBreakout(range_minutes=15, target_multiple=1.0),
    "orb30": lambda: ORBBreakout(range_minutes=30, target_multiple=1.0),
    "vwap": lambda: VWAPReversion(entry_std=2.0, exit_std=0.5),
    "rsi": lambda: RSIReversion(rsi_period=14, oversold=25, overbought=75),
}


def main():
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()),
                        default="orb_filtered")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", default="2025-01-02")
    parser.add_argument("--end", default="2026-04-04")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run walk-forward validation instead of single backtest")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML performance report")
    parser.add_argument("--compare", action="store_true",
                        help="Generate before/after comparison report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=ET)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=ET)

    strat = STRATEGIES[args.strategy]()
    print(f"Strategy: {strat.name} {strat.get_params()}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Symbols: {args.symbols}")

    results = []
    for sym in args.symbols:
        print(f"\n{'='*60}")
        print(f"{sym}")
        print(f"{'='*60}")

        df = get_minute_bars(sym, start, end, use_cache=True)
        df = prepare_features(df)
        df = strat.generate_signals(df)

        if args.walkforward:
            wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)
            print(format_results(wf))
        else:
            r = run_backtest(df, strat, sym)
            results.append(r)
            print(f"Trades:       {r.num_trades}")
            print(f"Total Return: {r.total_return*100:+.2f}%")
            print(f"Sharpe:       {r.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {r.max_drawdown*100:.2f}%")
            print(f"Win Rate:     {r.win_rate*100:.1f}%")
            print(f"Profit Factor:{r.profit_factor:.2f}")
            print(f"Avg Trade:    {r.avg_trade_pct:+.4f}%")
            print(f"Exposure:     {r.exposure_pct:.1f}%")
            print(f"Avg Hold:     {r.avg_bars_held:.0f} minutes")

    # Generate report
    if (args.report or args.compare) and results:
        from trading.reporting.dashboard import generate_report

        baseline = None
        if args.compare:
            print("\nRunning baseline (no filters, 95% position) for comparison...")
            from trading.backtest.engine import run_backtest as _bt
            from trading.config import INITIAL_CAPITAL, STOP_LOSS_PCT, TAKE_PROFIT_PCT
            baseline_strat = ORBBreakout(range_minutes=15, target_multiple=1.5)
            baseline = []
            for sym in args.symbols:
                df = get_minute_bars(sym, start, end, use_cache=True)
                df = prepare_features(df)
                df = baseline_strat.generate_signals(df)
                # Run with old 95% position sizing
                br = _bt(df, baseline_strat, sym, initial_capital=INITIAL_CAPITAL,
                         stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT)
                baseline.append(br)

        title = f"Backtest Report: {strat.name}"
        path = generate_report(results, title=title, comparison=baseline)
        print(f"\nReport saved: {path}")

    # Portfolio summary
    if len(results) > 1 and not args.walkforward:
        total_pnl = sum(r.equity_curve.iloc[-1] - 100_000 for r in results)
        total_trades = sum(r.num_trades for r in results)
        print(f"\n{'='*60}")
        print(f"PORTFOLIO SUMMARY ({len(results)} symbols)")
        print(f"{'='*60}")
        print(f"Total Trades:  {total_trades}")
        print(f"Combined P&L:  ${total_pnl:+,.0f}")
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        print(f"Avg Sharpe:    {avg_sharpe:.2f}")
        all_wins = sum(1 for r in results for t in r.trades if t.pnl > 0)
        all_total = sum(r.num_trades for r in results)
        print(f"Overall WR:    {all_wins/all_total*100:.1f}%" if all_total > 0 else "")


if __name__ == "__main__":
    main()
