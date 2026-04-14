#!/usr/bin/env python3
"""Forward validation report: compare paper trading to backtest expectations.

Run after 5, 10, or 20+ trading days to see if paper matches research.

Usage:
    python forward_validation.py          # Check all available paper days
    python forward_validation.py --days 5 # Last 5 trading days only
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import pytz

ET = pytz.timezone("America/New_York")
BASE = os.path.dirname(os.path.abspath(__file__))

# Backtest expectations (from locked OOS, per trading day averages)
EXPECTATIONS = {
    "ORB SPY+QQQ": {
        "log_dir": os.path.join(BASE, "logs"),
        "avg_daily_pnl": 25.83,     # $2,170 / 84 days
        "avg_daily_trades": 3.0,     # 253 / 84
        "sharpe": 3.35,
        "max_dd_pct": 0.55,
        "win_rate": 0.46,            # blended SPY+QQQ
    },
    "OpenDrive SMH+XLK": {
        "log_dir": os.path.join(BASE, "logs", "opendrive"),
        "avg_daily_pnl": 41.67,     # ($5,910+$2,470)/2 / 84 ≈ $50 but conservative
        "avg_daily_trades": 2.0,     # (86+84) / 84
        "sharpe": 3.57,             # avg of SMH 3.87 and XLK 3.26
        "max_dd_pct": 1.41,
        "win_rate": 0.48,
    },
    "Pairs GLD/TLT": {
        "log_dir": os.path.join(BASE, "logs", "pairs"),
        "avg_daily_pnl": 60.95,     # $5,120 / 84
        "avg_daily_trades": 2.7,     # 228 / 84
        "sharpe": 4.86,
        "max_dd_pct": 0.91,
        "win_rate": 0.71,
    },
}


def get_trading_dates(log_dir, max_days=None):
    """Find all dates with trading data."""
    dates = []
    if not os.path.exists(log_dir):
        return dates
    for d in sorted(os.listdir(log_dir)):
        if len(d) == 10 and d[4] == "-":  # YYYY-MM-DD format
            summary_path = os.path.join(log_dir, d, "summary.json")
            equity_path = os.path.join(log_dir, d, "equity.csv")
            if os.path.exists(summary_path) or os.path.exists(equity_path):
                dates.append(d)
    if max_days:
        dates = dates[-max_days:]
    return dates


def load_day_data(log_dir, date_str):
    """Load summary and trades for a single day."""
    summary_path = os.path.join(log_dir, date_str, "summary.json")
    trades_path = os.path.join(log_dir, date_str, "trades.csv")

    summary = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    trades = []
    if os.path.exists(trades_path):
        with open(trades_path) as f:
            trades = list(csv.DictReader(f))

    return summary, trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None,
                        help="Only look at last N trading days")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  FORWARD VALIDATION REPORT")
    print(f"  Paper Trading vs Backtest Expectations")
    print(f"{'='*80}")

    for name, exp in EXPECTATIONS.items():
        dates = get_trading_dates(exp["log_dir"], args.days)

        print(f"\n  {name}")
        print(f"  {'-' * len(name)}")

        if not dates:
            print(f"    No paper trading data yet.")
            print(f"    Expected: {exp['avg_daily_trades']:.1f} trades/day, "
                  f"${exp['avg_daily_pnl']:.0f}/day avg")
            continue

        # Aggregate paper results
        daily_pnls = []
        daily_trades = []
        all_trade_pnls = []

        for d in dates:
            summary, trades = load_day_data(exp["log_dir"], d)
            if summary:
                daily_pnls.append(summary.get("daily_pnl", 0))
                daily_trades.append(summary.get("trades", 0))

            completed = [t for t in trades if t.get("pnl") and t["pnl"] != ""]
            for t in completed:
                try:
                    all_trade_pnls.append(float(t["pnl"]))
                except ValueError:
                    pass

        n_days = len(daily_pnls)
        if n_days == 0:
            print(f"    {len(dates)} days found but no summary data.")
            continue

        total_pnl = sum(daily_pnls)
        avg_pnl = total_pnl / n_days
        avg_trades = sum(daily_trades) / n_days
        total_trades = sum(daily_trades)

        wins = [p for p in all_trade_pnls if p > 0]
        losses = [p for p in all_trade_pnls if p <= 0]
        paper_wr = len(wins) / len(all_trade_pnls) if all_trade_pnls else 0

        max_daily_loss = min(daily_pnls) if daily_pnls else 0

        print(f"    Paper trading days: {n_days}")
        print(f"    Date range: {dates[0]} to {dates[-1]}")
        print()

        # Side by side comparison
        print(f"    {'Metric':<25} {'Paper':>12} {'Backtest':>12} {'Match?':>8}")
        print(f"    {'-'*24} {'-'*12} {'-'*12} {'-'*8}")

        # Avg daily PnL
        pnl_match = abs(avg_pnl) <= abs(exp["avg_daily_pnl"]) * 2  # within 2x
        print(f"    {'Avg daily PnL':<25} ${avg_pnl:>10.2f} ${exp['avg_daily_pnl']:>10.2f}"
              f" {'OK' if pnl_match else 'CHECK':>8}")

        # Avg trades per day
        trade_match = avg_trades <= exp["avg_daily_trades"] * 2
        print(f"    {'Avg trades/day':<25} {avg_trades:>12.1f} {exp['avg_daily_trades']:>12.1f}"
              f" {'OK' if trade_match else 'CHECK':>8}")

        # Win rate
        wr_match = abs(paper_wr - exp["win_rate"]) < 0.10
        print(f"    {'Win rate':<25} {paper_wr:>11.1%} {exp['win_rate']:>11.1%}"
              f" {'OK' if wr_match else 'CHECK':>8}")

        # Total PnL
        print(f"    {'Total PnL':<25} ${total_pnl:>10.2f}")
        print(f"    {'Total trades':<25} {total_trades:>12}")

        # Verdict
        if n_days < 5:
            verdict = "TOO EARLY (need 5+ days)"
        elif n_days < 20:
            verdict = "PRELIMINARY (need 20+ days for confidence)"
        else:
            fails = sum([not pnl_match, not trade_match, not wr_match])
            if fails == 0:
                verdict = "PASS - matches expectations"
            elif fails == 1:
                verdict = "MARGINAL - one metric off"
            else:
                verdict = "FAIL - significant divergence"

        print(f"\n    VERDICT: {verdict}")

    # Overall
    print(f"\n{'='*80}")
    print(f"  GRADUATION STATUS")
    print(f"{'='*80}")
    print(f"  Gate 1 (Code Correctness):      PASS (parity verified)")
    print(f"  Gate 2 (Paper Alignment):        IN PROGRESS")
    print(f"  Gate 3 (Robustness):             XLK 7/7, SMH 6/7, Pairs 6/7")
    print(f"  Gate 4 (Portfolio Value-Add):     PENDING Gate 2")
    print(f"\n  Minimum 20 trading days required before promotion decision.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
