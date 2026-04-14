#!/usr/bin/env python3
"""Daily end-of-day report across all trading strategies.

Pulls logs from all three traders, computes combined stats, and prints
a clean side-by-side summary. Run after market close (4 PM ET) or anytime
to see the current day's progress.

Usage:
    python daily_report.py              # Today's report
    python daily_report.py 2026-04-14   # Specific date
    python daily_report.py --week       # Last 5 trading days
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

LOG_DIRS = {
    "ORB SPY+QQQ": os.path.join(BASE, "logs"),
    "OpenDrive SMH+XLK": os.path.join(BASE, "logs", "opendrive"),
    "Pairs GLD/TLT": os.path.join(BASE, "logs", "pairs"),
}


def load_trades(log_dir, date_str):
    path = os.path.join(log_dir, date_str, "trades.csv")
    if not os.path.exists(path):
        return []
    trades = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return trades


def load_summary(log_dir, date_str):
    path = os.path.join(log_dir, date_str, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_equity(log_dir, date_str):
    path = os.path.join(log_dir, date_str, "equity.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def day_report(date_str):
    """Generate report for a single day."""
    print(f"\n{'='*80}")
    print(f"  DAILY TRADING REPORT — {date_str}")
    print(f"{'='*80}")

    total_pnl = 0
    total_trades = 0
    any_data = False

    for name, log_dir in LOG_DIRS.items():
        trades = load_trades(log_dir, date_str)
        summary = load_summary(log_dir, date_str)
        equity = load_equity(log_dir, date_str)

        # Count completed trades (entries that have a matching exit)
        completed = [t for t in trades if t.get("action") == "exit" and t.get("pnl")]
        entries = [t for t in trades if t.get("action") == "entry"]

        if summary:
            any_data = True
            pnl = summary.get("daily_pnl", 0)
            n_trades = summary.get("trades", 0)
            equity_val = summary.get("final_equity", 0)
            halted = summary.get("halted", False)

            total_pnl += pnl
            total_trades += n_trades

            print(f"\n  {name}:")
            print(f"    PnL:      ${pnl:+.2f}")
            print(f"    Trades:   {n_trades}")
            print(f"    Equity:   ${equity_val:,.2f}")
            if halted:
                print(f"    STATUS:   HALTED ({summary.get('halt_reason', 'unknown')})")

            # Show individual trades
            if completed:
                wins = [float(t["pnl"]) for t in completed if float(t.get("pnl", 0)) > 0]
                losses = [float(t["pnl"]) for t in completed if float(t.get("pnl", 0)) <= 0]
                print(f"    Wins:     {len(wins)}  (${sum(wins):+.2f})" if wins else "    Wins:     0")
                print(f"    Losses:   {len(losses)}  (${sum(losses):+.2f})" if losses else "    Losses:   0")

            for t in completed:
                sym = t.get("symbol", "?")
                direction = t.get("direction", "?")
                pnl_val = float(t.get("pnl", 0))
                reason = t.get("reason", "?")
                print(f"      {direction:>5} {sym:<5} ${pnl_val:+8.2f}  ({reason})")

        elif trades or equity:
            any_data = True
            # No summary yet (market still open?) — use equity log
            pnl = 0
            if equity:
                pnl = float(equity[-1].get("daily_pnl", 0))
                total_pnl += pnl

            n_completed = len([t for t in trades if t.get("action") == "exit"])
            total_trades += n_completed

            print(f"\n  {name}:")
            print(f"    PnL:      ${pnl:+.2f} (live — market may still be open)")
            print(f"    Trades:   {n_completed} completed")

            for t in completed:
                sym = t.get("symbol", "?")
                direction = t.get("direction", "?")
                pnl_val = float(t.get("pnl", 0))
                reason = t.get("reason", "?")
                print(f"      {direction:>5} {sym:<5} ${pnl_val:+8.2f}  ({reason})")
        else:
            print(f"\n  {name}:")
            print(f"    No data for {date_str}")

    if any_data:
        print(f"\n  {'—'*40}")
        print(f"  TOTAL PnL:    ${total_pnl:+.2f}")
        print(f"  TOTAL TRADES: {total_trades}")
        print(f"{'='*80}\n")
    else:
        print(f"\n  No trading data found for {date_str}.")
        print(f"  (Traders may not have run, or market was closed.)\n")


def week_report():
    """Generate report for the last 5 trading days."""
    today = datetime.now(ET).date()
    dates = []
    d = today
    while len(dates) < 5:
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d.strftime("%Y-%m-%d"))
        d -= timedelta(days=1)

    dates.reverse()

    print(f"\n{'='*80}")
    print(f"  WEEKLY SUMMARY — {dates[0]} to {dates[-1]}")
    print(f"{'='*80}")

    weekly_totals = defaultdict(float)
    weekly_trades = defaultdict(int)

    for date_str in dates:
        day_total = 0
        day_trades = 0
        for name, log_dir in LOG_DIRS.items():
            summary = load_summary(log_dir, date_str)
            if summary:
                pnl = summary.get("daily_pnl", 0)
                trades = summary.get("trades", 0)
                weekly_totals[name] += pnl
                weekly_trades[name] += trades
                day_total += pnl
                day_trades += trades

        if day_total != 0 or day_trades > 0:
            print(f"  {date_str}:  ${day_total:+8.2f}  ({day_trades} trades)")

    print(f"\n  By strategy:")
    for name in LOG_DIRS:
        if weekly_totals[name] != 0 or weekly_trades[name] > 0:
            print(f"    {name:<22} ${weekly_totals[name]:+8.2f}  ({weekly_trades[name]} trades)")

    grand = sum(weekly_totals.values())
    grand_t = sum(weekly_trades.values())
    print(f"\n  WEEKLY TOTAL: ${grand:+.2f}  ({grand_t} trades)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Daily trading report")
    parser.add_argument("date", nargs="?", default=None,
                        help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--week", action="store_true",
                        help="Show last 5 trading days")
    args = parser.parse_args()

    if args.week:
        week_report()
    else:
        date_str = args.date or datetime.now(ET).strftime("%Y-%m-%d")
        day_report(date_str)


if __name__ == "__main__":
    main()
