#!/usr/bin/env python3
"""End-of-day alert — generates a report and opens it in Notepad.

Run this via Task Scheduler at 4:05 PM ET every weekday.
It pulls all three traders' logs, builds a clean summary,
saves it to the Desktop, and opens it automatically.
"""

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from collections import defaultdict

import pytz

ET = pytz.timezone("America/New_York")
BASE = os.path.dirname(os.path.abspath(__file__))
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

LOG_DIRS = {
    "ORB SPY+QQQ": os.path.join(BASE, "logs"),
    "OpenDrive SMH+XLK": os.path.join(BASE, "logs", "opendrive"),
    "Pairs GLD/TLT": os.path.join(BASE, "logs", "pairs"),
}


def load_summary(log_dir, date_str):
    path = os.path.join(log_dir, date_str, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_trades(log_dir, date_str):
    path = os.path.join(log_dir, date_str, "trades.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_equity(log_dir, date_str):
    path = os.path.join(log_dir, date_str, "equity.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def build_report(date_str):
    lines = []
    lines.append("=" * 60)
    lines.append(f"  DAILY TRADING REPORT  —  {date_str}")
    lines.append("=" * 60)
    lines.append("")

    total_pnl = 0
    total_trades = 0
    total_equity = 0

    for name, log_dir in LOG_DIRS.items():
        summary = load_summary(log_dir, date_str)
        trades = load_trades(log_dir, date_str)
        equity = load_equity(log_dir, date_str)

        completed = [t for t in trades if t.get("pnl") and t["pnl"] != ""]

        lines.append(f"  {name}")
        lines.append(f"  {'-' * (len(name) + 2)}")

        if summary:
            pnl = summary.get("daily_pnl", 0)
            n_trades = summary.get("trades", 0)
            eq = summary.get("final_equity", 0)
            halted = summary.get("halted", False)

            total_pnl += pnl
            total_trades += n_trades
            total_equity = max(total_equity, eq)

            lines.append(f"    P&L today:  ${pnl:+.2f}")
            lines.append(f"    Trades:     {n_trades}")
            lines.append(f"    Equity:     ${eq:,.2f}")
            if halted:
                lines.append(f"    ** HALTED: {summary.get('halt_reason', '')} **")
        elif equity:
            pnl = float(equity[-1].get("daily_pnl", 0)) if equity else 0
            total_pnl += pnl
            lines.append(f"    P&L (live): ${pnl:+.2f}")
            lines.append(f"    (Market may still be open)")
        else:
            lines.append(f"    No data (trader may not have run)")

        if completed:
            lines.append(f"    Trades:")
            for t in completed:
                sym = t.get("symbol", "?")
                direction = t.get("direction", "?")
                pnl_val = t.get("pnl", "0")
                reason = t.get("reason", "")
                try:
                    pnl_f = float(pnl_val)
                    win = "W" if pnl_f > 0 else "L"
                    lines.append(f"      [{win}] {direction:>5} {sym:<5} ${pnl_f:+8.2f}  ({reason})")
                except ValueError:
                    pass

        lines.append("")

    lines.append("=" * 60)
    pnl_emoji = "UP" if total_pnl > 0 else "DOWN" if total_pnl < 0 else "FLAT"
    lines.append(f"  TOTAL P&L:     ${total_pnl:+.2f}  ({pnl_emoji})")
    lines.append(f"  TOTAL TRADES:  {total_trades}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  Backtest expectations (per day):")
    lines.append("    ORB:       ~3 trades, ~$30-50 avg daily P&L")
    lines.append("    OpenDrive: ~2 trades, ~$30-40 avg daily P&L")
    lines.append("    Pairs:     ~3 trades, ~$20-30 avg daily P&L")
    lines.append("    Combined:  ~8 trades, ~$80-120 avg daily P&L")
    lines.append("")
    lines.append("  If actual behavior diverges significantly from")
    lines.append("  expectations over 20+ days, investigate.")
    lines.append("")

    return "\n".join(lines)


def main():
    date_str = datetime.now(ET).strftime("%Y-%m-%d")

    report = build_report(date_str)

    # Save to Desktop
    report_path = os.path.join(DESKTOP, f"trading_report_{date_str}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Also save to logs
    log_report = os.path.join(BASE, "logs", f"report_{date_str}.txt")
    with open(log_report, "w") as f:
        f.write(report)

    # Open in Notepad
    subprocess.Popen(["notepad.exe", report_path])

    print(f"Report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    main()
