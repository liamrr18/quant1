#!/usr/bin/env python3
"""Combined daily report for equity and futures trading systems.

Reads logs from both systems, generates a summary, sends to Discord,
and opens the report in Notepad.

Run manually or via Task Scheduler at 4:15 PM ET daily.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import pytz

ET = pytz.timezone("America/New_York")

# --- Paths ---
FUTURES_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "futures")
EQUITY_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "spy-trader", ".claude", "worktrees", "flamboyant-lewin", "logs"
)

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_DAILY_REPORT", "")


def load_summary(log_dir: str, date_str: str) -> dict | None:
    """Load summary.json from a dated log directory."""
    path = os.path.join(log_dir, date_str, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_trades(log_dir: str, date_str: str) -> list[dict]:
    """Load trades.csv from a dated log directory."""
    path = os.path.join(log_dir, date_str, "trades.csv")
    if not os.path.exists(path):
        return []
    trades = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return trades


def count_wins(trades: list[dict]) -> tuple[int, int]:
    """Count winning and losing trades from trade log."""
    wins = 0
    losses = 0
    for t in trades:
        pnl_str = t.get("pnl", "") or t.get("futures_pnl", "")
        if not pnl_str:
            continue
        try:
            pnl = float(pnl_str)
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
        except (ValueError, TypeError):
            pass
    return wins, losses


def load_backtest_prediction(date_str: str) -> dict | None:
    """Load backtest prediction for comparison (if available).

    Checks for a precomputed daily prediction file from the backtest.
    """
    predictions_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "predictions"
    )
    path = os.path.join(predictions_dir, f"{date_str}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def generate_report(date_str: str = None) -> str:
    """Generate combined daily report text."""
    if date_str is None:
        date_str = datetime.now(ET).strftime("%Y-%m-%d")

    lines = []
    lines.append("=" * 60)
    lines.append(f"  DAILY TRADING REPORT - {date_str}")
    lines.append("=" * 60)
    lines.append("")

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    total_losses = 0

    # --- Equity System ---
    lines.append("-" * 40)
    lines.append("  EQUITY SYSTEM (SPY/QQQ via Alpaca)")
    lines.append("-" * 40)

    equity_summary = load_summary(EQUITY_LOG_DIR, date_str)
    if equity_summary:
        eq_pnl = equity_summary.get("daily_pnl", 0)
        eq_trades = equity_summary.get("trades", 0)
        eq_equity = equity_summary.get("final_equity", 0)
        total_pnl += eq_pnl
        total_trades += eq_trades

        eq_trade_list = load_trades(EQUITY_LOG_DIR, date_str)
        eq_wins, eq_losses = count_wins(eq_trade_list)
        total_wins += eq_wins
        total_losses += eq_losses
        eq_wr = f"{eq_wins/(eq_wins+eq_losses)*100:.0f}%" if (eq_wins + eq_losses) > 0 else "N/A"

        lines.append(f"  P&L:          ${eq_pnl:+,.2f}")
        lines.append(f"  Trades:       {eq_trades}")
        lines.append(f"  Win Rate:     {eq_wr} ({eq_wins}W / {eq_losses}L)")
        lines.append(f"  Final Equity: ${eq_equity:,.2f}")
        lines.append(f"  Halted:       {equity_summary.get('halted', False)}")
    else:
        lines.append("  No data for this date.")

    lines.append("")

    # --- Futures System ---
    lines.append("-" * 40)
    lines.append("  FUTURES SYSTEM (MES/MNQ via IB)")
    lines.append("-" * 40)

    futures_summary = load_summary(FUTURES_LOG_DIR, date_str)
    if futures_summary:
        ft_pnl = futures_summary.get("daily_pnl", 0)
        ft_trades = futures_summary.get("trades", 0)
        ft_equity = futures_summary.get("final_equity", 0)
        total_pnl += ft_pnl
        total_trades += ft_trades

        ft_trade_list = load_trades(FUTURES_LOG_DIR, date_str)
        ft_wins, ft_losses = count_wins(ft_trade_list)
        total_wins += ft_wins
        total_losses += ft_losses
        ft_wr = f"{ft_wins/(ft_wins+ft_losses)*100:.0f}%" if (ft_wins + ft_losses) > 0 else "N/A"

        lines.append(f"  P&L:          ${ft_pnl:+,.2f}")
        lines.append(f"  Trades:       {ft_trades}")
        lines.append(f"  Win Rate:     {ft_wr} ({ft_wins}W / {ft_losses}L)")
        lines.append(f"  Final Equity: ${ft_equity:,.2f}")
        lines.append(f"  Halted:       {futures_summary.get('halted', False)}")
    else:
        lines.append("  No data for this date.")

    lines.append("")

    # --- Combined ---
    lines.append("=" * 60)
    lines.append("  COMBINED TOTALS")
    lines.append("=" * 60)
    total_wr = f"{total_wins/(total_wins+total_losses)*100:.0f}%" if (total_wins + total_losses) > 0 else "N/A"
    lines.append(f"  Total P&L:    ${total_pnl:+,.2f}")
    lines.append(f"  Total Trades: {total_trades}")
    lines.append(f"  Win Rate:     {total_wr} ({total_wins}W / {total_losses}L)")

    # --- Backtest Comparison ---
    bt_pred = load_backtest_prediction(date_str)
    if bt_pred:
        lines.append("")
        lines.append("-" * 40)
        lines.append("  BACKTEST vs ACTUAL")
        lines.append("-" * 40)
        for sym in ["MES", "MNQ"]:
            if sym in bt_pred:
                pred = bt_pred[sym]
                actual_pnl = ft_pnl if futures_summary else 0
                lines.append(f"  {sym}: predicted ${pred.get('pnl', 0):+,.2f}, "
                             f"actual ${actual_pnl:+,.2f}")

    lines.append("")
    lines.append(f"  Report generated: {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S ET')}")
    lines.append("=" * 60)

    return "\n".join(lines)


def send_to_discord(report_text: str):
    """Send report to Discord webhook."""
    try:
        import urllib.request
        # Discord has a 2000 char limit per message
        content = f"```\n{report_text[:1900]}\n```"
        data = json.dumps({"content": content}).encode("utf-8")
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        print("Report sent to Discord.")
    except Exception as e:
        print(f"Discord send failed: {e}")


def open_in_notepad(report_text: str):
    """Save report to temp file and open in Notepad."""
    date_str = datetime.now(ET).strftime("%Y-%m-%d")
    report_path = os.path.join(tempfile.gettempdir(), f"trading_report_{date_str}.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    try:
        subprocess.Popen(["notepad.exe", report_path])
    except Exception as e:
        print(f"Could not open Notepad: {e}")
        print(report_text)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Daily Trading Report")
    parser.add_argument("--date", default=None, help="Date to report (YYYY-MM-DD)")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord send")
    parser.add_argument("--no-open", action="store_true", help="Don't open in Notepad")
    args = parser.parse_args()

    report = generate_report(args.date)
    print(report)

    if not args.no_discord:
        send_to_discord(report)

    if not args.no_open:
        open_in_notepad(report)


if __name__ == "__main__":
    main()
