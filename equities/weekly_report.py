#!/usr/bin/env python3
"""Weekly comparison dashboard: paper trading vs backtest expectations.

Reads all paper trading logs, aggregates by week, and produces an HTML
report with PnL, trade counts, win rates, and backtest comparison.
Auto-opens in the browser.

Usage:
    python weekly_report.py           # All available data
    python weekly_report.py --weeks 2 # Last 2 weeks only
"""

import argparse
import csv
import json
import os
import sys
import webbrowser
from datetime import datetime, timedelta
from collections import defaultdict

import pytz

ET = pytz.timezone("America/New_York")
BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE, "reports")

STRATEGIES = {
    "ORB SPY+QQQ": {
        "log_dir": os.path.join(BASE, "logs"),
        "expected_daily_pnl": 25.83,
        "expected_daily_trades": 3.0,
        "expected_win_rate": 0.46,
        "backtest_sharpe": 3.35,
    },
    "OpenDrive SMH+XLK": {
        "log_dir": os.path.join(BASE, "logs", "opendrive"),
        "expected_daily_pnl": 49.17,
        "expected_daily_trades": 2.0,
        "expected_win_rate": 0.48,
        "backtest_sharpe": 3.57,
    },
    "Pairs GLD/TLT": {
        "log_dir": os.path.join(BASE, "logs", "pairs"),
        "expected_daily_pnl": 60.95,
        "expected_daily_trades": 2.7,
        "expected_win_rate": 0.71,
        "backtest_sharpe": 4.86,
    },
}


def get_trading_dates(log_dir):
    dates = []
    if not os.path.exists(log_dir):
        return dates
    for d in sorted(os.listdir(log_dir)):
        if len(d) == 10 and d[4] == "-":
            if os.path.exists(os.path.join(log_dir, d, "summary.json")) or \
               os.path.exists(os.path.join(log_dir, d, "trades.csv")):
                dates.append(d)
    return dates


def load_day(log_dir, date_str):
    result = {"pnl": 0, "trades": 0, "wins": 0, "losses": 0, "equity": 0, "halted": False}

    summary_path = os.path.join(log_dir, date_str, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
            result["pnl"] = s.get("daily_pnl", 0)
            result["trades"] = s.get("trades", 0)
            result["equity"] = s.get("final_equity", 0)
            result["halted"] = s.get("halted", False)

    trades_path = os.path.join(log_dir, date_str, "trades.csv")
    if os.path.exists(trades_path):
        with open(trades_path) as f:
            for row in csv.DictReader(f):
                if row.get("pnl") and row["pnl"] != "":
                    try:
                        pnl = float(row["pnl"])
                        if pnl > 0:
                            result["wins"] += 1
                        else:
                            result["losses"] += 1
                    except ValueError:
                        pass

    return result


def date_to_week(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Week starts Monday
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def build_html(weeks_data, all_dates, max_weeks=None):
    """Build HTML report."""

    if max_weeks:
        week_keys = sorted(weeks_data.keys())[-max_weeks:]
    else:
        week_keys = sorted(weeks_data.keys())

    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    total_days = len(all_dates)

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Weekly Trading Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }}
    h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
    h2 {{ color: #8b949e; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th {{ background: #161b22; color: #58a6ff; padding: 10px 12px; text-align: right;
         border: 1px solid #30363d; font-weight: 600; }}
    th:first-child {{ text-align: left; }}
    td {{ padding: 8px 12px; text-align: right; border: 1px solid #30363d; }}
    td:first-child {{ text-align: left; font-weight: 500; }}
    .positive {{ color: #3fb950; }}
    .negative {{ color: #f85149; }}
    .neutral {{ color: #8b949e; }}
    .header-row {{ background: #161b22; }}
    .total-row {{ background: #1c2128; font-weight: bold; border-top: 2px solid #58a6ff; }}
    .match {{ color: #3fb950; }}
    .mismatch {{ color: #f85149; }}
    .warn {{ color: #d29922; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 20px; margin: 15px 0; }}
    .metric {{ display: inline-block; margin: 0 30px 10px 0; }}
    .metric-value {{ font-size: 24px; font-weight: bold; }}
    .metric-label {{ color: #8b949e; font-size: 13px; }}
    .footnote {{ color: #8b949e; font-size: 12px; margin-top: 20px; }}
</style>
</head>
<body>
<h1>Weekly Trading Report</h1>
<p class="neutral">Generated: {now} | Paper trading days: {total_days}</p>
"""

    # ── Summary cards ──
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    for wk in week_keys:
        for strat, data in weeks_data[wk].items():
            total_pnl += data["pnl"]
            total_trades += data["trades"]
            total_wins += data["wins"]
            total_losses += data["losses"]

    total_wr = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0
    pnl_class = "positive" if total_pnl >= 0 else "negative"

    html += f"""
<div class="card">
    <div class="metric">
        <div class="metric-value {pnl_class}">${total_pnl:+,.2f}</div>
        <div class="metric-label">Total P&L</div>
    </div>
    <div class="metric">
        <div class="metric-value">{total_trades}</div>
        <div class="metric-label">Total Trades</div>
    </div>
    <div class="metric">
        <div class="metric-value">{total_wr:.1f}%</div>
        <div class="metric-label">Win Rate</div>
    </div>
    <div class="metric">
        <div class="metric-value">{total_days}</div>
        <div class="metric-label">Trading Days</div>
    </div>
</div>
"""

    # ── Weekly breakdown table ──
    html += "<h2>Weekly Breakdown</h2>"
    html += """<table>
    <tr class="header-row">
        <th>Week</th>
        <th>ORB P&L</th><th>ORB Trades</th>
        <th>OpenDrive P&L</th><th>OD Trades</th>
        <th>Pairs P&L</th><th>Pairs Trades</th>
        <th>Total P&L</th><th>Total Trades</th>
    </tr>"""

    cumulative = 0
    for wk in week_keys:
        wk_data = weeks_data[wk]
        orb = wk_data.get("ORB SPY+QQQ", {"pnl": 0, "trades": 0})
        od = wk_data.get("OpenDrive SMH+XLK", {"pnl": 0, "trades": 0})
        pairs = wk_data.get("Pairs GLD/TLT", {"pnl": 0, "trades": 0})
        wk_total = orb["pnl"] + od["pnl"] + pairs["pnl"]
        wk_trades = orb["trades"] + od["trades"] + pairs["trades"]
        cumulative += wk_total

        def pnl_td(val):
            cls = "positive" if val > 0 else ("negative" if val < 0 else "neutral")
            return f'<td class="{cls}">${val:+,.2f}</td>'

        html += f"""<tr>
            <td>{wk}</td>
            {pnl_td(orb['pnl'])}<td>{orb['trades']}</td>
            {pnl_td(od['pnl'])}<td>{od['trades']}</td>
            {pnl_td(pairs['pnl'])}<td>{pairs['trades']}</td>
            {pnl_td(wk_total)}<td>{wk_trades}</td>
        </tr>"""

    html += "</table>"

    # ── Paper vs Backtest comparison ──
    html += "<h2>Paper Trading vs Backtest Expectations</h2>"
    html += """<table>
    <tr class="header-row">
        <th>Strategy</th>
        <th>Paper Avg Daily P&L</th><th>Backtest Expected</th><th>Match?</th>
        <th>Paper Trades/Day</th><th>Expected Trades/Day</th><th>Match?</th>
        <th>Paper Win Rate</th><th>Expected WR</th><th>Match?</th>
    </tr>"""

    for strat_name, cfg in STRATEGIES.items():
        strat_pnl = 0
        strat_trades = 0
        strat_wins = 0
        strat_losses = 0
        strat_days = 0

        for wk in week_keys:
            wk_data = weeks_data[wk]
            if strat_name in wk_data:
                d = wk_data[strat_name]
                strat_pnl += d["pnl"]
                strat_trades += d["trades"]
                strat_wins += d["wins"]
                strat_losses += d["losses"]
                strat_days += d["days"]

        if strat_days == 0:
            html += f'<tr><td>{strat_name}</td><td colspan="8" class="neutral">No data yet</td></tr>'
            continue

        avg_pnl = strat_pnl / strat_days
        avg_trades = strat_trades / strat_days
        wr = strat_wins / (strat_wins + strat_losses) * 100 if (strat_wins + strat_losses) > 0 else 0

        exp_pnl = cfg["expected_daily_pnl"]
        exp_trades = cfg["expected_daily_trades"]
        exp_wr = cfg["expected_win_rate"] * 100

        # Match checks
        pnl_ok = avg_pnl > -abs(exp_pnl)  # At least not deeply negative
        trades_ok = avg_trades <= exp_trades * 2.5
        wr_ok = abs(wr - exp_wr) < 15

        def match_td(ok):
            if ok:
                return '<td class="match">OK</td>'
            return '<td class="mismatch">CHECK</td>'

        pnl_cls = "positive" if avg_pnl > 0 else "negative"
        html += f"""<tr>
            <td>{strat_name}</td>
            <td class="{pnl_cls}">${avg_pnl:+.2f}</td><td>${exp_pnl:.2f}</td>{match_td(pnl_ok)}
            <td>{avg_trades:.1f}</td><td>{exp_trades:.1f}</td>{match_td(trades_ok)}
            <td>{wr:.1f}%</td><td>{exp_wr:.1f}%</td>{match_td(wr_ok)}
        </tr>"""

    html += "</table>"

    # ── Cumulative equity curve (text-based for simplicity) ──
    html += "<h2>Cumulative P&L by Day</h2>"
    html += '<div class="card"><pre style="font-size: 13px; line-height: 1.4;">'

    cum = 0
    for date_str in sorted(all_dates):
        day_pnl = 0
        for strat_name, cfg in STRATEGIES.items():
            d = load_day(cfg["log_dir"], date_str)
            day_pnl += d["pnl"]
        cum += day_pnl
        bar_len = int(abs(cum) / 10)
        bar = "+" * bar_len if cum >= 0 else "-" * bar_len
        cls = "positive" if cum >= 0 else "negative"
        html += f'<span class="{cls}">{date_str}  ${cum:>+9.2f}  {bar}</span>\n'

    html += "</pre></div>"

    # ── Footnotes ──
    html += f"""
<div class="footnote">
    <p>Backtest expectations are from locked OOS period (Dec 2025 - Apr 2026).</p>
    <p>Strategies need 20+ trading days before any promotion decision.</p>
    <p>Graduation criteria: see GRADUATION_CRITERIA.md</p>
</div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Weekly trading comparison dashboard")
    parser.add_argument("--weeks", type=int, default=None, help="Show last N weeks only")
    args = parser.parse_args()

    # Collect all dates and data
    all_dates = set()
    for strat_name, cfg in STRATEGIES.items():
        dates = get_trading_dates(cfg["log_dir"])
        all_dates.update(dates)

    all_dates = sorted(all_dates)

    if not all_dates:
        print("No paper trading data found yet.")
        return

    # Aggregate by week
    weeks_data = defaultdict(lambda: defaultdict(lambda: {
        "pnl": 0, "trades": 0, "wins": 0, "losses": 0, "days": 0
    }))

    for date_str in all_dates:
        week = date_to_week(date_str)
        for strat_name, cfg in STRATEGIES.items():
            d = load_day(cfg["log_dir"], date_str)
            if d["pnl"] != 0 or d["trades"] > 0 or d["equity"] > 0:
                wk = weeks_data[week][strat_name]
                wk["pnl"] += d["pnl"]
                wk["trades"] += d["trades"]
                wk["wins"] += d["wins"]
                wk["losses"] += d["losses"]
                wk["days"] += 1

    # Build HTML
    html = build_html(dict(weeks_data), all_dates, max_weeks=args.weeks)

    # Save and open
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "weekly_report.html")
    with open(report_path, "w") as f:
        f.write(html)

    print(f"Report saved to {report_path}")
    webbrowser.open(f"file:///{os.path.abspath(report_path).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
