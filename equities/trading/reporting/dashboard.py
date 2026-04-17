"""Performance reporting: generates HTML dashboard with embedded plots.

Works with both backtest results and live trade logs.
Saves output to reports/ directory.
"""

import os
import logging
from datetime import datetime
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from trading.strategies.base import Trade
from trading.backtest.engine import BacktestResult
from trading.config import REPORTS_DIR

log = logging.getLogger(__name__)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def plot_equity_curve(results: list[BacktestResult]) -> str:
    """Plot equity curves for one or more backtest results."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for r in results:
        label = f"{r.symbol} ({r.strategy_name})"
        eq = r.equity_curve
        ax.plot(eq.index, eq.values, label=label, linewidth=1.2)

    ax.set_title("Equity Curve", fontsize=14)
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def plot_daily_pnl(results: list[BacktestResult]) -> str:
    """Plot daily P&L bar chart."""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.5 * len(results)),
                             squeeze=False)
    for idx, r in enumerate(results):
        ax = axes[idx, 0]
        if r.trades:
            trade_df = pd.DataFrame([
                {"date": t.exit_time, "pnl": t.pnl} for t in r.trades
            ])
            trade_df["date"] = pd.to_datetime(trade_df["date"])
            daily = trade_df.groupby(trade_df["date"].dt.date)["pnl"].sum()
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in daily.values]
            ax.bar(range(len(daily)), daily.values, color=colors, width=0.8)
            ax.set_title(f"Daily P&L: {r.symbol}", fontsize=11)
            ax.set_ylabel("P&L ($)")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3, axis="y")
            # Show date labels sparsely
            tick_step = max(1, len(daily) // 10)
            ax.set_xticks(range(0, len(daily), tick_step))
            ax.set_xticklabels(
                [str(d) for d in daily.index[::tick_step]],
                rotation=45, fontsize=7,
            )

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_trade_distribution(results: list[BacktestResult]) -> str:
    """Plot trade return distribution histogram."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             squeeze=False)
    for idx, r in enumerate(results):
        ax = axes[0, idx]
        if r.trades:
            pnl_pcts = [t.pnl_pct * 100 for t in r.trades]
            ax.hist(pnl_pcts, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
            ax.axvline(0, color="red", linewidth=1, linestyle="--")
            ax.axvline(np.mean(pnl_pcts), color="green", linewidth=1.5,
                       linestyle="--", label=f"Mean: {np.mean(pnl_pcts):.3f}%")
            ax.set_title(f"Trade Returns: {r.symbol}", fontsize=11)
            ax.set_xlabel("Return (%)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_drawdown(results: list[BacktestResult]) -> str:
    """Plot drawdown curve."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for r in results:
        eq = r.equity_curve
        cummax = eq.cummax()
        dd = (eq - cummax) / cummax * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3,
                        label=f"{r.symbol} (max: {dd.min():.1f}%)")
        ax.plot(dd.index, dd.values, linewidth=0.8)

    ax.set_title("Drawdown", fontsize=14)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def build_trade_log_html(trades: list[Trade], symbol: str) -> str:
    """Build HTML table of trades."""
    if not trades:
        return "<p>No trades.</p>"

    rows = []
    for t in trades:
        pnl_class = "win" if t.pnl >= 0 else "loss"
        rows.append(f"""<tr class="{pnl_class}">
            <td>{t.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.entry_time, 'strftime') else t.entry_time}</td>
            <td>{t.exit_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.exit_time, 'strftime') else t.exit_time}</td>
            <td>{t.direction}</td>
            <td>{t.shares}</td>
            <td>${t.entry_price:.2f}</td>
            <td>${t.exit_price:.2f}</td>
            <td class="{pnl_class}">${t.pnl:+.2f}</td>
            <td class="{pnl_class}">{t.pnl_pct*100:+.3f}%</td>
            <td>{t.exit_reason}</td>
        </tr>""")

    return f"""
    <h3>Trade Log: {symbol} ({len(trades)} trades)</h3>
    <table>
        <thead><tr>
            <th>Entry</th><th>Exit</th><th>Dir</th><th>Shares</th>
            <th>Entry$</th><th>Exit$</th><th>P&L</th><th>P&L%</th><th>Reason</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>"""


def build_metrics_html(r: BacktestResult) -> str:
    """Build HTML metrics card."""
    return f"""
    <div class="metrics-card">
        <h3>{r.symbol} — {r.strategy_name}</h3>
        <table class="metrics">
            <tr><td>Period</td><td>{r.start_date} to {r.end_date}</td></tr>
            <tr><td>Total Return</td><td class="{'win' if r.total_return >= 0 else 'loss'}">{r.total_return*100:+.2f}%</td></tr>
            <tr><td>Sharpe Ratio</td><td>{r.sharpe_ratio:.2f}</td></tr>
            <tr><td>Max Drawdown</td><td class="loss">{r.max_drawdown*100:.2f}%</td></tr>
            <tr><td>Win Rate</td><td>{r.win_rate*100:.1f}%</td></tr>
            <tr><td>Profit Factor</td><td>{r.profit_factor:.2f}</td></tr>
            <tr><td>Total Trades</td><td>{r.num_trades}</td></tr>
            <tr><td>Avg Trade</td><td>{r.avg_trade_pct:+.4f}%</td></tr>
            <tr><td>Avg Hold (min)</td><td>{r.avg_bars_held:.0f}</td></tr>
            <tr><td>Exposure</td><td>{r.exposure_pct:.1f}%</td></tr>
            <tr><td>Params</td><td style="font-size:11px">{r.params}</td></tr>
        </table>
    </div>"""


def generate_report(results: list[BacktestResult],
                    title: str = "Trading System Report",
                    comparison: list[BacktestResult] | None = None) -> str:
    """Generate a full HTML report from backtest results.

    Args:
        results: List of BacktestResult objects
        title: Report title
        comparison: Optional baseline results for before/after comparison

    Returns:
        Path to the generated HTML file
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"report_{timestamp}.html")

    # Generate plots
    equity_img = plot_equity_curve(results)
    daily_pnl_img = plot_daily_pnl(results)
    dist_img = plot_trade_distribution(results)
    dd_img = plot_drawdown(results)

    # Build comparison table if baseline provided
    comparison_html = ""
    if comparison:
        comparison_html = _build_comparison_table(comparison, results)

    # Build metrics cards
    metrics_html = "".join(build_metrics_html(r) for r in results)

    # Build trade logs (cap at 200 most recent per symbol)
    trade_logs = ""
    for r in results:
        recent = r.trades[-200:] if len(r.trades) > 200 else r.trades
        trade_logs += build_trade_log_html(recent, r.symbol)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 20px; background: #f5f5f5; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #34495e; margin-top: 30px; }}
    h3 {{ color: #2c3e50; }}
    .metrics-card {{ background: white; padding: 15px; border-radius: 8px;
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: inline-block;
                     margin: 10px; vertical-align: top; min-width: 280px; }}
    .metrics td:first-child {{ font-weight: bold; padding-right: 15px; }}
    .metrics td {{ padding: 3px 5px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #ddd; font-size: 12px; }}
    th {{ background: #34495e; color: white; }}
    tr:hover {{ background: #f0f0f0; }}
    .win {{ color: #27ae60; }}
    .loss {{ color: #e74c3c; }}
    img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .plot {{ background: white; padding: 15px; border-radius: 8px;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
    .compare-table {{ background: white; padding: 15px; border-radius: 8px;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
    .compare-table th {{ background: #2c3e50; }}
    .better {{ color: #27ae60; font-weight: bold; }}
    .worse {{ color: #e74c3c; font-weight: bold; }}
    .generated {{ color: #999; font-size: 12px; margin-top: 30px; }}
</style>
</head><body>
<h1>{title}</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

{comparison_html}

<h2>Performance Summary</h2>
<div>{metrics_html}</div>

<h2>Equity Curve</h2>
<div class="plot"><img src="data:image/png;base64,{equity_img}"></div>

<h2>Drawdown</h2>
<div class="plot"><img src="data:image/png;base64,{dd_img}"></div>

<h2>Daily P&L</h2>
<div class="plot"><img src="data:image/png;base64,{daily_pnl_img}"></div>

<h2>Trade Return Distribution</h2>
<div class="plot"><img src="data:image/png;base64,{dist_img}"></div>

<h2>Trade Log</h2>
{trade_logs}

<p class="generated">Generated by spy-trader reporting module</p>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info("Report saved to %s", path)
    return path


def _build_comparison_table(baseline: list[BacktestResult],
                            current: list[BacktestResult]) -> str:
    """Build a before/after comparison table."""
    rows = []
    base_map = {r.symbol: r for r in baseline}
    curr_map = {r.symbol: r for r in current}

    all_syms = sorted(set(list(base_map.keys()) + list(curr_map.keys())))

    for sym in all_syms:
        b = base_map.get(sym)
        c = curr_map.get(sym)
        if not b or not c:
            continue

        def cmp(bv, cv, higher_better=True):
            if higher_better:
                cls = "better" if cv > bv else ("worse" if cv < bv else "")
            else:
                cls = "better" if cv < bv else ("worse" if cv > bv else "")
            return cls

        rows.append(f"""<tr>
            <td><b>{sym}</b></td>
            <td>{b.num_trades}</td><td>{c.num_trades}</td>
            <td>{b.total_return*100:+.2f}%</td>
            <td class="{cmp(b.total_return, c.total_return)}">{c.total_return*100:+.2f}%</td>
            <td>{b.sharpe_ratio:.2f}</td>
            <td class="{cmp(b.sharpe_ratio, c.sharpe_ratio)}">{c.sharpe_ratio:.2f}</td>
            <td>{b.max_drawdown*100:.1f}%</td>
            <td class="{cmp(b.max_drawdown, c.max_drawdown, higher_better=False)}">{c.max_drawdown*100:.1f}%</td>
            <td>{b.win_rate*100:.1f}%</td>
            <td class="{cmp(b.win_rate, c.win_rate)}">{c.win_rate*100:.1f}%</td>
            <td>{b.profit_factor:.2f}</td>
            <td class="{cmp(b.profit_factor, c.profit_factor)}">{c.profit_factor:.2f}</td>
        </tr>""")

    return f"""
    <h2>Before vs After Comparison</h2>
    <div class="compare-table">
    <table>
        <thead><tr>
            <th>Symbol</th>
            <th colspan="2">Trades (B/A)</th>
            <th colspan="2">Return (B/A)</th>
            <th colspan="2">Sharpe (B/A)</th>
            <th colspan="2">MaxDD (B/A)</th>
            <th colspan="2">WinR (B/A)</th>
            <th colspan="2">PF (B/A)</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    <p style="font-size:11px; color:#666;">
        B = Baseline (no filters, 95% position), A = After (filters + 30% position).
        <span class="better">Green = improved</span>,
        <span class="worse">Red = degraded</span>.
    </p>
    </div>"""
