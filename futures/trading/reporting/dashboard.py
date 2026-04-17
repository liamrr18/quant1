"""Reporting dashboard for futures backtests.

Generates HTML reports with equity curves, drawdowns, trade distributions,
and futures-specific metrics (dollar P&L, contracts, costs).
"""

import base64
import io
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from trading.backtest.engine import BacktestResult

log = logging.getLogger(__name__)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_equity_curve(results: list[BacktestResult]) -> str:
    fig, ax = plt.subplots(figsize=(12, 5))
    for r in results:
        eq = r.equity_curve.resample("D").last().dropna()
        ax.plot(eq.index, eq.values, label=f"{r.symbol}", linewidth=1.5)
    ax.set_title("Futures Equity Curve", fontsize=14)
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    return _fig_to_base64(fig)


def plot_daily_pnl(results: list[BacktestResult]) -> str:
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)),
                             squeeze=False)
    for idx, r in enumerate(results):
        ax = axes[idx, 0]
        daily = r.equity_curve.resample("D").last().dropna()
        daily_pnl = daily.diff().dropna()
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in daily_pnl.values]
        ax.bar(daily_pnl.index, daily_pnl.values, color=colors, width=0.8)
        ax.set_title(f"{r.symbol} Daily P&L ($)", fontsize=12)
        ax.set_ylabel("P&L ($)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
        n = max(len(daily_pnl) // 15, 1)
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % n != 0:
                label.set_visible(False)
    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_trade_distribution(results: list[BacktestResult]) -> str:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4),
                             squeeze=False)
    for idx, r in enumerate(results):
        ax = axes[0, idx]
        pnls = [t.pnl for t in r.trades]
        if pnls:
            ax.hist(pnls, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
            ax.axvline(x=np.mean(pnls), color="red", linestyle="--",
                       label=f"Mean: ${np.mean(pnls):+,.0f}")
            ax.legend()
        ax.set_title(f"{r.symbol} Trade P&L Distribution ($)", fontsize=12)
        ax.set_xlabel("P&L ($)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_drawdown(results: list[BacktestResult]) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    for r in results:
        eq = r.equity_curve.resample("D").last().dropna()
        cummax = eq.cummax()
        dd = (eq - cummax) / cummax * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3,
                        label=f"{r.symbol} (max: {r.max_drawdown*100:.1f}%)")
        ax.plot(dd.index, dd.values, linewidth=1)
    ax.set_title("Drawdown (%)", fontsize=14)
    ax.set_ylabel("Drawdown %")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def build_metrics_html(result: BacktestResult) -> str:
    """Build metrics card for a futures backtest result."""
    contract = result.contract
    return f"""
    <div class="metrics-card">
        <h3>{result.symbol} ({contract.name})</h3>
        <table>
            <tr><td>Period</td><td>{result.start_date} to {result.end_date}</td></tr>
            <tr><td>Proxy</td><td>{contract.proxy_symbol} (ETF data)</td></tr>
            <tr><td>Multiplier</td><td>${contract.multiplier:.0f} per $1 {contract.proxy_symbol} move</td></tr>
            <tr><td>Margin/Contract</td><td>${contract.margin_intraday:,.0f}</td></tr>
            <tr class="sep"><td colspan="2"></td></tr>
            <tr><td>Total Return</td><td class="{'pos' if result.total_return >= 0 else 'neg'}">{result.total_return*100:+.2f}%</td></tr>
            <tr><td>Total P&L</td><td class="{'pos' if result.total_pnl >= 0 else 'neg'}">${result.total_pnl:+,.0f}</td></tr>
            <tr><td>Sharpe Ratio</td><td>{result.sharpe_ratio:.2f}</td></tr>
            <tr><td>Max Drawdown</td><td class="neg">{result.max_drawdown*100:.2f}%</td></tr>
            <tr class="sep"><td colspan="2"></td></tr>
            <tr><td>Trades</td><td>{result.num_trades}</td></tr>
            <tr><td>Win Rate</td><td>{result.win_rate*100:.1f}%</td></tr>
            <tr><td>Profit Factor</td><td>{result.profit_factor:.2f}</td></tr>
            <tr><td>Avg Trade</td><td>${result.avg_trade_dollars:+,.0f} ({result.avg_trade_pct:+.3f}%)</td></tr>
            <tr><td>Avg Contracts</td><td>{result.avg_contracts:.1f}</td></tr>
            <tr><td>Avg Hold (min)</td><td>{result.avg_bars_held:.0f}</td></tr>
            <tr><td>Exposure</td><td>{result.exposure_pct:.1f}%</td></tr>
            <tr class="sep"><td colspan="2"></td></tr>
            <tr><td>Total Costs</td><td>${result.total_costs:,.0f}</td></tr>
            <tr><td>Params</td><td style="font-size:0.85em">{result.params}</td></tr>
        </table>
    </div>
    """


def build_trade_log_html(trades, symbol: str, max_trades: int = 200) -> str:
    if not trades:
        return "<p>No trades</p>"
    recent = trades[-max_trades:]
    rows = ""
    for t in recent:
        cls = "win" if t.pnl > 0 else "loss"
        rows += f"""<tr class="{cls}">
            <td>{t.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.entry_time, 'strftime') else t.entry_time}</td>
            <td>{t.exit_time.strftime('%H:%M') if hasattr(t.exit_time, 'strftime') else t.exit_time}</td>
            <td>{t.direction}</td>
            <td>{t.contracts}</td>
            <td>${t.entry_price:.2f}</td>
            <td>${t.exit_price:.2f}</td>
            <td class="{'pos' if t.pnl > 0 else 'neg'}">${t.pnl:+,.0f}</td>
            <td>{t.exit_reason}</td>
        </tr>"""

    return f"""
    <h4>{symbol} Trades (last {len(recent)})</h4>
    <table class="trades">
        <tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Contracts</th>
            <th>Entry$</th><th>Exit$</th><th>P&L</th><th>Reason</th></tr>
        {rows}
    </table>
    """


def generate_report(results: list[BacktestResult], title: str = "Futures ORB Backtest",
                    comparison: dict = None) -> str:
    """Generate full HTML report."""
    os.makedirs("reports", exist_ok=True)

    equity_img = plot_equity_curve(results)
    dd_img = plot_drawdown(results)
    pnl_img = plot_daily_pnl(results)
    dist_img = plot_trade_distribution(results)

    metrics_html = ""
    trade_logs = ""
    for r in results:
        metrics_html += build_metrics_html(r)
        trade_logs += build_trade_log_html(r.trades, r.symbol)

    comparison_html = ""
    if comparison:
        comparison_html = _build_comparison_table(comparison)

    html = f"""<!DOCTYPE html>
<html><head><title>{title}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
    h1, h2, h3 {{ color: #e94560; }}
    .metrics-card {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 10px;
                     display: inline-block; vertical-align: top; min-width: 350px; }}
    .metrics-card table {{ width: 100%; border-collapse: collapse; }}
    .metrics-card td {{ padding: 4px 8px; border-bottom: 1px solid #333; }}
    .metrics-card td:first-child {{ color: #888; }}
    .pos {{ color: #2ecc71; font-weight: bold; }}
    .neg {{ color: #e74c3c; font-weight: bold; }}
    .sep td {{ border: none; height: 8px; }}
    table.trades {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
    table.trades th {{ background: #16213e; padding: 6px; text-align: left; }}
    table.trades td {{ padding: 4px 6px; border-bottom: 1px solid #333; }}
    tr.win {{ background: rgba(46, 204, 113, 0.05); }}
    tr.loss {{ background: rgba(231, 76, 60, 0.05); }}
    img {{ max-width: 100%; border-radius: 8px; margin: 10px 0; }}
    .comparison {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 20px 0; }}
    .comparison table {{ width: 100%; border-collapse: collapse; }}
    .comparison th, .comparison td {{ padding: 8px; text-align: center; border: 1px solid #333; }}
    .improved {{ color: #2ecc71; }}
    .degraded {{ color: #e74c3c; }}
</style></head><body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

    <h2>Equity Curve</h2>
    <img src="data:image/png;base64,{equity_img}">

    <h2>Drawdown</h2>
    <img src="data:image/png;base64,{dd_img}">

    <h2>Daily P&L</h2>
    <img src="data:image/png;base64,{pnl_img}">

    <h2>Trade Distribution</h2>
    <img src="data:image/png;base64,{dist_img}">

    <h2>Metrics</h2>
    <div class="metrics-container">{metrics_html}</div>

    {comparison_html}

    <h2>Trade Log</h2>
    {trade_logs}
</body></html>"""

    path = os.path.join("reports", f"futures_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(path, "w") as f:
        f.write(html)
    log.info("Report saved: %s", path)
    return path


def _build_comparison_table(comparison: dict) -> str:
    """Build equity vs futures comparison table."""
    rows = ""
    for metric, values in comparison.items():
        equity_val = values.get("equity", "")
        futures_val = values.get("futures", "")
        cls = ""
        if values.get("better") == "futures":
            cls = "improved"
        elif values.get("better") == "equity":
            cls = "degraded"
        rows += f"<tr><td>{metric}</td><td>{equity_val}</td><td class='{cls}'>{futures_val}</td></tr>"

    return f"""
    <div class="comparison">
        <h2>Equity vs Futures Comparison</h2>
        <table>
            <tr><th>Metric</th><th>Equity (SPY/QQQ)</th><th>Futures (MES/MNQ)</th></tr>
            {rows}
        </table>
    </div>
    """
