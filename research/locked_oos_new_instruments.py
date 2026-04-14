#!/usr/bin/env python3
"""Phase 2, Experiment 13: Locked OOS confirmation for GLD and XLE.

Pre-specified configs from dev period analysis:
- GLD: ORB shared defaults + stale_exit_bars=120 (dev Sharpe 2.41)
- XLE: ORB shared defaults (no additional filters needed, dev Sharpe 1.82)

Tests portfolio configurations:
A: Baseline SPY+QQQ (current production)
B: SPY+QQQ+GLD (3-symbol with gold)
C: SPY+QQQ+XLE (3-symbol with energy)
D: SPY+QQQ+GLD+XLE (4-symbol)

ONE-TIME confirmation. Results are NOT used to iterate on configs.
"""

import sys, os, logging, warnings
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.orb import ORBBreakout
from trading.backtest.engine import run_backtest
from trading.config import ORB_SHARED_DEFAULTS, SYMBOL_PROFILES

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
LOCKED_START = datetime(2025, 12, 1, tzinfo=ET)

ALL_SYMS = ["SPY", "QQQ", "GLD", "XLE"]

# Pre-specified configs (frozen from dev period analysis)
PROFILES = {
    "SPY": dict(SYMBOL_PROFILES["SPY"]),  # Production profile
    "QQQ": dict(SYMBOL_PROFILES["QQQ"]),  # Production profile
    "GLD": {"stale_exit_bars": 120},      # Best on dev: Sharpe 2.41
    "XLE": {},                             # Best on dev: defaults, Sharpe 1.82
}

PORTFOLIO_CONFIGS = {
    "A: SPY+QQQ (baseline)":       ["SPY", "QQQ"],
    "B: SPY+QQQ+GLD":              ["SPY", "QQQ", "GLD"],
    "C: SPY+QQQ+XLE":              ["SPY", "QQQ", "XLE"],
    "D: SPY+QQQ+GLD+XLE":          ["SPY", "QQQ", "GLD", "XLE"],
    "E: GLD only":                  ["GLD"],
    "F: XLE only":                  ["XLE"],
}

# ── Load data ──
print("Loading data...")
raw_data = {}
for sym in ALL_SYMS:
    df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
    df = prepare_features(df)
    raw_data[sym] = df
    print(f"  {sym}: {len(df)} bars, {df['date'].nunique()} days")


def make_strat(sym):
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(PROFILES.get(sym, {}))
    return ORBBreakout(**params)


def split_data(df, start_dt, end_dt):
    mask = (df["dt"] >= start_dt) & (df["dt"] <= end_dt)
    return df.loc[mask].copy()


def get_benchmark_daily(start_dt, end_dt):
    df = raw_data["SPY"]
    sub = split_data(df, start_dt, end_dt)
    daily = sub.set_index("dt")["close"].resample("D").last().dropna()
    return daily.pct_change().dropna()


def compute_metrics(daily_returns, bench_daily):
    """Compute full metrics for a return series."""
    dr = daily_returns
    n = len(dr)
    if n < 5 or dr.std() == 0:
        return None

    total_ret = (1 + dr).prod() - 1
    annual_ret = (1 + total_ret) ** (252 / max(n, 1)) - 1
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
    down = dr[dr < 0]
    down_std = down.std() if len(down) > 2 else dr.std()
    sortino = (dr.mean() / down_std) * np.sqrt(252) if down_std > 0 else 0
    cum = (1 + dr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

    aligned = pd.DataFrame({"s": dr, "b": bench_daily}).dropna()
    if len(aligned) > 10:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(aligned["b"], aligned["s"])
        beta = slope
        alpha_ann = intercept * 252
        resid = aligned["s"] - (beta * aligned["b"] + intercept)
        te = resid.std() * np.sqrt(252)
        ir = alpha_ann / te if te > 0 else 0
        r_squared = r_val ** 2
    else:
        beta = alpha_ann = ir = r_squared = 0

    bench_ret = (1 + bench_daily).prod() - 1

    return {
        "n_days": n, "total_return": total_ret, "annual_return": annual_ret,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "max_drawdown": max_dd,
        "alpha": alpha_ann, "beta": beta, "ir": ir, "r_squared": r_squared,
        "bench_return": bench_ret, "active_passive": total_ret - bench_ret,
    }


def run_portfolio(config_name, symbols, start_dt, end_dt, bench):
    """Run a portfolio config on a period, return metrics."""
    sym_daily = {}
    sym_metrics = {}
    total_trades = 0

    for sym in symbols:
        strat = make_strat(sym)
        df = split_data(raw_data[sym], start_dt, end_dt)
        df = strat.generate_signals(df)
        result = run_backtest(df, strat, sym)

        dr = result.daily_returns
        if dr is not None and len(dr) > 5:
            sym_daily[sym] = dr
            m = compute_metrics(dr, bench)
            if m:
                m["trades"] = result.num_trades
                m["win_rate"] = result.win_rate
                m["pf"] = result.profit_factor
                m["exposure"] = result.exposure_pct
                sym_metrics[sym] = m
                total_trades += result.num_trades

    # Portfolio return (equal weight)
    if len(sym_daily) >= 1:
        port_df = pd.DataFrame(sym_daily).fillna(0)
        port_ret = port_df.mean(axis=1)
        port_m = compute_metrics(port_ret, bench)
        if port_m:
            port_m["trades"] = total_trades
            port_m["n_symbols"] = len(sym_daily)
            # Cross-correlations
            if len(sym_daily) >= 2:
                corr_matrix = port_df.corr()
                avg_corr = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                ).stack().mean()
                port_m["avg_corr"] = avg_corr
            else:
                port_m["avg_corr"] = 1.0
            return port_m, sym_metrics

    return None, sym_metrics


# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("EXPERIMENT 13: LOCKED OOS CONFIRMATION — GLD & XLE")
print("Configs frozen from dev period. Results are NOT used to iterate.")
print("=" * 100)

# Print profiles
for sym in ALL_SYMS:
    strat = make_strat(sym)
    print(f"  {sym}: {strat.get_params()}")

# ── Run on all three periods ──
periods = [
    ("DEV (Jan-Nov 2025)", DATA_START, DEV_END),
    ("LOCKED OOS (Dec 2025-Apr 2026)", LOCKED_START, DATA_END),
    ("FULL (Jan 2025-Apr 2026)", DATA_START, DATA_END),
]

all_results = {}

for period_name, start, end in periods:
    print(f"\n{'='*100}")
    print(f"{period_name}")
    print(f"{'='*100}")

    bench = get_benchmark_daily(start, end)

    # Per-symbol results first
    print(f"\n  Per-symbol results:")
    for sym in ALL_SYMS:
        strat = make_strat(sym)
        df = split_data(raw_data[sym], start, end)
        df = strat.generate_signals(df)
        result = run_backtest(df, strat, sym)
        dr = result.daily_returns
        if dr is not None and len(dr) > 5:
            m = compute_metrics(dr, bench)
            if m:
                print(f"    {sym}: Sharpe={m['sharpe']:>5.2f}  Sortino={m['sortino']:>5.2f}"
                      f"  Return={m['total_return']:>+7.2%}  Alpha={m['alpha']:>+7.2%}"
                      f"  Beta={m['beta']:>6.3f}  MaxDD={m['max_drawdown']:>6.2%}"
                      f"  Trades={result.num_trades}  WR={result.win_rate:.1%}"
                      f"  PF={result.profit_factor:.2f}")

    # Portfolio configs
    print(f"\n  Portfolio results:")
    print(f"  {'Config':<30} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Alpha':>8}"
          f" {'Beta':>6} {'MaxDD':>7} {'Trades':>7} {'AvgCorr':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")

    for config_name, symbols in PORTFOLIO_CONFIGS.items():
        port_m, sym_m = run_portfolio(config_name, symbols, start, end, bench)
        if port_m:
            all_results[(period_name, config_name)] = port_m
            print(f"  {config_name:<30} {port_m['sharpe']:>7.2f} {port_m['sortino']:>8.2f}"
                  f" {port_m['total_return']:>+8.2%} {port_m['alpha']:>+8.2%}"
                  f" {port_m['beta']:>6.3f} {port_m['max_drawdown']:>7.2%}"
                  f" {port_m['trades']:>7} {port_m.get('avg_corr', 0):>8.2f}")

# ── Summary comparison ──
print(f"\n\n{'='*100}")
print("LOCKED OOS COMPARISON (the only numbers that matter)")
print(f"{'='*100}")

print(f"\n  {'Config':<30} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Alpha':>8}"
      f" {'Beta':>6} {'MaxDD':>7} {'Calmar':>7} {'IR':>6}")
print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")

locked_key = "LOCKED OOS (Dec 2025-Apr 2026)"
for config_name in PORTFOLIO_CONFIGS:
    key = (locked_key, config_name)
    if key in all_results:
        m = all_results[key]
        print(f"  {config_name:<30} {m['sharpe']:>7.2f} {m['sortino']:>8.2f}"
              f" {m['total_return']:>+8.2%} {m['alpha']:>+8.2%}"
              f" {m['beta']:>6.3f} {m['max_drawdown']:>7.2%}"
              f" {m['calmar']:>7.2f} {m['ir']:>6.2f}")

print(f"\n{'='*100}")
print("CONFIRMATION COMPLETE")
print(f"{'='*100}")
