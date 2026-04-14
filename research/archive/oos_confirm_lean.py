#!/usr/bin/env python3
"""Lean locked OOS confirmation for GLD and XLE. Just the numbers that matter."""

import sys, os, logging, warnings, io
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Force unbuffered output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)

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
LOCKED_START = datetime(2025, 12, 1, tzinfo=ET)

PROFILES = {
    "SPY": dict(SYMBOL_PROFILES["SPY"]),
    "QQQ": dict(SYMBOL_PROFILES["QQQ"]),
    "GLD": {"stale_exit_bars": 120},
    "XLE": {},
}

print("Loading data...", flush=True)
raw = {}
for sym in ["SPY", "QQQ", "GLD", "XLE"]:
    df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
    df = prepare_features(df)
    raw[sym] = df
    print(f"  {sym}: {len(df)} bars", flush=True)


def make_strat(sym):
    p = dict(ORB_SHARED_DEFAULTS)
    p.update(PROFILES.get(sym, {}))
    return ORBBreakout(**p)


def run_sym(sym, start, end):
    strat = make_strat(sym)
    mask = (raw[sym]["dt"] >= start) & (raw[sym]["dt"] <= end)
    df = raw[sym].loc[mask].copy()
    df = strat.generate_signals(df)
    return run_backtest(df, strat, sym)


# SPY benchmark
spy_mask = (raw["SPY"]["dt"] >= LOCKED_START) & (raw["SPY"]["dt"] <= DATA_END)
spy_sub = raw["SPY"].loc[spy_mask]
bench = spy_sub.set_index("dt")["close"].resample("D").last().dropna().pct_change().dropna()

print(f"\n{'='*80}", flush=True)
print("LOCKED OOS: Dec 1, 2025 - Apr 4, 2026", flush=True)
print(f"{'='*80}", flush=True)

# Per-symbol
sym_daily = {}
for sym in ["SPY", "QQQ", "GLD", "XLE"]:
    print(f"\n  Running {sym}...", flush=True)
    result = run_sym(sym, LOCKED_START, DATA_END)
    dr = result.daily_returns
    if dr is not None and len(dr) > 5 and dr.std() > 0:
        sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
        total_ret = result.total_return
        cum = (1 + dr).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        down = dr[dr < 0]
        sortino = (dr.mean() / down.std()) * np.sqrt(252) if len(down) > 2 and down.std() > 0 else 0

        aligned = pd.DataFrame({"s": dr, "b": bench}).dropna()
        if len(aligned) > 10:
            slope, intercept, _, _, _ = scipy_stats.linregress(aligned["b"], aligned["s"])
            beta, alpha = slope, intercept * 252
        else:
            beta, alpha = 0, 0

        sym_daily[sym] = dr
        print(f"    Sharpe={sharpe:.2f}  Sortino={sortino:.2f}  Return={total_ret:+.2%}"
              f"  Alpha={alpha:+.2%}  Beta={beta:.3f}  MaxDD={max_dd:.2%}"
              f"  Trades={result.num_trades}  WR={result.win_rate:.1%}", flush=True)
    else:
        print(f"    No valid data", flush=True)

# Portfolios
print(f"\n{'='*80}", flush=True)
print("PORTFOLIO COMPARISONS (locked OOS)", flush=True)
print(f"{'='*80}", flush=True)

configs = [
    ("A: SPY+QQQ (baseline)", ["SPY", "QQQ"]),
    ("B: SPY+QQQ+GLD", ["SPY", "QQQ", "GLD"]),
    ("C: SPY+QQQ+XLE", ["SPY", "QQQ", "XLE"]),
    ("D: SPY+QQQ+GLD+XLE", ["SPY", "QQQ", "GLD", "XLE"]),
]

print(f"\n  {'Config':<25} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Alpha':>8}"
      f" {'Beta':>6} {'MaxDD':>7} {'Trades':>7}", flush=True)
print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*7}", flush=True)

for name, syms in configs:
    available = {s: sym_daily[s] for s in syms if s in sym_daily}
    if len(available) < 2:
        print(f"  {name:<25} INSUFFICIENT DATA", flush=True)
        continue

    port_df = pd.DataFrame(available).fillna(0)
    port_ret = port_df.mean(axis=1)
    n = len(port_ret)
    if n < 5 or port_ret.std() == 0:
        continue

    sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252)
    total_ret = (1 + port_ret).prod() - 1
    annual_ret = (1 + total_ret) ** (252 / max(n, 1)) - 1
    cum = (1 + port_ret).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    down = port_ret[port_ret < 0]
    sortino = (port_ret.mean() / down.std()) * np.sqrt(252) if len(down) > 2 and down.std() > 0 else 0

    aligned = pd.DataFrame({"s": port_ret, "b": bench}).dropna()
    if len(aligned) > 10:
        slope, intercept, _, _, _ = scipy_stats.linregress(aligned["b"], aligned["s"])
        beta, alpha = slope, intercept * 252
    else:
        beta, alpha = 0, 0

    total_trades = sum(run_sym(s, LOCKED_START, DATA_END).num_trades for s in syms)

    print(f"  {name:<25} {sharpe:>7.2f} {sortino:>8.2f} {total_ret:>+8.2%}"
          f" {alpha:>+8.2%} {beta:>6.3f} {max_dd:>7.2%} {total_trades:>7}", flush=True)

# Cross-correlations
print(f"\n  Cross-correlations (locked OOS):", flush=True)
if len(sym_daily) >= 2:
    corr_df = pd.DataFrame(sym_daily).corr()
    for i, s1 in enumerate(corr_df.columns):
        for s2 in corr_df.columns[i+1:]:
            print(f"    {s1}-{s2}: {corr_df.loc[s1, s2]:.3f}", flush=True)

print(f"\n{'='*80}", flush=True)
print("DONE", flush=True)
