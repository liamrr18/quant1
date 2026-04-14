#!/usr/bin/env python3
"""Compute TRUE portfolio-level OOS Sharpe from combined equity curves.

The evaluation script averaged individual symbol Sharpes (1.12 baseline, 1.53 per-symbol).
But the REAL portfolio Sharpe depends on correlation between symbols' returns.
If returns are uncorrelated, diversification pushes portfolio Sharpe above the average.

This script:
1. Runs walk-forward for each symbol
2. Stitches OOS equity curves together
3. Computes the ACTUAL combined portfolio Sharpe
4. Tests different portfolio compositions (all 3, QQQ+IWM, etc.)
5. Compares against buy-and-hold benchmarks
"""

import sys, os, logging
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.orb import ORBBreakout
from trading.backtest.engine import run_backtest
from trading.backtest.walkforward import walk_forward
from trading.config import SYMBOL_PROFILES, ORB_SHARED_DEFAULTS

ET = pytz.timezone("America/New_York")
start = datetime(2025, 1, 2, tzinfo=ET)
end = datetime(2026, 4, 4, tzinfo=ET)

SYMBOLS = ["SPY", "QQQ", "IWM"]

# ── Load data ──
data = {}
for sym in SYMBOLS:
    df = get_minute_bars(sym, start, end, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"Loaded {sym}: {len(df)} bars")


def make_strat(sym):
    params = dict(ORB_SHARED_DEFAULTS)
    if sym in SYMBOL_PROFILES:
        params.update(SYMBOL_PROFILES[sym])
    return ORBBreakout(**params)


def get_oos_daily_returns(sym, strat):
    """Run walk-forward and extract concatenated OOS daily returns."""
    df = data[sym].copy()
    df = strat.generate_signals(df)
    wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)

    all_daily = []
    for oos_result in wf.oos_results:
        dr = oos_result.daily_returns
        if dr is not None and len(dr) > 0:
            all_daily.append(dr)

    if all_daily:
        combined = pd.concat(all_daily).sort_index()
        # Remove duplicates (overlapping windows)
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined, wf
    return pd.Series(dtype=float), wf


def portfolio_sharpe(returns_dict, equal_weight=True):
    """Compute portfolio Sharpe from multiple symbol daily return series."""
    # Align all returns to same date index
    df = pd.DataFrame(returns_dict)
    df = df.fillna(0)  # No return = flat day

    if equal_weight:
        portfolio_returns = df.mean(axis=1)  # Equal weight
    else:
        portfolio_returns = df.sum(axis=1)  # Sum (for sizing-adjusted)

    if len(portfolio_returns) < 10 or portfolio_returns.std() == 0:
        return 0.0, portfolio_returns

    sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    return sharpe, portfolio_returns


def buy_and_hold_returns(sym):
    """Compute daily returns for buy-and-hold of a symbol over the same period."""
    df = data[sym]
    # Get daily close prices
    daily = df.set_index("dt")["close"].resample("D").last().dropna()
    returns = daily.pct_change().dropna()
    return returns


# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STEP 1: Extract OOS daily returns per symbol")
print("=" * 100)

sym_returns = {}
sym_wf = {}
for sym in SYMBOLS:
    strat = make_strat(sym)
    returns, wf = get_oos_daily_returns(sym, strat)
    sym_returns[sym] = returns
    sym_wf[sym] = wf
    print(f"  {sym}: {len(returns)} OOS daily returns, individual OOS Sharpe={wf.sharpe_ratio:.2f}")

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STEP 2: True portfolio Sharpe (combined equity curves)")
print("=" * 100)

# All 3 symbols
sharpe_3, returns_3 = portfolio_sharpe(sym_returns)
print(f"\n  SPY+QQQ+IWM (equal weight):")
print(f"    True portfolio Sharpe: {sharpe_3:.2f}")
print(f"    Avg daily return: {returns_3.mean()*100:.4f}%")
print(f"    Daily return std: {returns_3.std()*100:.4f}%")
print(f"    Total OOS return: {(1+returns_3).prod() - 1:.2%}")
print(f"    Max daily loss: {returns_3.min()*100:.2f}%")
print(f"    Trading days: {len(returns_3)}")

# Correlation matrix
corr_df = pd.DataFrame(sym_returns)
corr = corr_df.corr()
print(f"\n  Return correlation matrix:")
print(corr.to_string())

# QQQ+IWM only (drop SPY)
sharpe_2, returns_2 = portfolio_sharpe({k: v for k, v in sym_returns.items() if k != "SPY"})
print(f"\n  QQQ+IWM only (equal weight):")
print(f"    True portfolio Sharpe: {sharpe_2:.2f}")
print(f"    Total OOS return: {(1+returns_2).prod() - 1:.2%}")

# IWM only
sharpe_iwm = sym_wf["IWM"].sharpe_ratio
print(f"\n  IWM only: OOS Sharpe = {sharpe_iwm:.2f}")

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STEP 3: Buy-and-hold benchmark comparison")
print("=" * 100)

bh_returns = {}
for sym in SYMBOLS:
    bh = buy_and_hold_returns(sym)
    bh_returns[sym] = bh
    bh_sharpe = (bh.mean() / bh.std()) * np.sqrt(252) if bh.std() > 0 else 0
    bh_total = (1 + bh).prod() - 1
    print(f"  {sym} B&H: Sharpe={bh_sharpe:.2f}, Return={bh_total:.2%}")

# B&H portfolio
bh_port_sharpe, bh_port_ret = portfolio_sharpe(bh_returns)
bh_total = (1 + bh_port_ret).prod() - 1
print(f"  B&H Portfolio (equal weight): Sharpe={bh_port_sharpe:.2f}, Return={bh_total:.2%}")

# Compare
print(f"\n  ACTIVE vs PASSIVE:")
print(f"    Active portfolio Sharpe:  {sharpe_3:.2f}")
print(f"    Passive B&H Sharpe:       {bh_port_sharpe:.2f}")
print(f"    Active advantage:         {sharpe_3 - bh_port_sharpe:+.2f}")
print(f"    Active total return:      {(1+returns_3).prod() - 1:.2%}")
print(f"    Passive total return:     {bh_total:.2%}")

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STEP 4: Consistency analysis (per-window Sharpe)")
print("=" * 100)

for sym in SYMBOLS:
    wf = sym_wf[sym]
    window_sharpes = [r.sharpe_ratio for r in wf.oos_results]
    pos = sum(1 for s in window_sharpes if s > 0)
    print(f"  {sym}: {pos}/{len(window_sharpes)} positive windows, "
          f"median={np.median(window_sharpes):.2f}, "
          f"mean={np.mean(window_sharpes):.2f}, "
          f"min={min(window_sharpes):.2f}, max={max(window_sharpes):.2f}")

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STEP 5: Best achievable portfolios")
print("=" * 100)

combos = [
    ("SPY+QQQ+IWM", ["SPY", "QQQ", "IWM"]),
    ("QQQ+IWM", ["QQQ", "IWM"]),
    ("SPY+IWM", ["SPY", "IWM"]),
    ("SPY+QQQ", ["SPY", "QQQ"]),
    ("IWM only", ["IWM"]),
    ("QQQ only", ["QQQ"]),
    ("SPY only", ["SPY"]),
]

print(f"  {'Combo':<16} {'Sharpe':>8} {'Return':>10} {'Trades':>8}")
print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*8}")
for name, syms in combos:
    rets = {s: sym_returns[s] for s in syms}
    s, r = portfolio_sharpe(rets)
    total_trades = sum(sym_wf[s].total_trades for s in syms)
    total_ret = (1 + r).prod() - 1
    marker = " <-- TARGET" if s >= 2.0 else ""
    print(f"  {name:<16} {s:>8.2f} {total_ret:>+10.2%} {total_trades:>8}{marker}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
