#!/usr/bin/env python3
"""Phase 2, Experiment 11: Screen new instruments for ORB edge.

Methodology:
- Same walk-forward as baseline (60-day train, 20-day test, 20-day step)
- Same cost model ($0.01/share slippage, $0.00 commission)
- Test with ORB_SHARED_DEFAULTS first, then per-symbol tuning if promising
- Dev period only (Jan-Nov 2025) — locked OOS untouched
- Candidates must show standalone Sharpe > 0.5 on dev to warrant further study

Candidates: liquid ETFs with tight spreads, diverse sectors
- XLK (tech), XLF (financials), XLE (energy), XLV (healthcare)
- GLD (gold), TLT (long bonds), USO (oil), EEM (emerging markets)
- ARKK (innovation), SMH (semiconductors)
"""

import sys, os, logging, warnings
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.orb import ORBBreakout
from trading.backtest.engine import run_backtest
from trading.backtest.walkforward import walk_forward
from trading.config import ORB_SHARED_DEFAULTS, SYMBOL_PROFILES

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
LOCKED_START = datetime(2025, 12, 1, tzinfo=ET)

# Candidate instruments to screen
CANDIDATES = ["XLK", "XLF", "XLE", "XLV", "GLD", "TLT", "SMH", "ARKK", "EEM", "USO"]

# Also include baseline for comparison
BASELINE_SYMS = ["SPY", "QQQ"]


def make_strat(profile=None):
    params = dict(ORB_SHARED_DEFAULTS)
    if profile:
        params.update(profile)
    return ORBBreakout(**params)


def eval_walkforward(sym, df, strat, label=""):
    """Run walk-forward on given data, return metrics."""
    df = strat.generate_signals(df)
    wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)

    all_daily = []
    for oos_r in wf.oos_results:
        dr = oos_r.daily_returns
        if dr is not None and len(dr) > 0:
            all_daily.append(dr)

    if not all_daily:
        return None

    combined = pd.concat(all_daily).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    n = len(combined)
    if n < 10 or combined.std() == 0:
        return None

    sharpe = (combined.mean() / combined.std()) * np.sqrt(252)
    total_ret = (1 + combined).prod() - 1
    cum = (1 + combined).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    down = combined[combined < 0]
    sortino = (combined.mean() / down.std()) * np.sqrt(252) if len(down) > 2 and down.std() > 0 else 0

    window_sharpes = [r.sharpe_ratio for r in wf.oos_results]
    pos_windows = sum(1 for s in window_sharpes if s > 0)

    return {
        "sym": sym, "label": label, "sharpe": sharpe, "sortino": sortino,
        "return": total_ret, "max_dd": max_dd,
        "trades": wf.total_trades, "win_rate": wf.win_rate,
        "pf": wf.profit_factor, "windows_pos": f"{pos_windows}/{len(window_sharpes)}",
        "daily_returns": combined,
    }


def eval_period(sym, df_full, strat, start_dt, end_dt, label=""):
    """Run backtest on specific period, return metrics."""
    mask = (df_full["dt"] >= start_dt) & (df_full["dt"] <= end_dt)
    df = df_full.loc[mask].copy()
    df = strat.generate_signals(df)
    result = run_backtest(df, strat, sym)

    dr = result.daily_returns
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return None

    sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
    total_ret = result.total_return
    cum = (1 + dr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    return {
        "sym": sym, "label": label, "sharpe": sharpe,
        "return": total_ret, "max_dd": max_dd,
        "trades": result.num_trades, "win_rate": result.win_rate,
        "pf": result.profit_factor, "daily_returns": dr,
    }


# ═══════════════════════════════════════════════════════════
print("=" * 100)
print("EXPERIMENT 11: NEW INSTRUMENT SCREENING")
print("Testing ORB (shared defaults) on 10 candidate ETFs")
print("Dev period: Jan-Nov 2025 (walk-forward)")
print("=" * 100)

# Load data for all candidates
print("\nLoading data...")
data = {}
failed = []
for sym in CANDIDATES + BASELINE_SYMS:
    try:
        df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
        df = prepare_features(df)
        data[sym] = df
        days = df["date"].nunique()
        print(f"  {sym}: {len(df)} bars, {days} days")
    except Exception as e:
        print(f"  {sym}: FAILED - {e}")
        failed.append(sym)

# ── Phase A: Walk-forward screening on dev period ──
print("\n" + "=" * 100)
print("PHASE A: Walk-forward OOS screening (dev period, shared defaults)")
print("=" * 100)

strat_default = make_strat()
results = {}

# Dev period data
for sym in CANDIDATES:
    if sym in failed:
        continue
    df_dev = data[sym].loc[data[sym]["dt"] <= DEV_END].copy()
    print(f"\n  Evaluating {sym}...")
    r = eval_walkforward(sym, df_dev, strat_default, "default")
    results[sym] = r
    if r:
        print(f"    {sym}: Sharpe={r['sharpe']:>5.2f}  Sortino={r['sortino']:>5.2f}"
              f"  Ret={r['return']:>+7.2%}  DD={r['max_dd']:>6.2%}"
              f"  T={r['trades']:>4}  WR={r['win_rate']:.1%}  PF={r['pf']:.2f}"
              f"  Win={r['windows_pos']}")
    else:
        print(f"    {sym}: no valid data")

# Also run baseline for comparison
print("\n  --- Baseline comparison ---")
for sym in BASELINE_SYMS:
    profile = SYMBOL_PROFILES.get(sym, {})
    strat = make_strat(profile)
    df_dev = data[sym].loc[data[sym]["dt"] <= DEV_END].copy()
    r = eval_walkforward(sym, df_dev, strat, "production")
    results[sym] = r
    if r:
        print(f"    {sym}: Sharpe={r['sharpe']:>5.2f}  Sortino={r['sortino']:>5.2f}"
              f"  Ret={r['return']:>+7.2%}  DD={r['max_dd']:>6.2%}"
              f"  T={r['trades']:>4}  WR={r['win_rate']:.1%}  PF={r['pf']:.2f}"
              f"  Win={r['windows_pos']}")

# ── Phase B: Promising candidates get per-symbol tuning ──
PROMISING_THRESHOLD = 0.5  # Sharpe threshold to continue

promising = [sym for sym, r in results.items()
             if r is not None and r["sharpe"] > PROMISING_THRESHOLD
             and sym in CANDIDATES]

print("\n" + "=" * 100)
print(f"PHASE B: Per-symbol tuning for promising candidates (Sharpe > {PROMISING_THRESHOLD})")
print(f"Promising: {promising if promising else 'NONE'}")
print("=" * 100)

tuned_results = {}

for sym in promising:
    print(f"\n  {sym} tuning:")
    df_dev = data[sym].loc[data[sym]["dt"] <= DEV_END].copy()

    # Test common filter/exit variants
    variants = [
        ("defaults", {}),
        ("+ gap 0.3%", {"min_gap_pct": 0.3}),
        ("+ stale 90", {"stale_exit_bars": 90}),
        ("+ stale 120", {"stale_exit_bars": 120}),
        ("+ last_entry 900", {"last_entry_minute": 900}),
        ("+ gap 0.3% + stale 90", {"min_gap_pct": 0.3, "stale_exit_bars": 90}),
        ("+ gap 0.3% + last 900", {"min_gap_pct": 0.3, "last_entry_minute": 900}),
        ("+ stale 90 + last 900", {"stale_exit_bars": 90, "last_entry_minute": 900}),
        ("all filters", {"min_gap_pct": 0.3, "stale_exit_bars": 90, "last_entry_minute": 900}),
    ]

    best = None
    for label, extra in variants:
        strat = make_strat(extra)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<30} S={r['sharpe']:>5.2f}  Ret={r['return']:>+7.2%}"
                  f"  DD={r['max_dd']:>6.2%}  T={r['trades']:>4}  Win={r['windows_pos']}")
            if best is None or r["sharpe"] > best["sharpe"]:
                best = r
                best["profile"] = extra

    if best:
        tuned_results[sym] = best
        print(f"  -> Best: {best['label']} (Sharpe {best['sharpe']:.2f})")

# ── Phase C: Portfolio combination with baseline ──
print("\n" + "=" * 100)
print("PHASE C: Portfolio combinations (dev period)")
print("=" * 100)

baseline_daily = {}
for sym in BASELINE_SYMS:
    if results.get(sym) and results[sym].get("daily_returns") is not None:
        baseline_daily[sym] = results[sym]["daily_returns"]

if len(baseline_daily) == 2:
    port_base = pd.DataFrame(baseline_daily).fillna(0).mean(axis=1)
    base_sharpe = (port_base.mean() / port_base.std()) * np.sqrt(252) if port_base.std() > 0 else 0
    print(f"\n  Baseline SPY+QQQ portfolio: Sharpe={base_sharpe:.2f}")

    # Test adding each promising candidate
    for sym in list(tuned_results.keys()) + [s for s in results if s in CANDIDATES and s not in tuned_results and results[s] is not None and results[s]["sharpe"] > 0]:
        r = tuned_results.get(sym) or results.get(sym)
        if r is None or r.get("daily_returns") is None:
            continue

        combined_daily = dict(baseline_daily)
        combined_daily[sym] = r["daily_returns"]
        port_df = pd.DataFrame(combined_daily).fillna(0)
        port_ret = port_df.mean(axis=1)

        if len(port_ret) < 10 or port_ret.std() == 0:
            continue

        new_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252)
        delta = new_sharpe - base_sharpe

        # Cross-correlation with baseline
        corr_spy = r["daily_returns"].corr(baseline_daily.get("SPY", pd.Series()))
        corr_qqq = r["daily_returns"].corr(baseline_daily.get("QQQ", pd.Series()))

        print(f"  + {sym:<6} -> portfolio Sharpe={new_sharpe:.2f} (delta={delta:+.2f})"
              f"  corr(SPY)={corr_spy:.2f}  corr(QQQ)={corr_qqq:.2f}"
              f"  standalone={r['sharpe']:.2f}")

print("\n" + "=" * 100)
print("INSTRUMENT SCREENING COMPLETE")
print("=" * 100)
