#!/usr/bin/env python3
"""Test improvements on DEVELOPMENT PERIOD ONLY (Jan-Nov 2025).
DO NOT touch locked OOS (Dec 2025+).

Tests:
1. IWM filter/exit variants (fix the -0.74 OOS collapse)
2. Trailing stop variants (all symbols)
3. Universal stale exit
4. Portfolio compositions
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
from trading.backtest.walkforward import walk_forward
from trading.config import SYMBOL_PROFILES, ORB_SHARED_DEFAULTS

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)  # Development period only

SYMBOLS = ["SPY", "QQQ", "IWM"]

# ── Load data (dev period only) ──
print("Loading data (development period: Jan-Nov 2025)...")
data = {}
for sym in SYMBOLS:
    df = get_minute_bars(sym, DATA_START, DEV_END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"  {sym}: {len(df)} bars, {df['date'].nunique()} days")


def make_strat(profile):
    """Create ORB strategy from profile dict."""
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(profile)
    return ORBBreakout(**params)


def eval_symbol(sym, profile, label=""):
    """Run walk-forward on dev period, return concatenated OOS metrics."""
    strat = make_strat(profile)
    df = data[sym].copy()
    df = strat.generate_signals(df)
    wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)

    # Extract concatenated OOS daily returns
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

    # Window consistency
    window_sharpes = [r.sharpe_ratio for r in wf.oos_results]
    pos_windows = sum(1 for s in window_sharpes if s > 0)

    return {
        "sym": sym, "label": label, "sharpe": sharpe, "return": total_ret,
        "max_dd": max_dd, "trades": wf.total_trades,
        "win_rate": wf.win_rate, "pf": wf.profit_factor,
        "windows_pos": f"{pos_windows}/{len(window_sharpes)}",
        "daily_returns": combined,
    }


def portfolio_metrics(returns_dict):
    """Compute portfolio metrics from dict of symbol daily returns."""
    df = pd.DataFrame(returns_dict).fillna(0)
    port = df.mean(axis=1)
    if len(port) < 10 or port.std() == 0:
        return {"sharpe": 0, "return": 0, "max_dd": 0}
    sharpe = (port.mean() / port.std()) * np.sqrt(252)
    total_ret = (1 + port).prod() - 1
    cum = (1 + port).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    down = port[port < 0]
    sortino = (port.mean() / down.std()) * np.sqrt(252) if len(down) > 2 and down.std() > 0 else 0
    return {"sharpe": sharpe, "return": total_ret, "max_dd": max_dd, "sortino": sortino,
            "daily_returns": port}


def print_result(r):
    if r is None:
        print("    -> No data")
        return
    print(f"    {r['sym']:<4} {r['label']:<30} S={r['sharpe']:>5.2f}"
          f"  Ret={r['return']:>+7.2%}  DD={r['max_dd']:>6.2%}"
          f"  T={r['trades']:>4}  WR={r['win_rate']:.1%}  PF={r['pf']:.2f}"
          f"  Win={r['windows_pos']}")


# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TEST 1: IWM FILTER/EXIT VARIANTS (dev period walk-forward)")
print("=" * 100)

# Current IWM profile: pure ORB (no filters)
iwm_current = {"min_atr_percentile": 0, "min_breakout_volume": 0, "last_entry_minute": 0}

iwm_variants = [
    ("Current (pure ORB)", iwm_current),
    ("+ stale 60", {**iwm_current, "stale_exit_bars": 60}),
    ("+ stale 90", {**iwm_current, "stale_exit_bars": 90}),
    ("+ stale 120", {**iwm_current, "stale_exit_bars": 120}),
    ("+ ATR 25", {**iwm_current, "min_atr_percentile": 25}),
    ("+ vol 1.2", {**iwm_current, "min_breakout_volume": 1.2}),
    ("+ last_entry 900", {**iwm_current, "last_entry_minute": 900}),
    ("+ stale 90 + ATR 25", {**iwm_current, "stale_exit_bars": 90, "min_atr_percentile": 25}),
    ("+ stale 90 + vol 1.2", {**iwm_current, "stale_exit_bars": 90, "min_breakout_volume": 1.2}),
    ("+ stale 90 + last 900", {**iwm_current, "stale_exit_bars": 90, "last_entry_minute": 900}),
    ("Baseline (all filters)", {}),  # uses ORB_SHARED_DEFAULTS
    ("Baseline + stale 90", {"stale_exit_bars": 90}),
]

iwm_results = []
for label, profile in iwm_variants:
    r = eval_symbol("IWM", profile, label)
    print_result(r)
    if r:
        iwm_results.append(r)

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TEST 2: TRAILING STOP VARIANTS (all symbols, dev period)")
print("=" * 100)

trail_variants = [
    ("Current (fixed target)", {}),
    ("BE at 0.5x range", {"breakeven_trigger": 0.5}),
    ("BE at 0.75x range", {"breakeven_trigger": 0.75}),
    ("Trail at 0.75x, off 0.5", {"trail_trigger": 0.75, "trail_offset": 0.5}),
    ("Trail at 1.0x, off 0.5", {"trail_trigger": 1.0, "trail_offset": 0.5}),
    ("Trail at 0.75x, off 0.75", {"trail_trigger": 0.75, "trail_offset": 0.75}),
    ("No target + trail 0.75/0.5", {"target_multiple": 99.0, "trail_trigger": 0.75, "trail_offset": 0.5}),
    ("Target 2.0 + trail 1.0/0.5", {"target_multiple": 2.0, "trail_trigger": 1.0, "trail_offset": 0.5}),
]

for sym in SYMBOLS:
    base_profile = dict(SYMBOL_PROFILES.get(sym, {}))
    print(f"\n  {sym}:")
    for label, extra in trail_variants:
        profile = {**base_profile, **extra}
        r = eval_symbol(sym, profile, label)
        print_result(r)

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TEST 3: UNIVERSAL STALE EXIT (all symbols)")
print("=" * 100)

stale_vals = [0, 60, 90, 120, 150]
for sym in SYMBOLS:
    base_profile = dict(SYMBOL_PROFILES.get(sym, {}))
    print(f"\n  {sym} (base profile + stale exit):")
    for sb in stale_vals:
        profile = {**base_profile, "stale_exit_bars": sb}
        label = f"stale={sb}" if sb > 0 else "no stale"
        r = eval_symbol(sym, profile, label)
        print_result(r)

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TEST 4: PORTFOLIO COMPOSITIONS (dev period walk-forward)")
print("=" * 100)

# Collect best per-symbol configs for portfolio testing
# We'll test multiple portfolio compositions with current AND improved configs

configs_to_portfolio = {
    "Current profiles": {
        "SPY": SYMBOL_PROFILES.get("SPY", {}),
        "QQQ": SYMBOL_PROFILES.get("QQQ", {}),
        "IWM": SYMBOL_PROFILES.get("IWM", {}),
    },
    "IWM + stale 90": {
        "SPY": SYMBOL_PROFILES.get("SPY", {}),
        "QQQ": SYMBOL_PROFILES.get("QQQ", {}),
        "IWM": {**SYMBOL_PROFILES.get("IWM", {}), "stale_exit_bars": 90},
    },
    "IWM + stale 90 + ATR 25": {
        "SPY": SYMBOL_PROFILES.get("SPY", {}),
        "QQQ": SYMBOL_PROFILES.get("QQQ", {}),
        "IWM": {**SYMBOL_PROFILES.get("IWM", {}), "stale_exit_bars": 90, "min_atr_percentile": 25},
    },
    "All + trail 0.75/0.5": {
        "SPY": {**SYMBOL_PROFILES.get("SPY", {}), "trail_trigger": 0.75, "trail_offset": 0.5},
        "QQQ": {**SYMBOL_PROFILES.get("QQQ", {}), "trail_trigger": 0.75, "trail_offset": 0.5},
        "IWM": {**SYMBOL_PROFILES.get("IWM", {}), "trail_trigger": 0.75, "trail_offset": 0.5},
    },
    "IWM stale + all trail": {
        "SPY": {**SYMBOL_PROFILES.get("SPY", {}), "trail_trigger": 0.75, "trail_offset": 0.5},
        "QQQ": {**SYMBOL_PROFILES.get("QQQ", {}), "trail_trigger": 0.75, "trail_offset": 0.5},
        "IWM": {**SYMBOL_PROFILES.get("IWM", {}), "stale_exit_bars": 90, "trail_trigger": 0.75, "trail_offset": 0.5},
    },
    "SPY+QQQ only (current)": {
        "SPY": SYMBOL_PROFILES.get("SPY", {}),
        "QQQ": SYMBOL_PROFILES.get("QQQ", {}),
    },
    "SPY+QQQ + trail": {
        "SPY": {**SYMBOL_PROFILES.get("SPY", {}), "trail_trigger": 0.75, "trail_offset": 0.5},
        "QQQ": {**SYMBOL_PROFILES.get("QQQ", {}), "trail_trigger": 0.75, "trail_offset": 0.5},
    },
}

print(f"\n  {'Config':<30} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>7}")
print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")

for config_name, sym_profiles in configs_to_portfolio.items():
    sym_daily = {}
    total_trades = 0
    for sym, profile in sym_profiles.items():
        r = eval_symbol(sym, profile, config_name)
        if r:
            sym_daily[sym] = r["daily_returns"]
            total_trades += r["trades"]

    if len(sym_daily) >= 2:
        pm = portfolio_metrics(sym_daily)
        print(f"  {config_name:<30} {pm['sharpe']:>7.2f} {pm['sortino']:>8.2f}"
              f" {pm['return']:>+8.2%} {pm['max_dd']:>7.2%}  T={total_trades}")
    elif len(sym_daily) == 1:
        sym = list(sym_daily.keys())[0]
        ret = sym_daily[sym]
        s = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
        t = (1 + ret).prod() - 1
        print(f"  {config_name:<30} {s:>7.2f} {'':>8} {t:>+8.2%}  T={total_trades}")

print("\n" + "=" * 100)
print("IMPROVEMENT TESTING COMPLETE")
print("=" * 100)
