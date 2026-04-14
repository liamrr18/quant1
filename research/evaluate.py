#!/usr/bin/env python3
"""Final evaluation: baseline vs per-symbol portfolio.

Runs walk-forward OOS for:
1. Old baseline (universal orb_filtered) on each symbol
2. New per-symbol profiles on each symbol
3. Portfolio-level comparison
4. Parameter stability tests

Also tests DIA exclusion.
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

data = {}
for sym in SYMBOLS + ["DIA"]:
    df = get_minute_bars(sym, start, end, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"Loaded {sym}: {len(df)} bars")


def make_strat(sym=None, profile=None):
    params = dict(ORB_SHARED_DEFAULTS)
    if profile is not None:
        params.update(profile)
    elif sym and sym in SYMBOL_PROFILES:
        params.update(SYMBOL_PROFILES[sym])
    return ORBBreakout(**params)


def run_sym(sym, strat):
    df = data[sym].copy()
    df = strat.generate_signals(df)
    r = run_backtest(df, strat, sym)
    wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)
    ws = [wr.sharpe_ratio for wr in wf.oos_results]
    pct_pos = sum(1 for s in ws if s > 0) / max(len(ws), 1) * 100
    return r, wf, ws, pct_pos


# ═════════════��══════════════════════════════════���═════════════
print("\n" + "=" * 120)
print("SECTION 1: BASELINE vs PER-SYMBOL (walk-forward OOS)")
print("=" * 120)

baseline_strat = ORBBreakout(
    range_minutes=15, target_multiple=1.5,
    min_atr_percentile=25, min_breakout_volume=1.2, last_entry_minute=15*60,
)

for sym in SYMBOLS:
    print(f"\n  {sym}:")
    # Baseline
    r_b, wf_b, ws_b, pp_b = run_sym(sym, baseline_strat)
    print(f"    Baseline:    IS S={r_b.sharpe_ratio:.2f} PF={r_b.profit_factor:.2f} T={r_b.num_trades:4d}"
          f" | OOS S={wf_b.sharpe_ratio:.2f} PF={wf_b.profit_factor:.2f} WR={wf_b.win_rate*100:.1f}%"
          f" T={wf_b.total_trades:4d} Ret={wf_b.total_return*100:+.2f}% Win%={pp_b:.0f}%({len(ws_b)}w)")

    # Per-symbol
    strat = make_strat(sym)
    r_p, wf_p, ws_p, pp_p = run_sym(sym, strat)
    print(f"    Per-symbol:  IS S={r_p.sharpe_ratio:.2f} PF={r_p.profit_factor:.2f} T={r_p.num_trades:4d}"
          f" | OOS S={wf_p.sharpe_ratio:.2f} PF={wf_p.profit_factor:.2f} WR={wf_p.win_rate*100:.1f}%"
          f" T={wf_p.total_trades:4d} Ret={wf_p.total_return*100:+.2f}% Win%={pp_p:.0f}%({len(ws_p)}w)")

    delta = wf_p.sharpe_ratio - wf_b.sharpe_ratio
    print(f"    -> OOS Sharpe delta: {delta:+.2f}  Profile: {strat.get_params()}")

# ═════════════��════════════════════════════════════════════════
print("\n\n" + "=" * 120)
print("SECTION 2: PARAMETER STABILITY")
print("=" * 120)

# SPY gap filter
print("\n  SPY gap filter stability:")
for gap in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    strat = make_strat(profile={"min_gap_pct": gap, "min_atr_percentile": 25,
                                "min_breakout_volume": 1.2, "last_entry_minute": 900})
    _, wf, _, pp = run_sym("SPY", strat)
    print(f"    gap={gap:.2f}: OOS S={wf.sharpe_ratio:.2f} PF={wf.profit_factor:.2f}"
          f" T={wf.total_trades:4d} Ret={wf.total_return*100:+.2f}% Win%={pp:.0f}%")

# QQQ stale exit
print("\n  QQQ stale exit stability:")
for sb in [60, 75, 90, 105, 120, 150]:
    strat = make_strat(profile={"stale_exit_bars": sb, "min_atr_percentile": 25,
                                "min_breakout_volume": 1.2, "last_entry_minute": 900})
    _, wf, _, pp = run_sym("QQQ", strat)
    print(f"    stale={sb:3d}: OOS S={wf.sharpe_ratio:.2f} PF={wf.profit_factor:.2f}"
          f" T={wf.total_trades:4d} Ret={wf.total_return*100:+.2f}% Win%={pp:.0f}%")

# IWM filter levels
print("\n  IWM filter removal stability:")
cfgs = [
    ("All filters", {"min_atr_percentile": 25, "min_breakout_volume": 1.2, "last_entry_minute": 900}),
    ("No ATR", {"min_atr_percentile": 0, "min_breakout_volume": 1.2, "last_entry_minute": 900}),
    ("No vol", {"min_atr_percentile": 25, "min_breakout_volume": 0, "last_entry_minute": 900}),
    ("No late cutoff", {"min_atr_percentile": 25, "min_breakout_volume": 1.2, "last_entry_minute": 0}),
    ("No ATR+vol", {"min_atr_percentile": 0, "min_breakout_volume": 0, "last_entry_minute": 900}),
    ("Pure ORB", {"min_atr_percentile": 0, "min_breakout_volume": 0, "last_entry_minute": 0}),
]
for label, prof in cfgs:
    strat = make_strat(profile=prof)
    _, wf, _, pp = run_sym("IWM", strat)
    print(f"    {label:18s}: OOS S={wf.sharpe_ratio:.2f} PF={wf.profit_factor:.2f}"
          f" T={wf.total_trades:4d} Ret={wf.total_return*100:+.2f}% Win%={pp:.0f}%")

# ══════════════════════════���═══════════════════════════════════
print("\n\n" + "=" * 120)
print("SECTION 3: PORTFOLIO COMPARISON")
print("=" * 120)

print("\n  BASELINE PORTFOLIO:")
b_sharpes, b_rets, b_trades_total = [], [], 0
for sym in SYMBOLS:
    _, wf, _, _ = run_sym(sym, baseline_strat)
    b_sharpes.append(wf.sharpe_ratio)
    b_rets.append(wf.total_return)
    b_trades_total += wf.total_trades
    print(f"    {sym}: OOS S={wf.sharpe_ratio:.2f} Ret={wf.total_return*100:+.2f}% T={wf.total_trades}")
print(f"    AVG: S={np.mean(b_sharpes):.2f} Ret={np.mean(b_rets)*100:+.2f}% Total T={b_trades_total}")

print("\n  PER-SYMBOL PORTFOLIO:")
p_sharpes, p_rets, p_trades_total = [], [], 0
for sym in SYMBOLS:
    strat = make_strat(sym)
    _, wf, _, _ = run_sym(sym, strat)
    p_sharpes.append(wf.sharpe_ratio)
    p_rets.append(wf.total_return)
    p_trades_total += wf.total_trades
    print(f"    {sym}: OOS S={wf.sharpe_ratio:.2f} Ret={wf.total_return*100:+.2f}% T={wf.total_trades}")
print(f"    AVG: S={np.mean(p_sharpes):.2f} Ret={np.mean(p_rets)*100:+.2f}% Total T={p_trades_total}")

print(f"\n  DELTA: avg Sharpe {np.mean(b_sharpes):.2f} -> {np.mean(p_sharpes):.2f} ({np.mean(p_sharpes)-np.mean(b_sharpes):+.2f})")
print(f"  DELTA: avg Ret {np.mean(b_rets)*100:+.2f}% -> {np.mean(p_rets)*100:+.2f}% ({(np.mean(p_rets)-np.mean(b_rets))*100:+.2f}%)")

# ═══════════════════════════════════════════���══════════════════
print("\n\n" + "=" * 120)
print("SECTION 4: DIA EXCLUSION")
print("=" * 120)
_, dia_wf, _, dia_pp = run_sym("DIA", baseline_strat)
print(f"  DIA: OOS S={dia_wf.sharpe_ratio:.2f} PF={dia_wf.profit_factor:.2f}"
      f" WR={dia_wf.win_rate*100:.1f}% T={dia_wf.total_trades} Win%={dia_pp:.0f}%")
if dia_wf.sharpe_ratio < 0.3:
    print(f"  -> EXCLUDE DIA (OOS Sharpe {dia_wf.sharpe_ratio:.2f} < 0.3 threshold)")

print("\n" + "=" * 120)
print("EVALUATION COMPLETE")
print("=" * 120)
