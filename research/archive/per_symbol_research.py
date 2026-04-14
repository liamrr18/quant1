#!/usr/bin/env python3
"""Per-symbol research: test each modification independently per ticker.

Tests per symbol: baseline, time decay (60/90/120), gap filter (0.2/0.3/0.5),
target multiple (1.25/1.5/1.75/2.0), last_entry (14:00/14:30/15:00),
ATR filter (0/25/35), volume filter (0/1.2/1.5).

Reports walk-forward OOS results per symbol per variant.
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

ET = pytz.timezone("America/New_York")
start = datetime(2025, 1, 2, tzinfo=ET)
end = datetime(2026, 4, 4, tzinfo=ET)

# ── Load data ──
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
data = {}
for sym in SYMBOLS:
    df = get_minute_bars(sym, start, end, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"Loaded {sym}: {len(df)} bars, {df['date'].nunique()} days")


class ORBTimeDecay(ORBBreakout):
    """ORB with time-based stale exit."""
    name = "orb_td"
    def __init__(self, stale_bars=90, **kw):
        super().__init__(**kw)
        self.stale_bars = stale_bars
    def get_params(self):
        d = super().get_params()
        d["stale"] = self.stale_bars
        return d
    def generate_signals(self, df):
        df = df.copy()
        df["signal"] = 0
        if "or_high" not in df.columns:
            return df
        has_atr = self.min_atr_percentile > 0 and "atr_percentile" in df.columns
        has_vol = self.min_breakout_volume > 0 and "rel_volume" in df.columns
        has_late = self.last_entry_minute > 0
        or_end = 9*60+30+self.range_minutes
        n = len(df)
        signals = np.zeros(n)
        pos = 0; ep = 0.0; eb = 0
        orh = 0.0; orl = 0.0; tgt = 0.0; stp = 0.0
        cdate = None; skip = False
        for i in range(n):
            r = df.iloc[i]; d = r["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False
                orh = r["or_high"] if pd.notna(r["or_high"]) else 0
                orl = r["or_low"] if pd.notna(r["or_low"]) else 0
                if has_atr:
                    ap = r.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_percentile: skip = True
            if skip or r["minute_of_day"] < or_end:
                signals[i] = 0; continue
            if r["minute_of_day"] > 15*60+30:
                pos = 0; signals[i] = 0; continue
            if orh <= 0 or orl <= 0:
                signals[i] = 0; continue
            rw = orh - orl; rp = rw / orl
            if rp < self.min_range_pct or rp > self.max_range_pct:
                signals[i] = 0; pos = 0; continue
            if pos == 0:
                if has_late and r["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; continue
                if has_vol:
                    rv = r.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_breakout_volume:
                        signals[i] = 0; continue
                if r["close"] > orh:
                    pos = 1; ep = r["close"]; eb = i
                    tgt = ep + rw * self.target_multiple; stp = orl
                elif r["close"] < orl:
                    pos = -1; ep = r["close"]; eb = i
                    tgt = ep - rw * self.target_multiple; stp = orh
            elif pos == 1:
                if r["close"] >= tgt or r["close"] <= stp:
                    pos = 0
                elif (i - eb) >= self.stale_bars and r["close"] <= ep:
                    pos = 0
            elif pos == -1:
                if r["close"] <= tgt or r["close"] >= stp:
                    pos = 0
                elif (i - eb) >= self.stale_bars and r["close"] >= ep:
                    pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int)
        return df


class ORBGap(ORBBreakout):
    """ORB with minimum gap size filter."""
    name = "orb_gap"
    def __init__(self, min_gap=0.2, **kw):
        super().__init__(**kw)
        self.min_gap = min_gap
    def get_params(self):
        d = super().get_params()
        d["min_gap"] = self.min_gap
        return d
    def generate_signals(self, df):
        df = super().generate_signals(df)
        if "gap_pct" in df.columns:
            df.loc[df["gap_pct"].abs() < self.min_gap, "signal"] = 0
        return df


class ORBGapTimeDecay(ORBTimeDecay):
    """ORB with gap filter + time decay."""
    name = "orb_gap_td"
    def __init__(self, min_gap=0.2, **kw):
        super().__init__(**kw)
        self.min_gap = min_gap
    def get_params(self):
        d = super().get_params()
        d["min_gap"] = self.min_gap
        return d
    def generate_signals(self, df):
        df = super().generate_signals(df)
        if "gap_pct" in df.columns:
            df.loc[df["gap_pct"].abs() < self.min_gap, "signal"] = 0
        return df


def test_symbol(sym, strat, label):
    """Run IS + OOS for one symbol."""
    df = data[sym].copy()
    df = strat.generate_signals(df)
    r = run_backtest(df, strat, sym)
    wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)
    return {
        "sym": sym, "label": label,
        "is_sharpe": r.sharpe_ratio, "is_pf": r.profit_factor, "is_trades": r.num_trades,
        "is_ret": r.total_return, "is_pnl": r.equity_curve.iloc[-1] - 100_000,
        "oos_sharpe": wf.sharpe_ratio, "oos_pf": wf.profit_factor, "oos_wr": wf.win_rate,
        "oos_trades": wf.total_trades, "oos_ret": wf.total_return, "oos_dd": wf.max_drawdown,
        "oos_windows": wf.num_windows,
        # Per-window sharpes for consistency check
        "window_sharpes": [r.sharpe_ratio for r in wf.oos_results],
    }


def print_result(r):
    pct_pos_windows = sum(1 for s in r["window_sharpes"] if s > 0) / max(len(r["window_sharpes"]), 1) * 100
    print(f"  {r['label']:50s} | IS: S={r['is_sharpe']:.2f} PF={r['is_pf']:.2f} T={r['is_trades']:4d}"
          f" | OOS: S={r['oos_sharpe']:.2f} PF={r['oos_pf']:.2f} WR={r['oos_wr']*100:.1f}% "
          f"T={r['oos_trades']:4d} Ret={r['oos_ret']*100:+.2f}% DD={r['oos_dd']*100:.1f}% "
          f"Win%={pct_pos_windows:.0f}%({len(r['window_sharpes'])}w)")


# ══════════════════════════════════════════════════════════════════
# Define all strategy variants to test
# ══════════════════════════════════════════════════════════════════
BASE_KW = dict(range_minutes=15, min_range_pct=0.001, max_range_pct=0.008)

def make_variants():
    """Generate all strategy variants to test."""
    variants = []

    # 1. BASELINE (current production)
    variants.append(("Baseline (current prod)", ORBBreakout(
        target_multiple=1.5, min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60, **BASE_KW)))

    # 2. BASELINE with no filters (pure ORB)
    variants.append(("Pure ORB (no filters)", ORBBreakout(
        target_multiple=1.5, min_atr_percentile=0, min_breakout_volume=0,
        last_entry_minute=0, **BASE_KW)))

    # 3. TIME DECAY variants
    for sb in [60, 90, 120]:
        variants.append((f"Time decay {sb}b", ORBTimeDecay(
            stale_bars=sb, target_multiple=1.5,
            min_atr_percentile=25, min_breakout_volume=1.2,
            last_entry_minute=15*60, **BASE_KW)))

    # 4. GAP FILTER variants
    for mg in [0.2, 0.3, 0.5]:
        variants.append((f"Gap >= {mg}%", ORBGap(
            min_gap=mg, target_multiple=1.5,
            min_atr_percentile=25, min_breakout_volume=1.2,
            last_entry_minute=15*60, **BASE_KW)))

    # 5. TARGET MULTIPLE variants
    for tm in [1.0, 1.25, 1.75, 2.0]:
        variants.append((f"Target {tm}x", ORBBreakout(
            target_multiple=tm, min_atr_percentile=25, min_breakout_volume=1.2,
            last_entry_minute=15*60, **BASE_KW)))

    # 6. LAST ENTRY TIME variants
    for lem in [14*60, 14*60+30, 13*60+30]:
        h = lem // 60; m = lem % 60
        variants.append((f"Last entry {h}:{m:02d}", ORBBreakout(
            target_multiple=1.5, min_atr_percentile=25, min_breakout_volume=1.2,
            last_entry_minute=lem, **BASE_KW)))

    # 7. ATR FILTER variants
    for atr_p in [0, 15, 35]:
        variants.append((f"ATR pctl >= {atr_p}", ORBBreakout(
            target_multiple=1.5, min_atr_percentile=atr_p, min_breakout_volume=1.2,
            last_entry_minute=15*60, **BASE_KW)))

    # 8. VOLUME FILTER variants
    for vr in [0, 1.0, 1.5]:
        variants.append((f"Vol ratio >= {vr}", ORBBreakout(
            target_multiple=1.5, min_atr_percentile=25, min_breakout_volume=vr,
            last_entry_minute=15*60, **BASE_KW)))

    # 9. COMBINED: gap + time decay (best from universal research)
    for mg, sb in [(0.2, 90), (0.3, 90), (0.3, 120)]:
        variants.append((f"Gap>={mg}% + TD{sb}", ORBGapTimeDecay(
            min_gap=mg, stale_bars=sb, target_multiple=1.5,
            min_atr_percentile=25, min_breakout_volume=1.2,
            last_entry_minute=15*60, **BASE_KW)))

    return variants


variants = make_variants()
print(f"\nTesting {len(variants)} variants across {len(SYMBOLS)} symbols")
print(f"Total tests: {len(variants) * len(SYMBOLS)}\n")

# ══════════════════════════════════════════════════════════════════
# Run all tests
# ══════════════════════════════════════════════════════════════════
all_results = []

for sym in SYMBOLS:
    print("=" * 160)
    print(f"  {sym}")
    print("=" * 160)
    for label, strat in variants:
        try:
            r = test_symbol(sym, strat, label)
            all_results.append(r)
            print_result(r)
        except Exception as e:
            print(f"  {label:50s} | ERROR: {e}")

# ══════════════════════════════════════════════════════════════════
# Summary: Best per symbol
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 160)
print("SUMMARY: BEST VARIANTS PER SYMBOL (ranked by OOS Sharpe)")
print("=" * 160)

for sym in SYMBOLS:
    sym_results = [r for r in all_results if r["sym"] == sym]
    sym_results.sort(key=lambda r: r["oos_sharpe"], reverse=True)
    print(f"\n  {sym} (top 5):")
    for r in sym_results[:5]:
        pct_pos = sum(1 for s in r["window_sharpes"] if s > 0) / max(len(r["window_sharpes"]), 1) * 100
        print(f"    {r['label']:50s} OOS: S={r['oos_sharpe']:.2f} PF={r['oos_pf']:.2f} "
              f"WR={r['oos_wr']*100:.1f}% T={r['oos_trades']:4d} Win%={pct_pos:.0f}%")

    baseline = [r for r in sym_results if "Baseline" in r["label"]][0]
    best = sym_results[0]
    print(f"    → Baseline OOS Sharpe: {baseline['oos_sharpe']:.2f}")
    print(f"    → Best OOS Sharpe:     {best['oos_sharpe']:.2f} ({best['label']})")
    if best["oos_sharpe"] > baseline["oos_sharpe"] + 0.1:
        print(f"    → IMPROVEMENT: +{best['oos_sharpe'] - baseline['oos_sharpe']:.2f}")
    else:
        print(f"    → BASELINE IS BEST (or negligible improvement)")

print("\n" + "=" * 160)
print("RECOMMENDATION")
print("=" * 160)
for sym in SYMBOLS:
    sym_results = [r for r in all_results if r["sym"] == sym]
    baseline = [r for r in sym_results if "Baseline" in r["label"]][0]
    best = sorted(sym_results, key=lambda r: r["oos_sharpe"], reverse=True)[0]

    if best["oos_sharpe"] < 0.3:
        print(f"  {sym}: EXCLUDE (best OOS Sharpe = {best['oos_sharpe']:.2f})")
    elif best["oos_sharpe"] <= baseline["oos_sharpe"] + 0.15:
        print(f"  {sym}: USE BASELINE (OOS Sharpe = {baseline['oos_sharpe']:.2f})")
    else:
        print(f"  {sym}: USE '{best['label']}' (OOS Sharpe = {best['oos_sharpe']:.2f} vs baseline {baseline['oos_sharpe']:.2f})")

print("\nDONE")
