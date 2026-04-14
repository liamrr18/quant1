#!/usr/bin/env python3
"""Test improvements v2 — corrected thresholds.

chop_proxy = or_range_pct / atr_pct has mean ~3.7 (SPY), ~4.8 (QQQ).
Higher = OR consumed more of expected range = choppier.
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

data = {}
for sym in ["SPY", "QQQ"]:
    df = get_minute_bars(sym, start, end, use_cache=True)
    df = prepare_features(df)
    data[sym] = df

def test_full(strat, label):
    """Run full-sample + walk-forward."""
    results = []
    wf_results = []
    for sym in ["SPY", "QQQ"]:
        df = data[sym].copy()
        df = strat.generate_signals(df)
        r = run_backtest(df, strat, sym)
        results.append(r)
        wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)
        wf_results.append(wf)

    total_trades = sum(r.num_trades for r in results)
    if total_trades == 0:
        print(f"  {label}: NO TRADES")
        return

    avg_sharpe = np.mean([r.sharpe_ratio for r in results])
    avg_pf = np.mean([r.profit_factor for r in results])
    total_pnl = sum(r.equity_curve.iloc[-1] - 100_000 for r in results)

    oos_sharpe = np.mean([w.sharpe_ratio for w in wf_results])
    oos_pf = np.mean([w.profit_factor for w in wf_results])
    oos_wr = np.mean([w.win_rate for w in wf_results])
    oos_trades = sum(w.total_trades for w in wf_results)
    oos_ret = np.mean([w.total_return for w in wf_results])

    print(f"  {label:50s} | IS: Sharpe={avg_sharpe:.2f} PF={avg_pf:.2f} T={total_trades:4d} PnL=${total_pnl:+,.0f}"
          f" | OOS: Sharpe={oos_sharpe:.2f} PF={oos_pf:.2f} WR={oos_wr*100:.1f}% T={oos_trades:4d} Ret={oos_ret*100:+.2f}%")
    return oos_sharpe

# ── BASELINE ──
print("=" * 140)
print("BASELINE")
print("=" * 140)
baseline = ORBBreakout(range_minutes=15, target_multiple=1.5,
                       min_atr_percentile=25, min_breakout_volume=1.2,
                       last_entry_minute=15*60)
test_full(baseline, "Current system")

# ── TEST 1: CHOP PROXY (or_range_pct / atr_pct) ──
print("\n" + "=" * 140)
print("TEST 1: CHOP PROXY FILTER (skip days where OR already consumed too much of expected range)")
print("=" * 140)

class ORBChop(ORBBreakout):
    name = "orb_chop"
    def __init__(self, max_chop=5.0, **kw):
        super().__init__(**kw)
        self.max_chop = max_chop
    def get_params(self):
        d = super().get_params()
        d["max_chop"] = self.max_chop
        return d
    def generate_signals(self, df):
        df = df.copy()
        # Compute chop proxy per day
        daily = df.groupby("date").agg(
            or_high=("or_high", "first"), or_low=("or_low", "first"), atr_pct=("atr_pct", "first"),
        )
        daily["or_range_pct"] = (daily["or_high"] - daily["or_low"]) / daily["or_low"] * 100
        daily["chop_proxy"] = daily["or_range_pct"] / daily["atr_pct"].replace(0, np.nan)
        df["_chop"] = df["date"].map(daily["chop_proxy"])
        df = super().generate_signals(df)
        df.loc[df["_chop"] > self.max_chop, "signal"] = 0
        return df

for t in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
    strat = ORBChop(max_chop=t, range_minutes=15, target_multiple=1.5,
                    min_atr_percentile=25, min_breakout_volume=1.2, last_entry_minute=15*60)
    test_full(strat, f"Chop proxy <= {t}")

# ── TEST 2: TIME-DECAY EXIT ──
print("\n" + "=" * 140)
print("TEST 2: TIME-DECAY EXIT (exit underwater trades after N bars)")
print("=" * 140)

class ORBTimeDecay(ORBBreakout):
    name = "orb_td"
    def __init__(self, stale_bars=120, **kw):
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

for sb in [30, 60, 90, 120, 180]:
    strat = ORBTimeDecay(stale_bars=sb, range_minutes=15, target_multiple=1.5,
                         min_atr_percentile=25, min_breakout_volume=1.2, last_entry_minute=15*60)
    test_full(strat, f"Time decay {sb} bars")

# ── TEST 3: GAP FILTER ──
print("\n" + "=" * 140)
print("TEST 3: GAP SIZE MINIMUM (skip tiny-gap days)")
print("=" * 140)

class ORBGap(ORBBreakout):
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

for mg in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    strat = ORBGap(min_gap=mg, range_minutes=15, target_multiple=1.5,
                   min_atr_percentile=25, min_breakout_volume=1.2, last_entry_minute=15*60)
    test_full(strat, f"Min gap >= {mg}%")

# ── TEST 4: TARGET MULTIPLE ──
print("\n" + "=" * 140)
print("TEST 4: TARGET MULTIPLE VARIATIONS")
print("=" * 140)

for tm in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
    strat = ORBBreakout(range_minutes=15, target_multiple=tm,
                        min_atr_percentile=25, min_breakout_volume=1.2, last_entry_minute=15*60)
    test_full(strat, f"Target multiple {tm}")

# ── TEST 5: COMBINATIONS ──
print("\n" + "=" * 140)
print("TEST 5: BEST COMBINATIONS")
print("=" * 140)

class ORBCombo(ORBBreakout):
    name = "orb_combo"
    def __init__(self, max_chop=99.0, stale_bars=999, min_gap=0.0, **kw):
        super().__init__(**kw)
        self.max_chop = max_chop
        self.stale_bars = stale_bars
        self.min_gap = min_gap
    def get_params(self):
        d = super().get_params()
        if self.max_chop < 99: d["max_chop"] = self.max_chop
        if self.stale_bars < 999: d["stale"] = self.stale_bars
        if self.min_gap > 0: d["min_gap"] = self.min_gap
        return d
    def generate_signals(self, df):
        df = df.copy()
        # Chop proxy
        if self.max_chop < 99:
            daily = df.groupby("date").agg(
                or_high=("or_high", "first"), or_low=("or_low", "first"), atr_pct=("atr_pct", "first"),
            )
            daily["or_range_pct"] = (daily["or_high"] - daily["or_low"]) / daily["or_low"] * 100
            daily["chop_proxy"] = daily["or_range_pct"] / daily["atr_pct"].replace(0, np.nan)
            df["_chop"] = df["date"].map(daily["chop_proxy"])

        # Gap filter
        if self.min_gap > 0 and "gap_pct" in df.columns:
            df["_small_gap"] = df["gap_pct"].abs() < self.min_gap
        else:
            df["_small_gap"] = False

        # Base signals with time decay
        df["signal"] = 0
        if "or_high" not in df.columns:
            return df
        has_atr = self.min_atr_percentile > 0 and "atr_percentile" in df.columns
        has_vol = self.min_breakout_volume > 0 and "rel_volume" in df.columns
        has_late = self.last_entry_minute > 0
        has_chop = self.max_chop < 99
        has_stale = self.stale_bars < 999
        or_end = 9*60+30+self.range_minutes
        n = len(df)
        signals = np.zeros(n)
        pos = 0; ep = 0.0; eb = 0
        orh = 0.0; orl = 0.0; tgt = 0.0; stp = 0.0
        cdate = None; skip = False
        for i in range(n):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False
                orh = row["or_high"] if pd.notna(row["or_high"]) else 0
                orl = row["or_low"] if pd.notna(row["or_low"]) else 0
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_percentile: skip = True
                if has_chop and not skip:
                    cp = row.get("_chop", 0) if "_chop" in df.columns else 0
                    if pd.notna(cp) and cp > self.max_chop: skip = True
                if not skip and self.min_gap > 0:
                    if row.get("_small_gap", False): skip = True
            if skip or row["minute_of_day"] < or_end:
                signals[i] = 0; continue
            if row["minute_of_day"] > 15*60+30:
                pos = 0; signals[i] = 0; continue
            if orh <= 0 or orl <= 0:
                signals[i] = 0; continue
            rw = orh - orl; rp = rw / orl
            if rp < self.min_range_pct or rp > self.max_range_pct:
                signals[i] = 0; pos = 0; continue
            if pos == 0:
                if has_late and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; continue
                if has_vol:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_breakout_volume:
                        signals[i] = 0; continue
                if row["close"] > orh:
                    pos = 1; ep = row["close"]; eb = i
                    tgt = ep + rw * self.target_multiple; stp = orl
                elif row["close"] < orl:
                    pos = -1; ep = row["close"]; eb = i
                    tgt = ep - rw * self.target_multiple; stp = orh
            elif pos == 1:
                if row["close"] >= tgt or row["close"] <= stp:
                    pos = 0
                elif has_stale and (i - eb) >= self.stale_bars and row["close"] <= ep:
                    pos = 0
            elif pos == -1:
                if row["close"] <= tgt or row["close"] >= stp:
                    pos = 0
                elif has_stale and (i - eb) >= self.stale_bars and row["close"] >= ep:
                    pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int)
        return df

combos = [
    # Best chop + gap
    {"max_chop": 3.5, "stale_bars": 999, "min_gap": 0.0},
    {"max_chop": 4.0, "stale_bars": 999, "min_gap": 0.0},
    {"max_chop": 99, "stale_bars": 60, "min_gap": 0.0},
    {"max_chop": 99, "stale_bars": 90, "min_gap": 0.0},
    {"max_chop": 99, "stale_bars": 999, "min_gap": 0.10},
    {"max_chop": 4.0, "stale_bars": 90, "min_gap": 0.0},
    {"max_chop": 4.0, "stale_bars": 60, "min_gap": 0.0},
    {"max_chop": 3.5, "stale_bars": 90, "min_gap": 0.0},
    {"max_chop": 4.0, "stale_bars": 90, "min_gap": 0.10},
    {"max_chop": 3.5, "stale_bars": 60, "min_gap": 0.10},
]

for c in combos:
    label = f"chop={c['max_chop']} stale={c['stale_bars']} gap={c['min_gap']}"
    strat = ORBCombo(**c, range_minutes=15, target_multiple=1.5,
                     min_atr_percentile=25, min_breakout_volume=1.2, last_entry_minute=15*60)
    test_full(strat, label)

# ── TEST 6: IWM/DIA robustness ──
print("\n" + "=" * 140)
print("TEST 6: MULTI-SYMBOL ROBUSTNESS (IWM, DIA)")
print("=" * 140)
for sym in ["IWM", "DIA"]:
    try:
        df = get_minute_bars(sym, start, end, use_cache=True)
        df = prepare_features(df)
        df = baseline.generate_signals(df)
        r = run_backtest(df, baseline, sym)
        wf = walk_forward(df, baseline, sym, train_days=60, test_days=20, step_days=20)
        print(f"  {sym}: IS Sharpe={r.sharpe_ratio:.2f} PF={r.profit_factor:.2f} T={r.num_trades} "
              f"Ret={r.total_return*100:+.2f}% | OOS Sharpe={wf.sharpe_ratio:.2f} PF={wf.profit_factor:.2f} "
              f"WR={wf.win_rate*100:.1f}% T={wf.total_trades}")
    except Exception as e:
        print(f"  {sym}: ERROR - {e}")

print("\n" + "=" * 140)
print("DONE")
print("=" * 140)
