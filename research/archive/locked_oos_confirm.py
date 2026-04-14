#!/usr/bin/env python3
"""Experiment 14 Phase 2: Locked OOS confirmation and portfolio combination.

Tests top dev-period candidates on held-out Dec 2025 - Apr 2026 data.
Configs frozen from dev period. Results NOT used to iterate.
"""

import sys, os, io, logging, warnings
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.base import Strategy
from trading.backtest.engine import run_backtest
from trading.config import ORB_SHARED_DEFAULTS, SYMBOL_PROFILES
from trading.strategies.orb import ORBBreakout

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

# Inline strategy classes to avoid import issues with module-level stdout wrapping

class VWAPTrend(Strategy):
    name = "vwap_trend"
    def __init__(self, confirm_bars=10, min_rel_vol=1.0, min_vwap_dist_pct=0.05,
                 stale_bars=120, last_entry_minute=900, min_atr_pctl=0):
        self.confirm_bars = confirm_bars; self.min_rel_vol = min_rel_vol
        self.min_vwap_dist_pct = min_vwap_dist_pct; self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute; self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"conf": self.confirm_bars, "vol": self.min_rel_vol,
                "dist": self.min_vwap_dist_pct, "stale": self.stale_bars}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "intraday_vwap" not in df.columns: return df
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        has_vol = self.min_rel_vol > 0 and "rel_volume" in df.columns
        signals = np.zeros(len(df)); pos = 0; ebar = 0
        ba = 0; bb = 0; cdate = None; skip = False
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; ba = 0; bb = 0
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            vwap = row.get("intraday_vwap", 0)
            if pd.isna(vwap) or vwap <= 0: signals[i] = pos; continue
            p = row["close"]; dp = (p - vwap) / vwap * 100
            if p > vwap: ba += 1; bb = 0
            elif p < vwap: bb += 1; ba = 0
            else: ba = 0; bb = 0
            if pos == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                vok = True
                if has_vol:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_rel_vol: vok = False
                if ba >= self.confirm_bars and dp >= self.min_vwap_dist_pct and vok: pos = 1; ebar = i
                elif bb >= self.confirm_bars and dp <= -self.min_vwap_dist_pct and vok: pos = -1; ebar = i
            elif pos == 1:
                if p < vwap: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if p > vwap: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


class GapContinuation(Strategy):
    name = "gap_continuation"
    def __init__(self, min_gap_pct=0.3, max_gap_pct=2.5, confirm_minutes=15,
                 target_pct=0.3, stop_pct=0.15, stale_bars=180,
                 last_entry_minute=720, min_atr_pctl=0):
        self.min_gap_pct = min_gap_pct; self.max_gap_pct = max_gap_pct
        self.confirm_minutes = confirm_minutes; self.target_pct = target_pct
        self.stop_pct = stop_pct; self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute; self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"gap": self.min_gap_pct, "tgt": self.target_pct,
                "stp": self.stop_pct, "conf": self.confirm_minutes}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "gap_pct" not in df.columns or "prev_close" not in df.columns: return df
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        ce = 9*60+30+self.confirm_minutes
        signals = np.zeros(len(df)); pos = 0; ep = 0.0; ebar = 0
        tgt = 0.0; stp = 0.0; cdate = None; dg = 0.0; pc = 0.0; skip = False; gh = False
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; gh = False
                dg = row.get("gap_pct", 0); pc = row.get("prev_close", 0)
                if pd.isna(dg) or pd.isna(pc) or pc <= 0: skip = True
                elif abs(dg) < self.min_gap_pct or abs(dg) > self.max_gap_pct: skip = True
                if has_atr and not skip:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            if row["minute_of_day"] < ce:
                if dg > 0 and row["low"] <= pc: skip = True
                elif dg < 0 and row["high"] >= pc: skip = True
                signals[i] = 0; continue
            if not gh and row["minute_of_day"] >= ce: gh = True
            if pos == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                if gh:
                    if dg > 0:
                        pos = 1; ep = row["close"]; ebar = i
                        tgt = ep*(1+self.target_pct/100); stp = ep*(1-self.stop_pct/100); gh = False
                    elif dg < 0:
                        pos = -1; ep = row["close"]; ebar = i
                        tgt = ep*(1-self.target_pct/100); stp = ep*(1+self.stop_pct/100); gh = False
            elif pos == 1:
                if row["close"] >= tgt or row["close"] <= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if row["close"] <= tgt or row["close"] >= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


class MomentumScore(Strategy):
    name = "momentum_score"
    def __init__(self, entry_threshold=3, exit_threshold=1, rsi_bull=55, rsi_bear=45,
                 min_rel_vol=1.0, stale_bars=120, last_entry_minute=900,
                 min_atr_pctl=0, cooldown_bars=10):
        self.entry_threshold = entry_threshold; self.exit_threshold = exit_threshold
        self.rsi_bull = rsi_bull; self.rsi_bear = rsi_bear; self.min_rel_vol = min_rel_vol
        self.stale_bars = stale_bars; self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl; self.cooldown_bars = cooldown_bars
    def get_params(self):
        return {"thr": self.entry_threshold, "exit": self.exit_threshold, "stale": self.stale_bars}
    def _score(self, row):
        s = 0
        e8 = row.get("ema_8", 0); e21 = row.get("ema_21", 0)
        if pd.notna(e8) and pd.notna(e21) and e21 > 0:
            if e8 > e21: s += 1
            elif e8 < e21: s -= 1
        v = row.get("intraday_vwap", 0)
        if pd.notna(v) and v > 0:
            if row["close"] > v: s += 1
            elif row["close"] < v: s -= 1
        r = row.get("rsi", 50)
        if pd.notna(r):
            if r > self.rsi_bull: s += 1
            elif r < self.rsi_bear: s -= 1
        rv = row.get("rel_volume", 1.0)
        if pd.notna(rv) and rv >= self.min_rel_vol:
            if s > 0: s += 1
            elif s < 0: s -= 1
        return s
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); pos = 0; ebar = 0
        cdate = None; skip = False; bse = 999
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; bse = 999
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            sc = self._score(row)
            if pos == 0:
                bse += 1
                if bse < self.cooldown_bars: signals[i] = 0; continue
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                if sc >= self.entry_threshold: pos = 1; ebar = i
                elif sc <= -self.entry_threshold: pos = -1; ebar = i
            elif pos == 1:
                if sc <= self.exit_threshold: pos = 0; bse = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0; bse = 0
            elif pos == -1:
                if sc >= -self.exit_threshold: pos = 0; bse = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0; bse = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


def period_backtest(df, strat, sym, start_dt, end_dt):
    """Run backtest on a specific date range."""
    mask = df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())
    period_df = df[mask].copy()
    if len(period_df) < 100:
        return None, None
    period_df = strat.generate_signals(period_df)
    r = run_backtest(period_df, strat, sym)
    dr = r.daily_returns
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return None, None
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
    ds = dr[dr < 0]
    dss = ds.std() if len(ds) > 1 else dr.std()
    sortino = (dr.mean() / dss) * np.sqrt(252) if dss > 0 else 0
    cum = (1 + dr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "sharpe": sharpe, "sortino": sortino, "return": r.total_return,
        "max_dd": max_dd, "trades": r.num_trades, "win_rate": r.win_rate,
        "pf": r.profit_factor,
    }, dr


def alpha_beta(strat_ret, bench_ret):
    # Normalize indexes to plain dates for alignment
    s = strat_ret.copy()
    b = bench_ret.copy()
    s.index = pd.to_datetime(s.index).normalize().tz_localize(None)
    b.index = pd.to_datetime(b.index).normalize().tz_localize(None)
    aligned = pd.DataFrame({"s": s, "b": b}).dropna()
    if len(aligned) < 10:
        return 0, 0
    beta = aligned["b"].cov(aligned["s"]) / aligned["b"].var() if aligned["b"].var() > 0 else 0
    alpha = (aligned["s"].mean() - beta * aligned["b"].mean()) * 252
    return alpha, beta


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading data...", flush=True)
symbols_needed = ["SPY", "QQQ", "GLD", "SMH"]
data = {}
for sym in symbols_needed:
    df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"  {sym}: {len(df)} bars", flush=True)

spy_bench_dev = data["SPY"][data["SPY"]["dt"] <= DEV_END].groupby("date")["close"].last().pct_change().dropna()
spy_bench_oos = data["SPY"][data["SPY"]["dt"] > DEV_END].groupby("date")["close"].last().pct_change().dropna()
spy_bench_full = data["SPY"].groupby("date")["close"].last().pct_change().dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# CANDIDATES TO TEST
# ═══════════════════════════════════════════════════════════════════════════════

# Top candidates from dev screen — best variant per strategy-instrument
CANDIDATES = [
    # GapCont on GLD: 8/8 variants positive, best atr25
    ("GapCont", "atr25", GapContinuation,
     {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "min_atr_pctl": 25}, "GLD"),
    ("GapCont", "t0.4_s0.20", GapContinuation,
     {"min_gap_pct": 0.3, "target_pct": 0.4, "stop_pct": 0.20}, "GLD"),
    ("GapCont", "default", GapContinuation,
     {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15}, "GLD"),

    # VWAPTrend on QQQ/SPY/SMH
    ("VWAPTrend", "atr25", VWAPTrend,
     {"confirm_bars": 10, "min_rel_vol": 1.0, "min_atr_pctl": 25}, "QQQ"),
    ("VWAPTrend", "conf5", VWAPTrend,
     {"confirm_bars": 5, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}, "SPY"),
    ("VWAPTrend", "atr25", VWAPTrend,
     {"confirm_bars": 10, "min_rel_vol": 1.0, "min_atr_pctl": 25}, "SMH"),
    ("VWAPTrend", "vol1.2", VWAPTrend,
     {"confirm_bars": 10, "min_rel_vol": 1.2, "min_vwap_dist_pct": 0.05}, "QQQ"),

    # MomScore on SPY (marginal)
    ("MomScore", "atr25", MomentumScore,
     {"entry_threshold": 3, "exit_threshold": 1, "min_atr_pctl": 25}, "SPY"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE: ORB on all three periods
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100, flush=True)
print("BASELINE: Production ORB (SPY+QQQ)", flush=True)
print("=" * 100, flush=True)

baseline_daily = {"dev": {}, "oos": {}, "full": {}}
for sym in ["SPY", "QQQ"]:
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(SYMBOL_PROFILES.get(sym, {}))
    strat = ORBBreakout(**params)

    for period, start, end in [("dev", DATA_START, DEV_END),
                                ("oos", OOS_START, DATA_END),
                                ("full", DATA_START, DATA_END)]:
        r, dr = period_backtest(data[sym], strat, sym, start, end)
        if r:
            bench = {"dev": spy_bench_dev, "oos": spy_bench_oos, "full": spy_bench_full}[period]
            a, b = alpha_beta(dr, bench)
            baseline_daily[period][sym] = dr
            print(f"  {sym} {period}: Sharpe={r['sharpe']:.2f}  Sort={r['sortino']:.2f}"
                  f"  Ret={r['return']:+.2%}  DD={r['max_dd']:.2%}  T={r['trades']}"
                  f"  a={a:+.1%}  b={b:.3f}", flush=True)

# Portfolio baselines
for period in ["dev", "oos", "full"]:
    if len(baseline_daily[period]) >= 2:
        port = pd.DataFrame(baseline_daily[period]).fillna(0).mean(axis=1)
        sh = (port.mean() / port.std()) * np.sqrt(252)
        ret = (1 + port).prod() - 1
        cum = (1 + port).cumprod()
        dd = ((cum - cum.cummax()) / cum.cummax()).min()
        print(f"  Portfolio {period}: Sharpe={sh:.2f}  Ret={ret:+.2%}  DD={dd:.2%}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOCKED OOS CONFIRMATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100, flush=True)
print("LOCKED OOS CONFIRMATION (Dec 2025 - Apr 2026)", flush=True)
print("Configs frozen from dev period. Results NOT used to iterate.", flush=True)
print("=" * 100, flush=True)

oos_survivors = []

print(f"\n  {'Strategy':<12} {'Variant':<15} {'Sym':<4} {'DevS':>6} {'OOSS':>6} {'OOSSort':>8}"
      f" {'OOSRet':>8} {'OOSDD':>7} {'T':>4} {'WR':>5} {'PF':>5}"
      f" {'Alpha':>7} {'Beta':>6} {'Verdict':<10}", flush=True)
print(f"  {'-'*11} {'-'*14} {'-'*3} {'-'*6} {'-'*6} {'-'*8}"
      f" {'-'*8} {'-'*7} {'-'*4} {'-'*5} {'-'*5}"
      f" {'-'*7} {'-'*6} {'-'*10}", flush=True)

for sname, vlabel, cls, kwargs, sym in CANDIDATES:
    strat = cls(**kwargs)

    # Dev period
    r_dev, dr_dev = period_backtest(data[sym], strat, sym, DATA_START, DEV_END)
    dev_sharpe = r_dev["sharpe"] if r_dev else 0

    # Locked OOS
    r_oos, dr_oos = period_backtest(data[sym], strat, sym, OOS_START, DATA_END)

    if r_oos and dr_oos is not None:
        a, b = alpha_beta(dr_oos, spy_bench_oos)

        # Verdict
        if r_oos["sharpe"] >= 0.5:
            verdict = "PASS"
        elif r_oos["sharpe"] >= 0:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

        print(f"  {sname:<12} {vlabel:<15} {sym:<4}"
              f" {dev_sharpe:>6.2f} {r_oos['sharpe']:>6.2f} {r_oos['sortino']:>8.2f}"
              f" {r_oos['return']:>+8.2%} {r_oos['max_dd']:>7.2%} {r_oos['trades']:>4}"
              f" {r_oos['win_rate']:>5.1%} {r_oos['pf']:>5.2f}"
              f" {a:>+7.1%} {b:>6.3f} {verdict:<10}", flush=True)

        if r_oos["sharpe"] >= 0:
            oos_survivors.append((sname, vlabel, cls, kwargs, sym, r_dev, r_oos, dr_dev, dr_oos))
    else:
        print(f"  {sname:<12} {vlabel:<15} {sym:<4} {dev_sharpe:>6.2f}  -> no OOS data", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO COMBINATION on locked OOS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100, flush=True)
print("PORTFOLIO COMBINATION (new + ORB baseline on locked OOS)", flush=True)
print("=" * 100, flush=True)

if oos_survivors and len(baseline_daily["oos"]) >= 2:
    port_base_oos = pd.DataFrame(baseline_daily["oos"]).fillna(0).mean(axis=1)
    base_oos_sharpe = (port_base_oos.mean() / port_base_oos.std()) * np.sqrt(252)
    base_oos_ret = (1 + port_base_oos).prod() - 1
    print(f"\n  Baseline SPY+QQQ ORB locked OOS: Sharpe={base_oos_sharpe:.2f}  Ret={base_oos_ret:+.2%}", flush=True)

    print(f"\n  {'Candidate':<40} {'PortS':>6} {'Delta':>6} {'Corr':>5}"
          f" {'SoloS':>6} {'SoloRet':>8} {'Alpha':>7} {'Beta':>6} {'Beats?':<8}", flush=True)
    print(f"  {'-'*39} {'-'*6} {'-'*6} {'-'*5}"
          f" {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*8}", flush=True)

    best_standalone = None
    best_portfolio = None

    for sname, vlabel, cls, kwargs, sym, r_dev, r_oos, dr_dev, dr_oos in oos_survivors:
        combined = dict(baseline_daily["oos"])
        key = f"{sname}/{vlabel}/{sym}"
        combined[key] = dr_oos
        port_df = pd.DataFrame(combined).fillna(0)
        port_ret = port_df.mean(axis=1)

        if len(port_ret) < 10 or port_ret.std() == 0:
            continue

        new_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252)
        delta = new_sharpe - base_oos_sharpe
        corr = dr_oos.corr(port_base_oos)
        a, b = alpha_beta(dr_oos, spy_bench_oos)

        beats = "YES" if delta > 0 else "no"

        print(f"  {key:<40} {new_sharpe:>6.2f} {delta:>+6.2f} {corr:>5.2f}"
              f" {r_oos['sharpe']:>6.2f} {r_oos['return']:>+8.2%}"
              f" {a:>+7.1%} {b:>6.3f} {beats:<8}", flush=True)

        if best_standalone is None or r_oos["sharpe"] > best_standalone[1]:
            best_standalone = (key, r_oos["sharpe"], r_oos)
        if best_portfolio is None or new_sharpe > best_portfolio[1]:
            best_portfolio = (key, new_sharpe, delta)

    # ═══════════════════════════════════════════════════════════════════════════
    # FULL PERIOD CHECK (Jan 2025 - Apr 2026)
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*100}", flush=True)
    print("FULL PERIOD CONSISTENCY CHECK (Jan 2025 - Apr 2026)", flush=True)
    print(f"{'='*100}", flush=True)

    for sname, vlabel, cls, kwargs, sym, r_dev, r_oos, dr_dev, dr_oos in oos_survivors:
        strat = cls(**kwargs)
        r_full, dr_full = period_backtest(data[sym], strat, sym, DATA_START, DATA_END)
        if r_full:
            a, b = alpha_beta(dr_full, spy_bench_full)
            print(f"  {sname}/{vlabel}/{sym}: Sharpe={r_full['sharpe']:.2f}"
                  f"  Sort={r_full['sortino']:.2f}  Ret={r_full['return']:+.2%}"
                  f"  DD={r_full['max_dd']:.2%}  T={r_full['trades']}"
                  f"  a={a:+.1%}  b={b:.3f}", flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*100}", flush=True)
    print("FINAL VERDICT", flush=True)
    print(f"{'='*100}", flush=True)

    print(f"\n  Baseline: SPY+QQQ ORB portfolio", flush=True)
    print(f"    Locked OOS Sharpe: {base_oos_sharpe:.2f}", flush=True)

    if best_standalone:
        print(f"\n  Best standalone new edge: {best_standalone[0]}", flush=True)
        print(f"    Locked OOS Sharpe: {best_standalone[1]:.2f}", flush=True)
        if best_standalone[1] > base_oos_sharpe:
            print(f"    BEATS baseline standalone? YES (+{best_standalone[1]-base_oos_sharpe:.2f})", flush=True)
        else:
            print(f"    BEATS baseline standalone? NO ({best_standalone[1]-base_oos_sharpe:+.2f})", flush=True)

    if best_portfolio:
        print(f"\n  Best portfolio addition: {best_portfolio[0]}", flush=True)
        print(f"    Combined locked OOS Sharpe: {best_portfolio[1]:.2f} (delta {best_portfolio[2]:+.2f})", flush=True)
        if best_portfolio[2] > 0:
            print(f"    IMPROVES portfolio? YES", flush=True)
        else:
            print(f"    IMPROVES portfolio? NO", flush=True)

    print(f"\n  BLUNT CONCLUSION:", flush=True)
    if best_standalone and best_standalone[1] > base_oos_sharpe:
        print(f"    A new edge ({best_standalone[0]}) beats the ORB baseline standalone on locked OOS.", flush=True)
    elif best_portfolio and best_portfolio[2] > 0:
        print(f"    No new edge beats baseline standalone, but {best_portfolio[0]}", flush=True)
        print(f"    improves the portfolio by {best_portfolio[2]:+.2f} Sharpe on locked OOS.", flush=True)
    else:
        print(f"    No new strategy tested honestly beats or improves the SPY+QQQ ORB baseline", flush=True)
        print(f"    on locked out-of-sample data. The current system remains the best validated solution.", flush=True)

else:
    print("  No candidates survived locked OOS or baseline missing.", flush=True)

print(f"\n{'='*100}", flush=True)
print("EXPERIMENT 14 COMPLETE", flush=True)
print(f"{'='*100}", flush=True)
