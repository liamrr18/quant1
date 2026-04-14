#!/usr/bin/env python3
"""Experiment 15 Phase 2: Locked OOS confirmation for Wave 2 candidates.

Top dev-period candidates tested on held-out Dec 2025 - Apr 2026 data.
Configs frozen from dev period.
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

from trading.strategies.base import Strategy
from trading.backtest.engine import run_backtest
from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.config import ORB_SHARED_DEFAULTS, SYMBOL_PROFILES
from trading.strategies.orb import ORBBreakout

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)


# ── Strategy classes (inlined) ──

class OpeningDrive(Strategy):
    name = "opening_drive"
    def __init__(self, drive_minutes=5, min_drive_pct=0.10, target_multiple=2.0,
                 stop_multiple=1.0, require_gap_align=False, min_gap_pct=0.0,
                 stale_bars=120, last_entry_minute=720, min_atr_pctl=0):
        self.drive_minutes = drive_minutes; self.min_drive_pct = min_drive_pct
        self.target_multiple = target_multiple; self.stop_multiple = stop_multiple
        self.require_gap_align = require_gap_align; self.min_gap_pct = min_gap_pct
        self.stale_bars = stale_bars; self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"drive": self.drive_minutes, "min_d": self.min_drive_pct,
                "tgt": self.target_multiple, "stp": self.stop_multiple}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        de = 9*60+30+self.drive_minutes
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); pos = 0; ep = 0.0; ebar = 0
        tgt = 0.0; stp = 0.0; cdate = None; skip = False
        do = 0.0; dd = 0; ds = 0.0; entered = False
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; dd = 0; ds = 0.0; entered = False
                do = row["open"]
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
                if self.require_gap_align and "gap_pct" in df.columns:
                    gp = row.get("gap_pct", 0)
                    if pd.isna(gp) or abs(gp) < self.min_gap_pct: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            if row["minute_of_day"] <= de:
                if do > 0:
                    mp = (row["close"] - do) / do * 100
                    if abs(mp) > abs(ds): ds = mp
                signals[i] = 0; continue
            if dd == 0 and not entered:
                if ds >= self.min_drive_pct: dd = 1
                elif ds <= -self.min_drive_pct: dd = -1
                else: skip = True; signals[i] = 0; continue
                if self.require_gap_align and "gap_pct" in df.columns:
                    gp = row.get("gap_pct", 0)
                    if pd.notna(gp):
                        if (dd > 0 and gp < 0) or (dd < 0 and gp > 0): skip = True; signals[i] = 0; continue
            if pos == 0 and not entered and dd != 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                entered = True; pos = dd; ep = row["close"]; ebar = i
                ad = abs(ds / 100 * ep)
                if pos == 1: tgt = ep + ad*self.target_multiple; stp = ep - ad*self.stop_multiple
                else: tgt = ep - ad*self.target_multiple; stp = ep + ad*self.stop_multiple
            elif pos == 1:
                if row["close"] >= tgt or row["close"] <= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if row["close"] <= tgt or row["close"] >= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


class PairsSpread(Strategy):
    name = "pairs_spread"
    def __init__(self, lookback=60, entry_zscore=2.0, exit_zscore=0.5,
                 stale_bars=90, last_entry_minute=900):
        self.lookback = lookback; self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore; self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
    def get_params(self):
        return {"look": self.lookback, "z_in": self.entry_zscore, "z_out": self.exit_zscore}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "pair_close" not in df.columns: return df
        spread = np.log(df["close"]) - np.log(df["pair_close"])
        sm = spread.rolling(self.lookback, min_periods=20).mean()
        ss = spread.rolling(self.lookback, min_periods=20).std()
        zs = (spread - sm) / ss.replace(0, np.nan)
        signals = np.zeros(len(df)); pos = 0; ebar = 0; cdate = None
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate: pos = 0; cdate = d
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            z = zs.iloc[i] if i < len(zs) else 0
            if pd.isna(z): signals[i] = pos; continue
            if pos == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                if z < -self.entry_zscore: pos = 1; ebar = i
                elif z > self.entry_zscore: pos = -1; ebar = i
            elif pos == 1:
                if z >= -self.exit_zscore: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if z <= self.exit_zscore: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


class RangeExpansion(Strategy):
    name = "range_expansion"
    def __init__(self, nr_lookback=7, min_narrow_bars=3, target_multiple=2.0,
                 stop_multiple=0.5, stale_bars=60, last_entry_minute=900, min_atr_pctl=0):
        self.nr_lookback = nr_lookback; self.min_narrow_bars = min_narrow_bars
        self.target_multiple = target_multiple; self.stop_multiple = stop_multiple
        self.stale_bars = stale_bars; self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"nr": self.nr_lookback, "narrow": self.min_narrow_bars, "tgt": self.target_multiple}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        br = df["high"] - df["low"]
        mr = br.rolling(self.nr_lookback, min_periods=3).min()
        is_nr = br <= mr * 1.05
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); pos = 0; ep = 0.0; ebar = 0
        tgt = 0.0; stp = 0.0; cdate = None; skip = False
        nc = 0; ch = 0.0; cl = 999999.0
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; nc = 0; ch = 0.0; cl = 999999.0
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            nr = is_nr.iloc[i] if i < len(is_nr) else False
            if pd.isna(nr): nr = False
            if pos == 0:
                if nr:
                    nc += 1; ch = max(ch, row["high"]); cl = min(cl, row["low"])
                else:
                    if nc >= self.min_narrow_bars:
                        if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                            nc = 0; signals[i] = 0; continue
                        cr = ch - cl
                        if cr > 0:
                            if row["close"] > ch:
                                pos = 1; ep = row["close"]; ebar = i
                                tgt = ep + cr*self.target_multiple; stp = ep - cr*self.stop_multiple
                            elif row["close"] < cl:
                                pos = -1; ep = row["close"]; ebar = i
                                tgt = ep - cr*self.target_multiple; stp = ep + cr*self.stop_multiple
                    nc = 0; ch = 0.0; cl = 999999.0
            elif pos == 1:
                if row["close"] >= tgt or row["close"] <= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if row["close"] <= tgt or row["close"] >= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


# ── Helpers ──

def period_bt(df, strat, sym, start, end):
    mask = df["date"].apply(lambda d: start.date() <= d <= end.date())
    p = df[mask].copy()
    if len(p) < 100: return None, None
    p = strat.generate_signals(p)
    r = run_backtest(p, strat, sym)
    dr = r.daily_returns
    if dr is None or len(dr) < 5 or dr.std() == 0: return None, None
    sh = (dr.mean()/dr.std())*np.sqrt(252)
    ds = dr[dr<0]; dss = ds.std() if len(ds)>1 else dr.std()
    so = (dr.mean()/dss)*np.sqrt(252) if dss>0 else 0
    cum = (1+dr).cumprod(); dd = ((cum-cum.cummax())/cum.cummax()).min()
    return {"sharpe": sh, "sortino": so, "return": r.total_return,
            "max_dd": dd, "trades": r.num_trades, "win_rate": r.win_rate,
            "pf": r.profit_factor}, dr


def ab(s, b):
    si = s.copy(); bi = b.copy()
    si.index = pd.to_datetime(si.index).normalize().tz_localize(None)
    bi.index = pd.to_datetime(bi.index).normalize().tz_localize(None)
    al = pd.DataFrame({"s": si, "b": bi}).dropna()
    if len(al) < 10: return 0, 0
    beta = al["b"].cov(al["s"]) / al["b"].var() if al["b"].var() > 0 else 0
    alpha = (al["s"].mean() - beta * al["b"].mean()) * 252
    return alpha, beta


# ── Load data ──

print("Loading data...", flush=True)
syms = ["SPY", "QQQ", "GLD", "XLE", "XLK", "SMH", "TLT"]
data = {}
for s in syms:
    df = get_minute_bars(s, DATA_START, DATA_END, use_cache=True)
    df = prepare_features(df)
    data[s] = df
    print(f"  {s}: {len(df)} bars", flush=True)

spy_bench_oos = data["SPY"][data["SPY"]["dt"] > DEV_END].groupby("date")["close"].last().pct_change().dropna()

# ── Candidates ──

CANDIDATES = [
    # Pairs: XLK vs SMH (6/6 variants positive)
    ("Pairs", "XLKvSMH_z2.5", PairsSpread, {"lookback": 60, "entry_zscore": 2.5, "exit_zscore": 0.5},
     "XLK", "SMH"),
    ("Pairs", "XLKvSMH_z1.5", PairsSpread, {"lookback": 60, "entry_zscore": 1.5, "exit_zscore": 0.3},
     "XLK", "SMH"),
    ("Pairs", "XLKvSMH_z2.0_look120", PairsSpread, {"lookback": 120, "entry_zscore": 2.0, "exit_zscore": 0.5},
     "XLK", "SMH"),

    # Pairs: SPY vs QQQ
    ("Pairs", "SPYvQQQ_z2.5", PairsSpread, {"lookback": 60, "entry_zscore": 2.5, "exit_zscore": 0.5},
     "SPY", "QQQ"),

    # Pairs: GLD vs TLT
    ("Pairs", "GLDvTLT_look120", PairsSpread, {"lookback": 120, "entry_zscore": 2.0, "exit_zscore": 0.5},
     "GLD", "TLT"),

    # OpeningDrive: XLK, SMH
    ("OpenDrive", "5m_tgt1.5x", OpeningDrive,
     {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 1.5, "stop_multiple": 1.0},
     "XLK", None),
    ("OpenDrive", "5m_tgt3x", OpeningDrive,
     {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 3.0, "stop_multiple": 1.0},
     "SMH", None),
    ("OpenDrive", "5m_gap_align", OpeningDrive,
     {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 2.0, "stop_multiple": 1.0,
      "require_gap_align": True, "min_gap_pct": 0.2},
     "SMH", None),

    # RangeExpansion: SPY
    ("RangeExp", "nr10_3bars", RangeExpansion,
     {"nr_lookback": 10, "min_narrow_bars": 3, "target_multiple": 2.0},
     "SPY", None),
]


# ── Baseline ──

print("\n" + "=" * 100, flush=True)
print("BASELINE: ORB SPY+QQQ on locked OOS", flush=True)
print("=" * 100, flush=True)

bl_oos = {}
for s in ["SPY", "QQQ"]:
    params = dict(ORB_SHARED_DEFAULTS); params.update(SYMBOL_PROFILES.get(s, {}))
    strat = ORBBreakout(**params)
    r, dr = period_bt(data[s], strat, s, OOS_START, DATA_END)
    if r:
        a, b = ab(dr, spy_bench_oos)
        print(f"  {s}: Sharpe={r['sharpe']:.2f}  Ret={r['return']:+.2%}  T={r['trades']}  a={a:+.1%}", flush=True)
        bl_oos[s] = dr

port_bl = pd.DataFrame(bl_oos).fillna(0).mean(axis=1)
bl_sh = (port_bl.mean()/port_bl.std())*np.sqrt(252)
print(f"  Portfolio: Sharpe={bl_sh:.2f}", flush=True)


# ── Locked OOS ──

print("\n" + "=" * 100, flush=True)
print("LOCKED OOS CONFIRMATION (Dec 2025 - Apr 2026)", flush=True)
print("=" * 100, flush=True)

print(f"\n  {'Name':<30} {'Sym':<4} {'DevS':>5} {'OOSS':>6} {'Sort':>6}"
      f" {'Ret':>8} {'DD':>7} {'T':>4} {'PF':>5} {'Alpha':>7} {'Beta':>6}", flush=True)
print(f"  {'-'*29} {'-'*3} {'-'*5} {'-'*6} {'-'*6}"
      f" {'-'*8} {'-'*7} {'-'*4} {'-'*5} {'-'*7} {'-'*6}", flush=True)

oos_survivors = []

for sname, vlabel, cls, kwargs, sym, pair_sym in CANDIDATES:
    strat = cls(**kwargs)

    # Merge pair data if needed
    df_full = data[sym].copy()
    if pair_sym:
        pc = data[pair_sym].set_index("dt")["close"].rename("pair_close")
        df_full = df_full.set_index("dt").join(pc, how="left").reset_index()
        df_full["pair_close"] = df_full["pair_close"].ffill()

    # Dev
    r_dev, _ = period_bt(df_full, strat, sym, DATA_START, DEV_END)
    dev_sh = r_dev["sharpe"] if r_dev else 0

    # OOS
    r_oos, dr_oos = period_bt(df_full, strat, sym, OOS_START, DATA_END)
    if r_oos and dr_oos is not None:
        a, b = ab(dr_oos, spy_bench_oos)
        tag = "PASS" if r_oos["sharpe"] >= 0.5 else ("marg" if r_oos["sharpe"] >= 0 else "FAIL")
        key = f"{sname}/{vlabel}"
        print(f"  {key:<30} {sym:<4} {dev_sh:>5.2f} {r_oos['sharpe']:>6.2f} {r_oos['sortino']:>6.2f}"
              f" {r_oos['return']:>+8.2%} {r_oos['max_dd']:>7.2%} {r_oos['trades']:>4}"
              f" {r_oos['pf']:>5.2f} {a:>+7.1%} {b:>6.3f}  {tag}", flush=True)

        if r_oos["sharpe"] >= 0:
            oos_survivors.append((sname, vlabel, sym, pair_sym, r_oos, dr_oos))
    else:
        print(f"  {sname}/{vlabel:<24} {sym:<4} {dev_sh:>5.2f}  -> no OOS data", flush=True)


# ── Portfolio combination ──

print("\n" + "=" * 100, flush=True)
print("PORTFOLIO COMBINATION (new + ORB baseline on locked OOS)", flush=True)
print("=" * 100, flush=True)

print(f"\n  Baseline SPY+QQQ ORB: Sharpe={bl_sh:.2f}", flush=True)

if oos_survivors:
    best_combo = None
    for sname, vlabel, sym, pair_sym, r_oos, dr_oos in oos_survivors:
        combined = dict(bl_oos)
        key = f"{sname}/{vlabel}/{sym}"
        combined[key] = dr_oos
        pdf = pd.DataFrame(combined).fillna(0)
        pr = pdf.mean(axis=1)
        if len(pr) < 10 or pr.std() == 0: continue
        nsh = (pr.mean()/pr.std())*np.sqrt(252)
        delta = nsh - bl_sh
        corr = dr_oos.corr(port_bl)
        a, b = ab(dr_oos, spy_bench_oos)
        beats = "YES" if delta > 0 else "no"
        print(f"  + {key:<40} port={nsh:.2f} (d={delta:+.2f})"
              f"  solo={r_oos['sharpe']:.2f}  corr={corr:.2f}"
              f"  a={a:+.1%}  {beats}", flush=True)
        if best_combo is None or nsh > best_combo[1]:
            best_combo = (key, nsh, delta, r_oos["sharpe"], corr)

    # Multi-strategy portfolio: combine ALL positive OOS survivors
    if len(oos_survivors) >= 2:
        multi = dict(bl_oos)
        for sname, vlabel, sym, pair_sym, r_oos, dr_oos in oos_survivors:
            if r_oos["sharpe"] > 0:
                multi[f"{sname}/{vlabel}/{sym}"] = dr_oos
        if len(multi) > 2:
            mpdf = pd.DataFrame(multi).fillna(0)
            mpr = mpdf.mean(axis=1)
            if len(mpr) > 10 and mpr.std() > 0:
                msh = (mpr.mean()/mpr.std())*np.sqrt(252)
                mdelta = msh - bl_sh
                print(f"\n  Multi-strategy portfolio ({len(multi)} streams): Sharpe={msh:.2f} (d={mdelta:+.2f})", flush=True)

    if best_combo:
        print(f"\n  BEST SINGLE ADDITION: {best_combo[0]}", flush=True)
        print(f"    Portfolio Sharpe: {best_combo[1]:.2f} (delta {best_combo[2]:+.2f})", flush=True)
else:
    print("  No candidates survived locked OOS.", flush=True)


# ── Verdict ──

print(f"\n{'='*100}", flush=True)
print("FINAL VERDICT", flush=True)
print(f"{'='*100}", flush=True)

print(f"\n  Baseline ORB locked OOS: Sharpe={bl_sh:.2f}", flush=True)

if oos_survivors:
    best_solo = max(oos_survivors, key=lambda x: x[4]["sharpe"])
    print(f"  Best new standalone: {best_solo[0]}/{best_solo[1]}/{best_solo[2]}"
          f" Sharpe={best_solo[4]['sharpe']:.2f}", flush=True)
    if best_solo[4]["sharpe"] > bl_sh:
        print(f"  BEATS BASELINE STANDALONE: YES", flush=True)
    else:
        print(f"  BEATS BASELINE STANDALONE: NO", flush=True)
    if best_combo and best_combo[2] > 0:
        print(f"  IMPROVES PORTFOLIO: YES ({best_combo[0]}, delta {best_combo[2]:+.2f})", flush=True)
    else:
        print(f"  IMPROVES PORTFOLIO: NO", flush=True)
else:
    print(f"  No new strategies survived locked OOS.", flush=True)

print(f"\n{'='*100}", flush=True)
print("EXPERIMENT 15 LOCKED OOS COMPLETE", flush=True)
print(f"{'='*100}", flush=True)
