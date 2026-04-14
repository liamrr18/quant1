#!/usr/bin/env python3
"""Pull fresh data through today and re-run locked OOS with extended period."""

import sys, os, io, warnings
from datetime import datetime
import pytz, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.backtest.engine import run_backtest
from trading.strategies.orb import ORBBreakout
from trading.strategies.pairs_spread import PairsSpread
from trading.strategies.opening_drive import OpeningDrive
from trading.config import *

ET = pytz.timezone("America/New_York")
START = datetime(2025, 1, 2, tzinfo=ET)
END = datetime(2026, 4, 14, tzinfo=ET)  # Through today
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

print("Pulling fresh data through 2026-04-14...", flush=True)
data = {}
for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    # Force fresh pull (don't use old cache)
    df = get_minute_bars(sym, START, END, use_cache=False)
    df = prepare_features(df)
    data[sym] = df
    n_days = df["date"].nunique()
    print(f"  {sym}: {len(df)} bars, {n_days} days", flush=True)

# Merge pairs
gld = data["GLD"].copy()
tc = data["TLT"].set_index("dt")["close"].rename("pair_close")
gld = gld.set_index("dt").join(tc, how="left").reset_index()
gld["pair_close"] = gld["pair_close"].ffill()

spy_bench = data["SPY"].groupby("date")["close"].last().pct_change().dropna()

def run_period(df, strat, sym, s, e):
    mask = df["date"].apply(lambda d: s.date() <= d <= e.date())
    p = df[mask].copy()
    if len(p) < 100: return None, None
    p = strat.generate_signals(p)
    r = run_backtest(p, strat, sym)
    return r, r.daily_returns

def metrics(dr):
    if dr is None or len(dr) < 5 or dr.std() == 0: return {}
    sh = (dr.mean()/dr.std())*np.sqrt(252)
    ret = (1+dr).prod()-1
    cum = (1+dr).cumprod()
    dd = ((cum-cum.cummax())/cum.cummax()).min()
    ds = dr[dr<0]; dss = ds.std() if len(ds)>1 else dr.std()
    so = (dr.mean()/dss)*np.sqrt(252) if dss>0 else 0
    return {"sharpe": sh, "sortino": so, "return": ret, "max_dd": dd}

print(f"\n{'='*80}", flush=True)
print(f"EXTENDED LOCKED OOS (Dec 2025 - Apr 14, 2026)", flush=True)
print(f"10 more trading days than original (Apr 4)", flush=True)
print(f"{'='*80}", flush=True)

strats = [
    ("ORB SPY", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["SPY"]}), data["SPY"], "SPY"),
    ("ORB QQQ", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["QQQ"]}), data["QQQ"], "QQQ"),
    ("Pairs GLD/TLT", PairsSpread(lookback=120, entry_zscore=2.0, exit_zscore=0.5, stale_bars=90, last_entry_minute=900), gld, "GLD"),
    ("OD SMH", OpeningDrive(**OPENDRIVE_SMH), data["SMH"], "SMH"),
    ("OD XLK", OpeningDrive(**OPENDRIVE_XLK), data["XLK"], "XLK"),
]

oos_daily = {}
print(f"\n  {'Strategy':<16} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>7} {'Trades':>6} {'Days':>5}", flush=True)
print(f"  {'-'*15} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*5}", flush=True)

for name, strat, df, sym in strats:
    r, dr = run_period(df, strat, sym, OOS_START, END)
    if r and dr is not None:
        m = metrics(dr)
        n_days = len(dr)
        oos_daily[name] = dr
        print(f"  {name:<16} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['return']:>+8.2%} {m['max_dd']:>7.2%} {r.num_trades:>6} {n_days:>5}", flush=True)

# Portfolio
print(f"\n  Portfolio configs:", flush=True)
configs = {
    "A: ORB only": ["ORB SPY", "ORB QQQ"],
    "B: ORB + XLK": ["ORB SPY", "ORB QQQ", "OD XLK"],
    "G: FULL": ["ORB SPY", "ORB QQQ", "OD XLK", "OD SMH", "Pairs GLD/TLT"],
}
for cname, streams in configs.items():
    port = {s: oos_daily[s] for s in streams if s in oos_daily}
    if len(port) < 2: continue
    pr = pd.DataFrame(port).fillna(0).mean(axis=1)
    m = metrics(pr)
    print(f"  {cname:<16} Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}  Ret={m['return']:+.2%}  DD={m['max_dd']:.2%}", flush=True)

print(f"\n  Compare to original OOS (through Apr 4):", flush=True)
print(f"    A: ORB only   Sharpe=3.35", flush=True)
print(f"    G: FULL       Sharpe=6.17", flush=True)
print(f"\nFresh data pull complete.", flush=True)
