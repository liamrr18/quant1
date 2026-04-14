#!/usr/bin/env python3
"""Full portfolio stats: baseline + new candidates, all periods."""

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
from trading.config import (
    ORB_SHARED_DEFAULTS, SYMBOL_PROFILES,
    PAIRS_GLD_TLT, OPENDRIVE_SMH, OPENDRIVE_XLK,
)

ET = pytz.timezone("America/New_York")
START = datetime(2025, 1, 2, tzinfo=ET)
END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

print("Loading data...", flush=True)
data = {}
for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    df = get_minute_bars(sym, START, END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df

# Merge pairs data
gld_full = data["GLD"].copy()
tlt_close = data["TLT"].set_index("dt")["close"].rename("pair_close")
gld_full = gld_full.set_index("dt").join(tlt_close, how="left").reset_index()
gld_full["pair_close"] = gld_full["pair_close"].ffill()

spy_bench = data["SPY"].groupby("date")["close"].last().pct_change().dropna()


def run_period(df, strat, sym, start_dt, end_dt):
    mask = df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())
    p = df[mask].copy()
    if len(p) < 100: return None, None
    p = strat.generate_signals(p)
    r = run_backtest(p, strat, sym)
    return r, r.daily_returns


def metrics(dr, label=""):
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return {}
    sh = (dr.mean() / dr.std()) * np.sqrt(252)
    ret = (1 + dr).prod() - 1
    cum = (1 + dr).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    ds = dr[dr < 0]; dss = ds.std() if len(ds) > 1 else dr.std()
    so = (dr.mean() / dss) * np.sqrt(252) if dss > 0 else 0
    calmar = sh / abs(dd) * (sh / abs(sh)) if dd != 0 and sh != 0 else 0
    # Alpha/beta
    si = dr.copy(); bi = spy_bench.copy()
    si.index = pd.to_datetime(si.index).normalize().tz_localize(None)
    bi.index = pd.to_datetime(bi.index).normalize().tz_localize(None)
    al = pd.DataFrame({"s": si, "b": bi}).dropna()
    if len(al) >= 10:
        beta = al["b"].cov(al["s"]) / al["b"].var() if al["b"].var() > 0 else 0
        alpha = (al["s"].mean() - beta * al["b"].mean()) * 252
    else:
        alpha, beta = 0, 0
    return {"sharpe": sh, "sortino": so, "return": ret, "max_dd": dd,
            "alpha": alpha, "beta": beta, "calmar": calmar}


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD ALL STRATEGY STREAMS
# ═══════════════════════════════════════════════════════════════════════════════

streams = {}

# Baseline ORB
for sym in ["SPY", "QQQ"]:
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(SYMBOL_PROFILES.get(sym, {}))
    strat = ORBBreakout(**params)
    for period, s, e in [("dev", START, DEV_END), ("oos", OOS_START, END), ("full", START, END)]:
        r, dr = run_period(data[sym], strat, sym, s, e)
        if dr is not None:
            streams[(f"ORB_{sym}", period)] = dr

# Pairs GLD/TLT
cfg = PAIRS_GLD_TLT
strat = PairsSpread(lookback=cfg["lookback"], entry_zscore=cfg["entry_zscore"],
                    exit_zscore=cfg["exit_zscore"], stale_bars=cfg["stale_bars"],
                    last_entry_minute=cfg["last_entry_minute"])
for period, s, e in [("dev", START, DEV_END), ("oos", OOS_START, END), ("full", START, END)]:
    r, dr = run_period(gld_full, strat, "GLD", s, e)
    if dr is not None:
        streams[("Pairs_GLD_TLT", period)] = dr

# OpenDrive SMH
strat = OpeningDrive(**OPENDRIVE_SMH)
for period, s, e in [("dev", START, DEV_END), ("oos", OOS_START, END), ("full", START, END)]:
    r, dr = run_period(data["SMH"], strat, "SMH", s, e)
    if dr is not None:
        streams[("OD_SMH", period)] = dr

# OpenDrive XLK
strat = OpeningDrive(**OPENDRIVE_XLK)
for period, s, e in [("dev", START, DEV_END), ("oos", OOS_START, END), ("full", START, END)]:
    r, dr = run_period(data["XLK"], strat, "XLK", s, e)
    if dr is not None:
        streams[("OD_XLK", period)] = dr


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL STRATEGY STATS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 120, flush=True)
print("INDIVIDUAL STRATEGY PERFORMANCE", flush=True)
print("=" * 120, flush=True)

strat_names = ["ORB_SPY", "ORB_QQQ", "Pairs_GLD_TLT", "OD_SMH", "OD_XLK"]

for period_label, period_key in [("DEV (Jan-Nov 2025)", "dev"),
                                  ("LOCKED OOS (Dec 2025-Apr 2026)", "oos"),
                                  ("FULL (Jan 2025-Apr 2026)", "full")]:
    print(f"\n  {period_label}:", flush=True)
    print(f"  {'Strategy':<18} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>7} {'Alpha':>7} {'Beta':>6}", flush=True)
    print(f"  {'-'*17} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6}", flush=True)
    for sn in strat_names:
        dr = streams.get((sn, period_key))
        if dr is not None:
            m = metrics(dr)
            print(f"  {sn:<18} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['return']:>+8.2%}"
                  f" {m['max_dd']:>7.2%} {m['alpha']:>+7.1%} {m['beta']:>6.3f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 120, flush=True)
print("PORTFOLIO CONFIGURATIONS (equal-weight daily returns)", flush=True)
print("=" * 120, flush=True)

configs = {
    "A: ORB baseline (SPY+QQQ)": ["ORB_SPY", "ORB_QQQ"],
    "B: ORB + XLK OpenDrive": ["ORB_SPY", "ORB_QQQ", "OD_XLK"],
    "C: ORB + SMH OpenDrive": ["ORB_SPY", "ORB_QQQ", "OD_SMH"],
    "D: ORB + XLK + SMH": ["ORB_SPY", "ORB_QQQ", "OD_XLK", "OD_SMH"],
    "E: ORB + Pairs GLD/TLT": ["ORB_SPY", "ORB_QQQ", "Pairs_GLD_TLT"],
    "F: ORB + XLK + Pairs": ["ORB_SPY", "ORB_QQQ", "OD_XLK", "Pairs_GLD_TLT"],
    "G: FULL (ORB + XLK + SMH + Pairs)": ["ORB_SPY", "ORB_QQQ", "OD_XLK", "OD_SMH", "Pairs_GLD_TLT"],
}

for period_label, period_key in [("DEV (Jan-Nov 2025)", "dev"),
                                  ("LOCKED OOS (Dec 2025-Apr 2026)", "oos"),
                                  ("FULL (Jan 2025-Apr 2026)", "full")]:
    print(f"\n  {period_label}:", flush=True)
    print(f"  {'Config':<40} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>7}"
          f" {'Alpha':>7} {'Beta':>6} {'Streams':>7}", flush=True)
    print(f"  {'-'*39} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*7}", flush=True)

    for config_name, stream_list in configs.items():
        port_streams = {}
        for sn in stream_list:
            dr = streams.get((sn, period_key))
            if dr is not None:
                port_streams[sn] = dr

        if len(port_streams) < 2:
            continue

        port_df = pd.DataFrame(port_streams).fillna(0)
        port_ret = port_df.mean(axis=1)
        m = metrics(port_ret)
        if not m:
            continue

        print(f"  {config_name:<40} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['return']:>+8.2%}"
              f" {m['max_dd']:>7.2%} {m['alpha']:>+7.1%} {m['beta']:>6.3f} {len(port_streams):>7}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION MATRIX (locked OOS)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 120, flush=True)
print("CORRELATION MATRIX (locked OOS daily returns)", flush=True)
print("=" * 120, flush=True)

oos_streams = {}
for sn in strat_names:
    dr = streams.get((sn, "oos"))
    if dr is not None:
        oos_streams[sn] = dr

if len(oos_streams) >= 2:
    corr_df = pd.DataFrame(oos_streams).fillna(0).corr()
    print(f"\n  {'':>18}", end="", flush=True)
    for sn in corr_df.columns:
        print(f" {sn:>15}", end="", flush=True)
    print(flush=True)
    for sn in corr_df.index:
        print(f"  {sn:<18}", end="", flush=True)
        for sn2 in corr_df.columns:
            print(f" {corr_df.loc[sn, sn2]:>15.3f}", end="", flush=True)
        print(flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE COUNTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 120, flush=True)
print("TRADE COUNTS BY PERIOD", flush=True)
print("=" * 120, flush=True)

trade_counts = {}
# Re-run to get trade counts
all_strats = [
    ("ORB_SPY", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["SPY"]}), data["SPY"]),
    ("ORB_QQQ", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["QQQ"]}), data["QQQ"]),
    ("Pairs_GLD_TLT", PairsSpread(lookback=cfg["lookback"], entry_zscore=cfg["entry_zscore"],
                                   exit_zscore=cfg["exit_zscore"], stale_bars=cfg["stale_bars"],
                                   last_entry_minute=cfg["last_entry_minute"]), gld_full),
    ("OD_SMH", OpeningDrive(**OPENDRIVE_SMH), data["SMH"]),
    ("OD_XLK", OpeningDrive(**OPENDRIVE_XLK), data["XLK"]),
]

print(f"\n  {'Strategy':<18} {'Dev':>6} {'OOS':>6} {'Full':>6} {'Avg/Day(OOS)':>12}", flush=True)
print(f"  {'-'*17} {'-'*6} {'-'*6} {'-'*6} {'-'*12}", flush=True)

for sn, strat, df in all_strats:
    counts = {}
    for period, s, e in [("dev", START, DEV_END), ("oos", OOS_START, END), ("full", START, END)]:
        r, _ = run_period(df, strat, sn.split("_")[-1] if "ORB" in sn else sn, s, e)
        counts[period] = r.num_trades if r else 0
    oos_days = 84
    avg_per_day = counts["oos"] / oos_days if oos_days > 0 else 0
    print(f"  {sn:<18} {counts['dev']:>6} {counts['oos']:>6} {counts['full']:>6}"
          f" {avg_per_day:>12.1f}", flush=True)


print("\n" + "=" * 120, flush=True)
print("PORTFOLIO STATS COMPLETE", flush=True)
print("=" * 120, flush=True)
