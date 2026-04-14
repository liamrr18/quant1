#!/usr/bin/env python3
"""Pull extended data (Jan 2024 - Apr 2026) and re-run full portfolio backtest.

Tests whether the strategy edge holds over a longer period or was
specific to 2025. Reports per-year and per-strategy breakdown.
"""

import sys, os, io, warnings, time
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

# Extended period
EXT_START = datetime(2024, 1, 2, tzinfo=ET)
EXT_END = datetime(2026, 4, 14, tzinfo=ET)

# Original period
ORIG_START = datetime(2025, 1, 2, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

SYMBOLS = ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]

t0 = time.time()

# ── Pull and cache extended data ──
print("Pulling extended data (Jan 2024 - Apr 2026)...", flush=True)
print("This will take a few minutes — caching for future use.\n", flush=True)

data = {}
for sym in SYMBOLS:
    cache_label = f"{sym}_1min_20240102_20260414"
    cache_path = os.path.join("data", "cache", f"{cache_label}.csv")

    if os.path.exists(cache_path):
        print(f"  {sym}: loading from cache...", end="", flush=True)
        df = pd.read_csv(cache_path, parse_dates=["dt"])
        df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)
        print(f" {len(df)} bars", flush=True)
    else:
        print(f"  {sym}: pulling from Alpaca...", end="", flush=True)
        df = get_minute_bars(sym, EXT_START, EXT_END, use_cache=False)
        # Save to cache
        os.makedirs(os.path.join("data", "cache"), exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f" {len(df)} bars, cached to {cache_path}", flush=True)

    df = prepare_features(df)
    data[sym] = df

# Merge pairs
gld_full = data["GLD"].copy()
tc = data["TLT"].set_index("dt")["close"].rename("pair_close")
gld_full = gld_full.set_index("dt").join(tc, how="left").reset_index()
gld_full["pair_close"] = gld_full["pair_close"].ffill()

print(f"\n{time.time()-t0:.0f}s | Data loaded.", flush=True)


def run_period(df, strat, sym, start_dt, end_dt):
    mask = df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())
    p = df[mask].copy()
    if len(p) < 100:
        return None, None
    p = strat.generate_signals(p)
    r = run_backtest(p, strat, sym)
    return r, r.daily_returns


def metrics(dr):
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return {"sharpe": 0, "sortino": 0, "return": 0, "max_dd": 0, "days": 0}
    sh = (dr.mean() / dr.std()) * np.sqrt(252)
    ret = (1 + dr).prod() - 1
    cum = (1 + dr).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    ds = dr[dr < 0]
    dss = ds.std() if len(ds) > 1 else dr.std()
    so = (dr.mean() / dss) * np.sqrt(252) if dss > 0 else 0
    return {"sharpe": sh, "sortino": so, "return": ret, "max_dd": dd, "days": len(dr)}


def alpha_beta(strat_ret, bench_ret):
    si = strat_ret.copy(); bi = bench_ret.copy()
    si.index = pd.to_datetime(si.index).normalize().tz_localize(None)
    bi.index = pd.to_datetime(bi.index).normalize().tz_localize(None)
    al = pd.DataFrame({"s": si, "b": bi}).dropna()
    if len(al) < 10: return 0, 0
    beta = al["b"].cov(al["s"]) / al["b"].var() if al["b"].var() > 0 else 0
    alpha = (al["s"].mean() - beta * al["b"].mean()) * 252
    return alpha, beta


# ── Strategy configs ──
STRATS = [
    ("ORB_SPY", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["SPY"]}), data["SPY"], "SPY"),
    ("ORB_QQQ", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["QQQ"]}), data["QQQ"], "QQQ"),
    ("Pairs_GLD_TLT", PairsSpread(lookback=120, entry_zscore=2.0, exit_zscore=0.5,
                                   stale_bars=90, last_entry_minute=900), gld_full, "GLD"),
    ("OD_SMH", OpeningDrive(**OPENDRIVE_SMH), data["SMH"], "SMH"),
    ("OD_XLK", OpeningDrive(**OPENDRIVE_XLK), data["XLK"], "XLK"),
]

rp_weights = np.array([PORTFOLIO_WEIGHTS[n] for n in
                        ["ORB_SPY", "ORB_QQQ", "Pairs_GLD_TLT", "OD_SMH", "OD_XLK"]])

spy_bench = data["SPY"].groupby("date")["close"].last().pct_change().dropna()

# ── Periods to test ──
PERIODS = [
    ("2024 only (new data)", datetime(2024, 1, 2, tzinfo=ET), datetime(2024, 12, 31, tzinfo=ET)),
    ("2025 dev (original)", ORIG_START, DEV_END),
    ("Locked OOS", OOS_START, EXT_END),
    ("Full extended (Jan 2024 - Apr 2026)", EXT_START, EXT_END),
    ("Full original (Jan 2025 - Apr 2026)", ORIG_START, EXT_END),
]

# ── Per-strategy results ──
print(f"\n{'='*110}", flush=True)
print("EXTENDED BACKTEST: PER-STRATEGY RESULTS", flush=True)
print(f"{'='*110}", flush=True)

for period_name, pstart, pend in PERIODS:
    print(f"\n  {period_name}:", flush=True)
    print(f"  {'Strategy':<16} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>7}"
          f" {'Alpha':>7} {'Beta':>6} {'Trades':>6} {'Days':>5}", flush=True)
    print(f"  {'-'*15} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5}", flush=True)

    period_drs = {}
    for name, strat, df, sym in STRATS:
        r, dr = run_period(df, strat, sym, pstart, pend)
        if r and dr is not None and len(dr) > 5:
            m = metrics(dr)
            a, b = alpha_beta(dr, spy_bench)
            period_drs[name] = dr
            print(f"  {name:<16} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['return']:>+8.2%}"
                  f" {m['max_dd']:>7.2%} {a:>+7.1%} {b:>6.3f} {r.num_trades:>6} {m['days']:>5}", flush=True)
        else:
            print(f"  {name:<16} — no data for this period", flush=True)

    # Portfolio
    if len(period_drs) >= 2:
        port_df = pd.DataFrame(period_drs).fillna(0)
        # Use risk-parity weights where available, equal for missing
        w = []
        for n in ["ORB_SPY", "ORB_QQQ", "Pairs_GLD_TLT", "OD_SMH", "OD_XLK"]:
            if n in period_drs:
                w.append(PORTFOLIO_WEIGHTS[n])
            # skip missing
        if len(w) == len(period_drs):
            w = np.array(w) / np.sum(w)  # Renormalize
            port = pd.Series(port_df.values @ w, index=port_df.index)
        else:
            port = port_df.mean(axis=1)

        m = metrics(port)
        a, b = alpha_beta(port, spy_bench)
        print(f"  {'PORTFOLIO':<16} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['return']:>+8.2%}"
              f" {m['max_dd']:>7.2%} {a:>+7.1%} {b:>6.3f} {'':>6} {m['days']:>5}", flush=True)


# ── Key comparison: 2024 vs 2025 ──
print(f"\n{'='*110}", flush=True)
print("KEY COMPARISON: Does the edge hold on 2024 data?", flush=True)
print(f"{'='*110}", flush=True)

print(f"\n  {'Strategy':<16} {'2024 Sharpe':>11} {'2025 dev':>9} {'OOS':>7} {'Holds?':>7}", flush=True)
print(f"  {'-'*15} {'-'*11} {'-'*9} {'-'*7} {'-'*7}", flush=True)

for name, strat, df, sym in STRATS:
    r24, dr24 = run_period(df, strat, sym,
                            datetime(2024, 1, 2, tzinfo=ET), datetime(2024, 12, 31, tzinfo=ET))
    r25, dr25 = run_period(df, strat, sym, ORIG_START, DEV_END)
    roos, droos = run_period(df, strat, sym, OOS_START, EXT_END)

    m24 = metrics(dr24)
    m25 = metrics(dr25)
    moos = metrics(droos)

    # "Holds" = 2024 Sharpe is positive (strategy had edge before our dev period)
    holds = "YES" if m24["sharpe"] > 0 else "NO"
    if m24["sharpe"] <= 0:
        holds = "NO"
    elif m24["sharpe"] < m25["sharpe"] * 0.3:
        holds = "WEAK"
    else:
        holds = "YES"

    print(f"  {name:<16} {m24['sharpe']:>11.2f} {m25['sharpe']:>9.2f} {moos['sharpe']:>7.2f} {holds:>7}", flush=True)


# ── Verdict ──
print(f"\n{'='*110}", flush=True)
print("VERDICT", flush=True)
print(f"{'='*110}", flush=True)

# Compute portfolio metrics for key periods
port_results = {}
for period_name, pstart, pend in PERIODS:
    period_drs = {}
    for name, strat, df, sym in STRATS:
        _, dr = run_period(df, strat, sym, pstart, pend)
        if dr is not None and len(dr) > 5:
            period_drs[name] = dr
    if len(period_drs) >= 2:
        port_df = pd.DataFrame(period_drs).fillna(0)
        w = np.array([PORTFOLIO_WEIGHTS[n] for n in period_drs.keys()])
        w = w / w.sum()
        port = pd.Series(port_df.values @ w, index=port_df.index)
        port_results[period_name] = metrics(port)

for pname, m in port_results.items():
    print(f"  {pname:<45} Sharpe={m['sharpe']:>6.2f}  MaxDD={m['max_dd']:>6.2%}  Ret={m['return']:>+7.2%}", flush=True)

print(f"\n{time.time()-t0:.0f}s | EXTENDED BACKTEST COMPLETE", flush=True)
