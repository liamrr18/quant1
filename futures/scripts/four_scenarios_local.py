#!/usr/bin/env python3
"""4-scenario projection — runs LOCALLY using cached data."""

import json
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

FUTURES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUITY_DIR = r"C:\Users\liamr\Desktop\spy-trader\.claude\worktrees\flamboyant-lewin"
ET = pytz.timezone("America/New_York")

sys.path.insert(0, FUTURES_DIR)

from trading.strategies.orb import ORBBreakout as FORB
from trading.strategies.vwap_reversion import VWAPReversion
from trading.strategies.overnight_reversion import OvernightReversion
from trading.config import (
    ORB_SHARED_DEFAULTS as F_ORB_DEFAULTS, SYMBOL_PROFILES as F_PROFILES,
    VWAP_REVERSION_DEFAULTS, OVERNIGHT_REVERSION_DEFAULTS,
)

DAYS_3MO = 63
MONTH_DAYS = 21
N_SIMS = 10000

MARGIN = {"MES": 1500, "MNQ": 1800, "MGC": 1000}
MULT = {"MES": 5, "MNQ": 2, "MGC": 10}
COMM_RT = 0.62

cache = os.path.join(FUTURES_DIR, "data", "cache")


def load_cash(f):
    df = pd.read_csv(os.path.join(cache, f))
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)
    df["date"] = df["dt"].dt.date
    return df


def run_strat(df, strat):
    sig = strat.generate_signals(df.copy())
    sig["pos"] = sig["signal"].shift(1).fillna(0)
    sig["br"] = sig["close"].pct_change().fillna(0)
    sig["sr"] = sig["pos"] * sig["br"]
    return sig.groupby("date")["sr"].sum()


def oos(df):
    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.8)
    return df[df["date"].isin(set(dates[split:]))].copy()


print("Loading futures OOS data...", flush=True)
futures_streams = {}
for sym, fn in [("MES", "MES_futures_cash_1min.csv"),
                ("MNQ", "MNQ_futures_cash_1min.csv"),
                ("MGC", "MGC_futures_cash_1min.csv")]:
    df = load_cash(fn)
    oos_df = oos(df)
    vwap_strat = VWAPReversion(**VWAP_REVERSION_DEFAULTS)
    futures_streams[f"VWAP_{sym}"] = run_strat(oos_df, vwap_strat)
    if sym != "MGC":
        orb_params = dict(F_ORB_DEFAULTS); orb_params.update(F_PROFILES.get(sym, {}))
        orb_strat = FORB(**orb_params)
        futures_streams[f"ORB_{sym}"] = run_strat(oos_df, orb_strat)

on_path = os.path.join(cache, "MNQ_futures_full_1min.csv")
on_df = pd.read_csv(on_path)
on_df["dt"] = pd.to_datetime(on_df["dt"], utc=True).dt.tz_convert(ET)
on_df["date"] = on_df["dt"].dt.date
on_oos = oos(on_df)
on_strat = OvernightReversion(**OVERNIGHT_REVERSION_DEFAULTS)
futures_streams["ON_MNQ"] = run_strat(on_oos, on_strat)


# Equity data via subprocess
print("Loading equity OOS data...", flush=True)

eq_script = '''
import sys, json, pandas as pd, pytz, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"''' + EQUITY_DIR.replace("\\", "\\\\") + '''")
from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.backtest.engine import run_backtest
from trading.strategies.orb import ORBBreakout
from trading.strategies.opening_drive import OpeningDrive
from trading.strategies.pairs_spread import PairsSpread
from trading.config import (ORB_SHARED_DEFAULTS, SYMBOL_PROFILES, PAIRS_GLD_TLT,
    OPENDRIVE_SMH, OPENDRIVE_XLK)
from datetime import datetime
ET = pytz.timezone("America/New_York")
S = datetime(2025, 12, 1, tzinfo=ET)
E = datetime(2026, 4, 4, tzinfo=ET)
R = {}
for sym in ["SPY", "QQQ"]:
    df = get_minute_bars(sym, S, E, use_cache=True); df = prepare_features(df)
    p = dict(ORB_SHARED_DEFAULTS); p.update(SYMBOL_PROFILES.get(sym, {}))
    s = ORBBreakout(**p); df = s.generate_signals(df); r = run_backtest(df, s, sym)
    if hasattr(r, "daily_returns") and r.daily_returns is not None:
        R[f"ORB_{sym}"] = {"d": [str(x) for x in r.daily_returns.index], "r": r.daily_returns.values.tolist()}
for sym, cfg in [("SMH", OPENDRIVE_SMH), ("XLK", OPENDRIVE_XLK)]:
    df = get_minute_bars(sym, S, E, use_cache=True); df = prepare_features(df)
    s = OpeningDrive(**cfg); df = s.generate_signals(df); r = run_backtest(df, s, sym)
    if hasattr(r, "daily_returns") and r.daily_returns is not None:
        R[f"OD_{sym}"] = {"d": [str(x) for x in r.daily_returns.index], "r": r.daily_returns.values.tolist()}
cfg = PAIRS_GLD_TLT
g = get_minute_bars("GLD", S, E, use_cache=True); g = prepare_features(g)
t = get_minute_bars("TLT", S, E, use_cache=True)
tc = t.set_index("dt")["close"].rename("pair_close")
g = g.set_index("dt").join(tc, how="left").reset_index(); g["pair_close"] = g["pair_close"].ffill()
s = PairsSpread(lookback=cfg["lookback"], entry_zscore=cfg["entry_zscore"],
    exit_zscore=cfg["exit_zscore"], stale_bars=cfg["stale_bars"],
    last_entry_minute=cfg["last_entry_minute"])
g = s.generate_signals(g); r = run_backtest(g, s, "GLD")
if hasattr(r, "daily_returns") and r.daily_returns is not None:
    R["Pairs"] = {"d": [str(x) for x in r.daily_returns.index], "r": r.daily_returns.values.tolist()}
with open(sys.argv[1], "w") as f: json.dump(R, f)
'''

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=FUTURES_DIR) as tf:
    tf.write(eq_script); sp = tf.name
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=FUTURES_DIR) as tf:
    op = tf.name
r = subprocess.run([sys.executable, sp, op], capture_output=True, text=True, timeout=300, cwd=EQUITY_DIR)
with open(op) as f:
    eq = json.load(f)
os.unlink(sp); os.unlink(op)

equity_streams = {}
for name, data in eq.items():
    idx = pd.to_datetime(data["d"], utc=True).date
    equity_streams[name] = pd.Series(data["r"], index=idx)


def project(daily_dollar_pnl, starting_balance, n_days=DAYS_3MO, n_sims=N_SIMS, seed=42):
    np.random.seed(seed)
    sims = np.zeros(n_sims)
    paths_by_month = {1: [], 2: [], 3: []}
    for i in range(n_sims):
        sampled = np.random.choice(daily_dollar_pnl, size=n_days, replace=True)
        cumul = np.cumsum(sampled)
        sims[i] = cumul[-1]
        paths_by_month[1].append(cumul[MONTH_DAYS - 1])
        paths_by_month[2].append(cumul[2 * MONTH_DAYS - 1])
        paths_by_month[3].append(cumul[3 * MONTH_DAYS - 1])
    result = {
        "daily_mean": float(np.mean(daily_dollar_pnl)),
        "daily_std": float(np.std(daily_dollar_pnl)),
        "sims": sims,
    }
    for m in [1, 2, 3]:
        arr = np.array(paths_by_month[m])
        result[f"m{m}"] = {
            "bad": starting_balance + np.percentile(arr, 10),
            "avg": starting_balance + np.percentile(arr, 50),
            "great": starting_balance + np.percentile(arr, 90),
        }
    result["p_loss"] = float((sims < 0).mean())
    result["p_halved"] = float((sims < -starting_balance * 0.5).mean())
    median_cumul = np.median(np.array([np.cumsum(np.random.choice(daily_dollar_pnl, size=n_days, replace=True))
                                        for _ in range(1000)]), axis=0)
    dd = np.min(median_cumul - np.maximum.accumulate(median_cumul))
    result["max_dd"] = float(dd)
    result["worst_day"] = float(np.percentile(daily_dollar_pnl, 1))
    result["bad"] = starting_balance + float(np.percentile(sims, 10))
    result["avg"] = starting_balance + float(np.percentile(sims, 50))
    result["great"] = starting_balance + float(np.percentile(sims, 90))
    return result


def scale_fut(daily_ret_series, symbol, contracts):
    typ = {"MES": 7000, "MNQ": 26000, "MGC": 4800}[symbol]
    return daily_ret_series.values * typ * MULT[symbol] * contracts


def scale_eq(daily_ret, capital_share):
    return daily_ret.values * capital_share


# Scenario 1: $1M all 7
def sc1():
    account = 1_000_000
    sizes = {"ORB_MES": 30, "ORB_MNQ": 20, "VWAP_MES": 30, "VWAP_MNQ": 20, "VWAP_MGC": 5, "ON_MNQ": 1}
    streams = []
    tpd = 0
    for key, size in sizes.items():
        sym = key.split("_")[-1]
        s = futures_streams.get(key)
        if s is None: continue
        d = scale_fut(s, sym, size)
        active = (s != 0).sum()
        t = active / len(s) if len(s) > 0 else 0
        tpd += t
        d = d - t * COMM_RT * size
        streams.append(d)
    per_strat_cap = 200_000
    for name in ["ORB_SPY", "ORB_QQQ", "OD_SMH", "OD_XLK", "Pairs"]:
        s = equity_streams.get(name)
        if s is None: continue
        streams.append(scale_eq(s, per_strat_cap))
        active = (s != 0).sum()
        tpd += active / len(s) if len(s) > 0 else 0
    min_len = min(len(d) for d in streams)
    combined = np.sum([d[:min_len] for d in streams], axis=0)
    return {"name": "$1M all 7", "starting": account, "n_strats": 7,
            "strategies": "F_ORB, F_VWAP, ON, E_ORB, OD, Pairs",
            "trades_per_day": tpd, "combined_daily": combined,
            "commissions_approx": tpd * 63 * 106 * COMM_RT}


def sc2():
    account = 1_000_000
    sizes = {"ORB_MES": 30, "ORB_MNQ": 20, "VWAP_MES": 30, "VWAP_MNQ": 20, "VWAP_MGC": 5, "ON_MNQ": 1}
    streams = []
    tpd = 0
    for key, size in sizes.items():
        sym = key.split("_")[-1]
        s = futures_streams.get(key)
        if s is None: continue
        d = scale_fut(s, sym, size)
        active = (s != 0).sum()
        t = active / len(s) if len(s) > 0 else 0
        tpd += t
        d = d - t * COMM_RT * size
        streams.append(d)
    min_len = min(len(d) for d in streams)
    combined = np.sum([d[:min_len] for d in streams], axis=0)
    return {"name": "$1M futures only", "starting": account, "n_strats": 3,
            "strategies": "F_ORB, F_VWAP, ON",
            "trades_per_day": tpd, "combined_daily": combined,
            "commissions_approx": tpd * 63 * 106 * COMM_RT}


def sc3():
    """5k: VWAP and ORB blocked by risk sizer. Only ON fits."""
    account = 5000
    s = futures_streams.get("ON_MNQ")
    d = scale_fut(s, "MNQ", 1)
    active = (s != 0).sum()
    tpd = active / len(s) if len(s) > 0 else 0
    d = d - tpd * COMM_RT * 1
    return {"name": "$5k futures-attempt", "starting": account, "n_strats": 1,
            "strategies": "ON(MNQ) only (ORB+VWAP blocked by $50 risk cap)",
            "trades_per_day": tpd, "combined_daily": d,
            "commissions_approx": tpd * 63 * 1 * COMM_RT}


def sc4():
    account = 5000
    s = futures_streams.get("ON_MNQ")
    d = scale_fut(s, "MNQ", 1)
    active = (s != 0).sum()
    tpd = active / len(s) if len(s) > 0 else 0
    d = d - tpd * COMM_RT * 1
    return {"name": "$5k best single (ON_MNQ)", "starting": account, "n_strats": 1,
            "strategies": "ON(MNQ) 1 contract",
            "trades_per_day": tpd, "combined_daily": d,
            "commissions_approx": tpd * 63 * 1 * COMM_RT}


scenarios = [sc1(), sc2(), sc3(), sc4()]
for s in scenarios:
    s["proj"] = project(s["combined_daily"], s["starting"])

# Print per-scenario
for s in scenarios:
    p = s["proj"]
    print("=" * 80)
    print(f"SCENARIO: {s['name']}")
    print(f"Starting: ${s['starting']:,} | Strategies: {s['strategies']}")
    print(f"Trades/day: {s['trades_per_day']:.1f}")
    print(f"Daily mean P&L: ${p['daily_mean']:+,.0f} | Std: ${p['daily_std']:,.0f}")
    print()
    print(f"{'Month':<8} {'Bad (10%)':>15} {'Expected':>15} {'Great (90%)':>15}")
    for m in [1, 2, 3]:
        mp = p[f"m{m}"]
        print(f"M{m:<7} ${mp['bad']:>14,.0f} ${mp['avg']:>14,.0f} ${mp['great']:>14,.0f}")
    print()
    print(f"P(loss over 3mo): {p['p_loss']:.1%}")
    print(f"P(acct halved): {p['p_halved']:.1%}")
    print(f"Est. max drawdown: ${p['max_dd']:,.0f} ({abs(p['max_dd'])/s['starting']*100:.1f}%)")
    print(f"Worst single day (1%ile): ${p['worst_day']:,.0f}")
    print(f"Approx commissions over 3mo: ${s['commissions_approx']:,.0f}")
    gross = p['avg'] - s['starting']
    if gross != 0:
        print(f"Commissions as % of gross P&L: {abs(s['commissions_approx'] / gross) * 100:.1f}%")
    print(f"3-MONTH RESULT: Bad ${p['bad']:,.0f} | Expected ${p['avg']:,.0f} | Great ${p['great']:,.0f}")
    print()

# Comparison
print("=" * 135)
print("COMPARISON TABLE")
print("=" * 135)
print(f"{'Scenario':<30} {'Start':>12} {'Strats':>7} {'Tr/d':>6} "
      f"{'Expected':>13} {'Bad':>13} {'Great':>13} {'P(Loss)':>8} {'MaxDD':>10} {'Comms':>10}")
print("-" * 135)
for s in scenarios:
    p = s["proj"]
    exp_pnl = p["avg"] - s["starting"]
    bad_pnl = p["bad"] - s["starting"]
    grt_pnl = p["great"] - s["starting"]
    print(f"{s['name']:<30} ${s['starting']:>11,} {s['n_strats']:>7} {s['trades_per_day']:>6.1f} "
          f"${exp_pnl:>+12,.0f} ${bad_pnl:>+12,.0f} ${grt_pnl:>+12,.0f} "
          f"{p['p_loss']:>7.1%} ${abs(p['max_dd']):>9,.0f} ${s['commissions_approx']:>9,.0f}")
