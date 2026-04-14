#!/usr/bin/env python3
"""Volatility-targeted position sizing: scale positions to constant risk.

Replace fixed 30%/20% sizing with: target_vol / realized_vol.
Cap so it never exceeds current max (can only size DOWN, never UP).
Test on dev and OOS — keep only if MaxDD improves on both without
Sharpe dropping more than 10%.
"""

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
END = datetime(2026, 4, 14, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

TARGET_VOL = 0.10  # 10% annualized target volatility
VOL_LOOKBACK = 20  # 20 trading days

print("Loading data...", flush=True)
data = {}
for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    df = get_minute_bars(sym, START, END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df

gld_full = data["GLD"].copy()
tc = data["TLT"].set_index("dt")["close"].rename("pair_close")
gld_full = gld_full.set_index("dt").join(tc, how="left").reset_index()
gld_full["pair_close"] = gld_full["pair_close"].ffill()


def compute_vol_sized_pct(df, base_max_pct):
    """Compute per-day position sizing based on trailing volatility.

    Returns a dict of {date: position_pct} where position_pct is
    min(target_vol / realized_vol, base_max_pct).
    """
    daily_close = df.groupby("date")["close"].last()
    daily_ret = daily_close.pct_change().dropna()

    # 20-day trailing realized vol (annualized)
    realized_vol = daily_ret.rolling(VOL_LOOKBACK, min_periods=10).std() * np.sqrt(252)

    sizing = {}
    dates = sorted(df["date"].unique())
    for d in dates:
        rv = realized_vol.get(d, None)
        if rv is not None and rv > 0 and not pd.isna(rv):
            raw_pct = TARGET_VOL / rv
            # Cap at base max — can only size DOWN, never UP
            sizing[d] = min(raw_pct, base_max_pct)
        else:
            sizing[d] = base_max_pct  # Default to base if no vol data

    return sizing


def run_with_vol_sizing(df, strat, sym, base_max_pct, start_dt, end_dt):
    """Run backtest with per-day volatility-targeted position sizing.

    Since the backtest engine uses a single position_pct, we simulate
    vol-targeting by scaling each trade's PnL by (vol_sized_pct / base_pct).
    This is equivalent to running at vol-targeted size.
    """
    mask = df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())
    period_df = df[mask].copy()
    if len(period_df) < 100:
        return None, None

    # Get per-day sizing
    sizing = compute_vol_sized_pct(df, base_max_pct)

    # Run at base sizing
    period_df = strat.generate_signals(period_df)
    r = run_backtest(period_df, strat, sym, position_pct=base_max_pct)

    # Scale each trade's PnL by the vol-sizing ratio for that day
    scaled_trades_pnl = []
    for t in r.trades:
        trade_date = t.entry_time.date() if hasattr(t.entry_time, 'date') else t.entry_time
        vol_pct = sizing.get(trade_date, base_max_pct)
        scale = vol_pct / base_max_pct
        scaled_trades_pnl.append(t.pnl * scale)

    # Reconstruct daily returns by scaling the equity curve
    # Simpler approach: scale daily returns by avg daily sizing ratio
    daily_dates = period_df.groupby("date").first().index
    dr = r.daily_returns
    if dr is None or len(dr) < 5:
        return None, None

    # For each day, compute the sizing ratio
    scaled_dr = dr.copy()
    for i, idx in enumerate(scaled_dr.index):
        d = idx.date() if hasattr(idx, 'date') else idx
        vol_pct = sizing.get(d, base_max_pct)
        scale = vol_pct / base_max_pct
        scaled_dr.iloc[i] = dr.iloc[i] * scale

    return scaled_dr, r


def metrics(dr):
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return {"sharpe": 0, "sortino": 0, "return": 0, "max_dd": 0}
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
    ret = (1 + dr).prod() - 1
    cum = (1 + dr).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    ds = dr[dr < 0]
    dss = ds.std() if len(ds) > 1 else dr.std()
    sortino = (dr.mean() / dss) * np.sqrt(252) if dss > 0 else 0
    return {"sharpe": sharpe, "sortino": sortino, "return": ret, "max_dd": dd}


# ── Run all strategies with fixed vs vol-targeted sizing ──

STRATS = [
    ("ORB_SPY", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["SPY"]}),
     data["SPY"], "SPY", MAX_POSITION_PCT),
    ("ORB_QQQ", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["QQQ"]}),
     data["QQQ"], "QQQ", MAX_POSITION_PCT),
    ("Pairs_GLD_TLT", PairsSpread(lookback=120, entry_zscore=2.0, exit_zscore=0.5,
                                   stale_bars=90, last_entry_minute=900),
     gld_full, "GLD", WAVE2_MAX_POSITION_PCT),
    ("OD_SMH", OpeningDrive(**OPENDRIVE_SMH),
     data["SMH"], "SMH", WAVE2_MAX_POSITION_PCT),
    ("OD_XLK", OpeningDrive(**OPENDRIVE_XLK),
     data["XLK"], "XLK", WAVE2_MAX_POSITION_PCT),
]

print(f"\n{'='*100}", flush=True)
print("VOLATILITY-TARGETED SIZING: BEFORE / AFTER", flush=True)
print(f"Target vol: {TARGET_VOL:.0%} annualized, lookback: {VOL_LOOKBACK} days", flush=True)
print(f"Cap: never exceed current max (30% ORB, 20% candidates)", flush=True)
print(f"{'='*100}", flush=True)

# Collect results for portfolio comparison
rp_weights = np.array([PORTFOLIO_WEIGHTS[n] for n in
                        ["ORB_SPY", "ORB_QQQ", "Pairs_GLD_TLT", "OD_SMH", "OD_XLK"]])

for period_name, start_dt, end_dt in [("DEV", START, DEV_END), ("LOCKED OOS", OOS_START, END)]:
    print(f"\n  {period_name}:", flush=True)
    print(f"  {'Strategy':<16} {'FixedS':>7} {'VolS':>7} {'dS':>6}"
          f" {'FixedSort':>9} {'VolSort':>8} {'FixedDD':>8} {'VolDD':>7} {'dDD':>7}", flush=True)
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*6}"
          f" {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*7}", flush=True)

    fixed_drs = {}
    vol_drs = {}

    for name, strat, df, sym, base_pct in STRATS:
        # Fixed sizing
        mask = df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())
        p = df[mask].copy()
        if len(p) < 100:
            continue
        p2 = strat.generate_signals(p)
        r_fixed = run_backtest(p2, strat, sym, position_pct=base_pct)
        m_fixed = metrics(r_fixed.daily_returns)
        fixed_drs[name] = r_fixed.daily_returns

        # Vol-targeted sizing
        dr_vol, _ = run_with_vol_sizing(df, strat, sym, base_pct, start_dt, end_dt)
        m_vol = metrics(dr_vol)
        vol_drs[name] = dr_vol

        ds = m_vol["sharpe"] - m_fixed["sharpe"]
        ddd = m_vol["max_dd"] - m_fixed["max_dd"]

        print(f"  {name:<16} {m_fixed['sharpe']:>7.2f} {m_vol['sharpe']:>7.2f} {ds:>+6.2f}"
              f" {m_fixed['sortino']:>9.2f} {m_vol['sortino']:>8.2f}"
              f" {m_fixed['max_dd']:>7.2%} {m_vol['max_dd']:>6.2%} {ddd:>+6.2%}", flush=True)

    # Portfolio comparison
    if len(fixed_drs) == 5 and len(vol_drs) == 5:
        fixed_df = pd.DataFrame(fixed_drs).fillna(0)
        vol_df = pd.DataFrame(vol_drs).fillna(0)

        port_fixed = pd.Series(fixed_df.values @ rp_weights, index=fixed_df.index)
        port_vol = pd.Series(vol_df.values @ rp_weights, index=vol_df.index)

        mf = metrics(port_fixed)
        mv = metrics(port_vol)

        print(f"\n  {'PORTFOLIO':<16} {mf['sharpe']:>7.2f} {mv['sharpe']:>7.2f} {mv['sharpe']-mf['sharpe']:>+6.2f}"
              f" {mf['sortino']:>9.2f} {mv['sortino']:>8.2f}"
              f" {mf['max_dd']:>7.2%} {mv['max_dd']:>6.2%} {mv['max_dd']-mf['max_dd']:>+6.2%}", flush=True)

# ── Final verdict ──
print(f"\n{'='*100}", flush=True)
print("VERDICT", flush=True)
print(f"{'='*100}", flush=True)

# Recompute for verdict
dev_fixed_drs = {}; dev_vol_drs = {}
oos_fixed_drs = {}; oos_vol_drs = {}

for name, strat, df, sym, base_pct in STRATS:
    for period, s, e, f_dict, v_dict in [
        ("dev", START, DEV_END, dev_fixed_drs, dev_vol_drs),
        ("oos", OOS_START, END, oos_fixed_drs, oos_vol_drs),
    ]:
        mask = df["date"].apply(lambda d: s.date() <= d <= e.date())
        p = df[mask].copy()
        if len(p) < 100: continue
        p2 = strat.generate_signals(p)
        r = run_backtest(p2, strat, sym, position_pct=base_pct)
        f_dict[name] = r.daily_returns
        dr_vol, _ = run_with_vol_sizing(df, strat, sym, base_pct, s, e)
        v_dict[name] = dr_vol

dev_fixed_port = pd.Series(pd.DataFrame(dev_fixed_drs).fillna(0).values @ rp_weights,
                           index=pd.DataFrame(dev_fixed_drs).fillna(0).index)
dev_vol_port = pd.Series(pd.DataFrame(dev_vol_drs).fillna(0).values @ rp_weights,
                         index=pd.DataFrame(dev_vol_drs).fillna(0).index)
oos_fixed_port = pd.Series(pd.DataFrame(oos_fixed_drs).fillna(0).values @ rp_weights,
                           index=pd.DataFrame(oos_fixed_drs).fillna(0).index)
oos_vol_port = pd.Series(pd.DataFrame(oos_vol_drs).fillna(0).values @ rp_weights,
                         index=pd.DataFrame(oos_vol_drs).fillna(0).index)

m_df = metrics(dev_fixed_port); m_dv = metrics(dev_vol_port)
m_of = metrics(oos_fixed_port); m_ov = metrics(oos_vol_port)

dev_dd_improves = m_dv["max_dd"] > m_df["max_dd"]  # Less negative = better
oos_dd_improves = m_ov["max_dd"] > m_of["max_dd"]
dev_sharpe_ok = m_dv["sharpe"] >= m_df["sharpe"] * 0.90  # Within 10%
oos_sharpe_ok = m_ov["sharpe"] >= m_of["sharpe"] * 0.90

print(f"\n  Dev  portfolio: Fixed Sharpe={m_df['sharpe']:.2f} MaxDD={m_df['max_dd']:.2%}"
      f" -> Vol Sharpe={m_dv['sharpe']:.2f} MaxDD={m_dv['max_dd']:.2%}", flush=True)
print(f"  OOS  portfolio: Fixed Sharpe={m_of['sharpe']:.2f} MaxDD={m_of['max_dd']:.2%}"
      f" -> Vol Sharpe={m_ov['sharpe']:.2f} MaxDD={m_ov['max_dd']:.2%}", flush=True)

print(f"\n  Dev MaxDD improves: {dev_dd_improves}  |  Dev Sharpe within 10%: {dev_sharpe_ok}", flush=True)
print(f"  OOS MaxDD improves: {oos_dd_improves}  |  OOS Sharpe within 10%: {oos_sharpe_ok}", flush=True)

if dev_dd_improves and oos_dd_improves and dev_sharpe_ok and oos_sharpe_ok:
    print(f"\n  ACCEPT: Vol-targeting improves MaxDD on both periods"
          f" without significant Sharpe loss.", flush=True)
else:
    reasons = []
    if not dev_dd_improves: reasons.append("dev MaxDD worsens")
    if not oos_dd_improves: reasons.append("oos MaxDD worsens")
    if not dev_sharpe_ok: reasons.append("dev Sharpe drops >10%")
    if not oos_sharpe_ok: reasons.append("oos Sharpe drops >10%")
    print(f"\n  REJECT: {', '.join(reasons)}. Keeping fixed sizing.", flush=True)

print(f"\n{'='*100}", flush=True)
print("VOL-TARGETING ANALYSIS COMPLETE", flush=True)
print(f"{'='*100}", flush=True)
