#!/usr/bin/env python3
"""Risk-parity portfolio allocation: compute weights from dev, verify on OOS."""

import sys, os, io, warnings
from datetime import datetime
import pytz, numpy as np, pandas as pd
from scipy.optimize import minimize

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

STREAM_NAMES = ["ORB_SPY", "ORB_QQQ", "Pairs_GLD_TLT", "OD_SMH", "OD_XLK"]

# ── Load data ──
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


def run_period(df, strat, sym, start_dt, end_dt):
    mask = df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())
    p = df[mask].copy()
    if len(p) < 100:
        return None
    p = strat.generate_signals(p)
    r = run_backtest(p, strat, sym)
    return r.daily_returns


def portfolio_metrics(daily_returns_df, weights=None):
    """Compute portfolio metrics with given weights (default equal)."""
    if weights is None:
        weights = np.ones(daily_returns_df.shape[1]) / daily_returns_df.shape[1]
    port = daily_returns_df.values @ weights
    port = pd.Series(port, index=daily_returns_df.index)
    if len(port) < 5 or port.std() == 0:
        return {"sharpe": 0, "sortino": 0, "return": 0, "max_dd": 0}
    sharpe = (port.mean() / port.std()) * np.sqrt(252)
    ret = (1 + port).prod() - 1
    cum = (1 + port).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    ds = port[port < 0]
    dss = ds.std() if len(ds) > 1 else port.std()
    sortino = (port.mean() / dss) * np.sqrt(252) if dss > 0 else 0
    return {"sharpe": sharpe, "sortino": sortino, "return": ret, "max_dd": dd}


def solve_risk_parity(cov_matrix):
    """Solve for risk-parity weights: each asset contributes equal risk."""
    n = cov_matrix.shape[0]

    def risk_contrib(w):
        port_vol = np.sqrt(w @ cov_matrix @ w)
        marginal = cov_matrix @ w
        rc = w * marginal / port_vol
        return rc

    def objective(w):
        rc = risk_contrib(w)
        target = np.ones(n) / n  # Equal risk contribution
        return np.sum((rc / rc.sum() - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.05, 0.50)] * n  # Min 5%, max 50% per stream
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        print(f"  WARNING: optimizer failed ({result.message}), using equal weights", flush=True)
        return np.ones(n) / n


# ── Build daily returns for each stream ──
print("Running backtests...", flush=True)

strat_configs = [
    ("ORB_SPY", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["SPY"]}), data["SPY"], "SPY"),
    ("ORB_QQQ", ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["QQQ"]}), data["QQQ"], "QQQ"),
    ("Pairs_GLD_TLT", PairsSpread(lookback=120, entry_zscore=2.0, exit_zscore=0.5,
                                   stale_bars=90, last_entry_minute=900), gld_full, "GLD"),
    ("OD_SMH", OpeningDrive(**OPENDRIVE_SMH), data["SMH"], "SMH"),
    ("OD_XLK", OpeningDrive(**OPENDRIVE_XLK), data["XLK"], "XLK"),
]

dev_returns = {}
oos_returns = {}

for name, strat, df, sym in strat_configs:
    dr_dev = run_period(df, strat, sym, START, DEV_END)
    dr_oos = run_period(df, strat, sym, OOS_START, END)
    if dr_dev is not None:
        dev_returns[name] = dr_dev
    if dr_oos is not None:
        oos_returns[name] = dr_oos
    print(f"  {name}: dev={len(dr_dev) if dr_dev is not None else 0} days, "
          f"oos={len(dr_oos) if dr_oos is not None else 0} days", flush=True)

# Build aligned DataFrames
dev_df = pd.DataFrame(dev_returns).fillna(0)
oos_df = pd.DataFrame(oos_returns).fillna(0)

# ── Compute risk-parity weights from DEV period ──
print(f"\n{'='*80}", flush=True)
print("RISK-PARITY WEIGHT COMPUTATION (from dev period covariance)", flush=True)
print(f"{'='*80}", flush=True)

dev_cov = dev_df.cov().values * 252  # Annualize
rp_weights = solve_risk_parity(dev_cov)
eq_weights = np.ones(5) / 5

print(f"\n  {'Stream':<18} {'Equal':>7} {'RiskParity':>11}", flush=True)
print(f"  {'-'*17} {'-'*7} {'-'*11}", flush=True)
for i, name in enumerate(STREAM_NAMES):
    print(f"  {name:<18} {eq_weights[i]:>6.1%} {rp_weights[i]:>10.1%}", flush=True)

# Risk contribution check
dev_cov_daily = dev_df.cov().values
def risk_contributions(w, cov):
    port_vol = np.sqrt(w @ cov @ w)
    mc = cov @ w
    rc = w * mc / port_vol
    return rc / rc.sum()

rc_eq = risk_contributions(eq_weights, dev_cov_daily)
rc_rp = risk_contributions(rp_weights, dev_cov_daily)

print(f"\n  Risk contributions (dev period):", flush=True)
print(f"  {'Stream':<18} {'EqualWt':>8} {'RiskParity':>11}", flush=True)
print(f"  {'-'*17} {'-'*8} {'-'*11}", flush=True)
for i, name in enumerate(STREAM_NAMES):
    print(f"  {name:<18} {rc_eq[i]:>7.1%} {rc_rp[i]:>10.1%}", flush=True)

# ── Apply weights to BOTH periods ──
print(f"\n{'='*80}", flush=True)
print("BEFORE / AFTER COMPARISON", flush=True)
print(f"{'='*80}", flush=True)

for period_name, period_df in [("DEV (Jan-Nov 2025)", dev_df), ("LOCKED OOS (Dec 2025-Apr 2026)", oos_df)]:
    eq_m = portfolio_metrics(period_df, eq_weights)
    rp_m = portfolio_metrics(period_df, rp_weights)

    print(f"\n  {period_name}:", flush=True)
    print(f"  {'Metric':<12} {'EqualWt':>10} {'RiskParity':>11} {'Delta':>10} {'Better?':>8}", flush=True)
    print(f"  {'-'*11} {'-'*10} {'-'*11} {'-'*10} {'-'*8}", flush=True)

    for metric in ["sharpe", "sortino", "return", "max_dd"]:
        eq_val = eq_m[metric]
        rp_val = rp_m[metric]
        delta = rp_val - eq_val

        if metric == "max_dd":
            better = "YES" if rp_val > eq_val else "no"  # Less negative = better
            print(f"  {'MaxDD':<12} {eq_val:>9.2%} {rp_val:>10.2%} {delta:>+9.2%} {better:>8}", flush=True)
        elif metric == "return":
            better = "YES" if rp_val > eq_val else "no"
            print(f"  {'Return':<12} {eq_val:>+9.2%} {rp_val:>+10.2%} {delta:>+9.2%} {better:>8}", flush=True)
        else:
            better = "YES" if rp_val > eq_val else "no"
            print(f"  {metric.capitalize():<12} {eq_val:>10.2f} {rp_val:>11.2f} {delta:>+10.2f} {better:>8}", flush=True)

# ── Verdict ──
dev_eq = portfolio_metrics(dev_df, eq_weights)
dev_rp = portfolio_metrics(dev_df, rp_weights)
oos_eq = portfolio_metrics(oos_df, eq_weights)
oos_rp = portfolio_metrics(oos_df, rp_weights)

dev_improves = dev_rp["sharpe"] > dev_eq["sharpe"]
oos_improves = oos_rp["sharpe"] > oos_eq["sharpe"]

print(f"\n{'='*80}", flush=True)
print("VERDICT", flush=True)
print(f"{'='*80}", flush=True)
print(f"\n  Dev Sharpe:  {dev_eq['sharpe']:.2f} -> {dev_rp['sharpe']:.2f} ({'IMPROVES' if dev_improves else 'WORSENS'})", flush=True)
print(f"  OOS Sharpe:  {oos_eq['sharpe']:.2f} -> {oos_rp['sharpe']:.2f} ({'IMPROVES' if oos_improves else 'WORSENS'})", flush=True)
print(f"  OOS MaxDD:   {oos_eq['max_dd']:.2%} -> {oos_rp['max_dd']:.2%}", flush=True)
print(f"  OOS Sortino: {oos_eq['sortino']:.2f} -> {oos_rp['sortino']:.2f}", flush=True)

if dev_improves and oos_improves:
    print(f"\n  ACCEPT: Risk-parity improves Sharpe on BOTH periods.", flush=True)
    print(f"\n  Weights to use:", flush=True)
    for i, name in enumerate(STREAM_NAMES):
        print(f"    {name}: {rp_weights[i]:.3f}", flush=True)
elif dev_improves or oos_improves:
    print(f"\n  REJECT: Only improves on one period. Keeping equal weights.", flush=True)
else:
    print(f"\n  REJECT: Does not improve either period. Keeping equal weights.", flush=True)

print(f"\n{'='*80}", flush=True)
print("RISK-PARITY ANALYSIS COMPLETE", flush=True)
print(f"{'='*80}", flush=True)
