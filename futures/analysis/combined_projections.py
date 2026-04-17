#!/usr/bin/env python3
"""Combined annual projections for ALL strategies across both repos.

Futures (IB): ORB MES+MNQ, VWAP Reversion MES+MNQ, Overnight MNQ
Equities (Alpaca): ORB SPY+QQQ, OpenDrive SMH+XLK, Pairs GLD+TLT

Uses locked OOS data from both repos. Runs equity backtests in a
subprocess to avoid import conflicts between the two trading packages.
"""

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

from trading.data.contracts import CONTRACTS
from trading.config import (
    ORB_SHARED_DEFAULTS as F_ORB_DEFAULTS,
    SYMBOL_PROFILES as F_PROFILES,
    VWAP_REVERSION_DEFAULTS,
    OVERNIGHT_REVERSION_DEFAULTS,
)

print("=" * 80)
print("  COMBINED PORTFOLIO PROJECTIONS — ALL 8 STRATEGY STREAMS")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Futures locked OOS daily returns
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading futures OOS data...", flush=True)

from analysis.intraday_strategies import load_cash_data
from trading.strategies.orb import ORBBreakout as FORB
from trading.strategies.vwap_reversion import VWAPReversion

# NOTE: Futures ORB on MES/MNQ mirrors equity ORB on SPY/QQQ (same index).
# We use equity ORB results as the ORB alpha source for both brokers.
# Futures-UNIQUE strategies are VWAP Reversion and Overnight Reversion.

futures_streams = {}

for sym in ["MES", "MNQ"]:
    contract = CONTRACTS[sym]
    df = load_cash_data(sym)
    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.8)
    oos_dates = set(dates[split:])
    oos_df = df[df["date"].isin(oos_dates)].copy()

    # VWAP Reversion (futures-unique, 10 AM - 3 PM)
    vwap = VWAPReversion(**VWAP_REVERSION_DEFAULTS)
    vwap_df = vwap.generate_signals(oos_df.copy())
    vwap_df["position"] = vwap_df["signal"].shift(1).fillna(0)
    vwap_df["bar_ret"] = vwap_df["close"].pct_change().fillna(0)
    vwap_df["strat_ret"] = vwap_df["position"] * vwap_df["bar_ret"]
    vwap_daily = vwap_df.groupby("date")["strat_ret"].sum()
    futures_streams[f"F_VWAP_{sym}"] = vwap_daily

    dr = futures_streams[f"F_VWAP_{sym}"]
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    print(f"  F_VWAP_{sym}: {len(dr)} OOS days, Sharpe {sh:.2f}", flush=True)

# Overnight MNQ
cache = os.path.join(FUTURES_DIR, "data", "cache")
on_file = os.path.join(cache, "MNQ_futures_full_1min.csv")
if os.path.exists(on_file):
    on_df = pd.read_csv(on_file)
    on_df["dt"] = pd.to_datetime(on_df["dt"], utc=True).dt.tz_convert(ET)
    on_df["date"] = on_df["dt"].dt.date

    from trading.strategies.overnight_reversion import OvernightReversion
    on_strat = OvernightReversion(**OVERNIGHT_REVERSION_DEFAULTS)
    on_df = on_strat.generate_signals(on_df)
    on_df["position"] = on_df["signal"].shift(1).fillna(0)
    on_df["bar_ret"] = on_df["close"].pct_change().fillna(0)
    on_df["strat_ret"] = on_df["position"] * on_df["bar_ret"]

    dates = sorted(on_df["date"].unique())
    split = int(len(dates) * 0.8)
    oos_dates = set(dates[split:])
    on_oos = on_df[on_df["date"].isin(oos_dates)]
    on_daily = on_oos.groupby("date")["strat_ret"].sum()
    futures_streams["F_ON_MNQ"] = on_daily
    sh = on_daily.mean() / on_daily.std() * np.sqrt(252) if on_daily.std() > 0 else 0
    print(f"  F_ON_MNQ: {len(on_daily)} OOS days, Sharpe {sh:.2f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Equity locked OOS daily returns (run in subprocess)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading equity OOS data (subprocess)...", flush=True)

equity_script = r'''
import sys, os, json, warnings
import numpy as np, pandas as pd, pytz
warnings.filterwarnings("ignore")

EQUITY_DIR = r"''' + EQUITY_DIR.replace("\\", "\\\\") + r'''"
sys.path.insert(0, EQUITY_DIR)

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.backtest.engine import run_backtest
from trading.strategies.orb import ORBBreakout
from trading.strategies.opening_drive import OpeningDrive
from trading.strategies.pairs_spread import PairsSpread
from trading.config import (
    ORB_SHARED_DEFAULTS, SYMBOL_PROFILES,
    PAIRS_GLD_TLT, OPENDRIVE_SMH, OPENDRIVE_XLK,
)
from datetime import datetime

ET = pytz.timezone("America/New_York")
OOS_START = datetime(2025, 12, 1, tzinfo=ET)
OOS_END = datetime(2026, 4, 4, tzinfo=ET)

results = {}

# ORB SPY + QQQ
for sym in ["SPY", "QQQ"]:
    df = get_minute_bars(sym, OOS_START, OOS_END, use_cache=True)
    df = prepare_features(df)
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(SYMBOL_PROFILES.get(sym, {}))
    strat = ORBBreakout(**params)
    df = strat.generate_signals(df)
    r = run_backtest(df, strat, sym)
    if hasattr(r, 'daily_returns') and r.daily_returns is not None:
        dr = r.daily_returns
        results[f"E_ORB_{sym}"] = {
            "dates": [str(d) for d in dr.index],
            "returns": dr.values.tolist(),
            "trades": r.num_trades,
        }

# OpenDrive SMH + XLK
for sym, cfg_dict in [("SMH", OPENDRIVE_SMH), ("XLK", OPENDRIVE_XLK)]:
    df = get_minute_bars(sym, OOS_START, OOS_END, use_cache=True)
    df = prepare_features(df)
    strat = OpeningDrive(**cfg_dict)
    df = strat.generate_signals(df)
    r = run_backtest(df, strat, sym)
    if hasattr(r, 'daily_returns') and r.daily_returns is not None:
        dr = r.daily_returns
        results[f"E_OD_{sym}"] = {
            "dates": [str(d) for d in dr.index],
            "returns": dr.values.tolist(),
            "trades": r.num_trades,
        }

# Pairs GLD/TLT
cfg = PAIRS_GLD_TLT
gld_df = get_minute_bars("GLD", OOS_START, OOS_END, use_cache=True)
gld_df = prepare_features(gld_df)
tlt_df = get_minute_bars("TLT", OOS_START, OOS_END, use_cache=True)
tlt_close = tlt_df.set_index("dt")["close"].rename("pair_close")
gld_df = gld_df.set_index("dt").join(tlt_close, how="left").reset_index()
gld_df["pair_close"] = gld_df["pair_close"].ffill()
strat = PairsSpread(
    lookback=cfg["lookback"], entry_zscore=cfg["entry_zscore"],
    exit_zscore=cfg["exit_zscore"], stale_bars=cfg["stale_bars"],
    last_entry_minute=cfg["last_entry_minute"],
)
gld_df = strat.generate_signals(gld_df)
r = run_backtest(gld_df, strat, "GLD")
if hasattr(r, 'daily_returns') and r.daily_returns is not None:
    dr = r.daily_returns
    results["E_Pairs_GLD_TLT"] = {
        "dates": [str(d) for d in dr.index],
        "returns": dr.values.tolist(),
        "trades": r.num_trades,
    }

OUTPUT_PATH = sys.argv[1]
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f)
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=FUTURES_DIR) as tf:
    tf.write(equity_script)
    eq_script_path = tf.name

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=FUTURES_DIR) as tf:
    eq_output_path = tf.name

try:
    result = subprocess.run(
        [sys.executable, eq_script_path, eq_output_path],
        capture_output=True, text=True, timeout=300,
        cwd=EQUITY_DIR,
    )
    if result.returncode != 0:
        print(f"  [ERROR] Equity subprocess failed:\n{result.stderr[:500]}", flush=True)
    else:
        with open(eq_output_path) as f:
            eq_results = json.load(f)
finally:
    os.unlink(eq_script_path)
    if os.path.exists(eq_output_path):
        os.unlink(eq_output_path)

equity_streams = {}
for name, data in eq_results.items():
    idx = pd.to_datetime(data["dates"], utc=True)
    dr = pd.Series(data["returns"], index=idx)
    dr.index = dr.index.date  # normalize to date only
    equity_streams[name] = dr
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    print(f"  {name}: {data['trades']} trades, Sharpe {sh:.2f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Combined analysis
# ═══════════════════════════════════════════════════════════════════════════════

all_streams = {**futures_streams, **equity_streams}
all_daily = pd.DataFrame(all_streams).fillna(0)
trading_days = len(all_daily)
capital = 100_000

print(f"\n  Total streams: {len(all_streams)}, OOS days: {trading_days}")

# Individual stats
print(f"\n{'=' * 80}")
print("  INDIVIDUAL STREAM STATS (locked OOS)")
print(f"{'=' * 80}")
print(f"  {'Stream':<20} {'Sharpe':>7} {'Ann.Ret':>8} {'MaxDD':>7} {'Days':>5} {'$/day':>8}")
print(f"  {'-'*19} {'-'*7} {'-'*8} {'-'*7} {'-'*5} {'-'*8}")

for name in sorted(all_daily.columns):
    dr = all_daily[name]
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    ann_ret = dr.mean() * 252
    cum = (1 + dr).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    active = (dr.abs() > 0).sum()
    daily_pnl = dr.mean() * capital
    print(f"  {name:<20} {sh:>7.2f} {ann_ret:>+8.2%} {dd:>7.2%} {active:>5} ${daily_pnl:>+7.0f}")


# Correlation matrix
print(f"\n{'=' * 80}")
print("  CORRELATION MATRIX (locked OOS daily returns)")
print(f"{'=' * 80}")

corr = all_daily.corr()
abbr = {}
for n in corr.columns:
    a = n.replace("F_", "f").replace("E_", "e").replace("Pairs_", "P")
    abbr[n] = a[:9]

print(f"\n  {'':>10}", end="")
for n in corr.columns:
    print(f" {abbr[n]:>9}", end="")
print()
for n in corr.columns:
    print(f"  {abbr[n]:<10}", end="")
    for n2 in corr.columns:
        print(f" {corr.loc[n, n2]:>9.2f}", end="")
    print()


# Portfolio configs
print(f"\n{'=' * 80}")
print("  PORTFOLIO CONFIGURATIONS (equal-weight)")
print(f"{'=' * 80}")

fut_cols = [c for c in all_daily.columns if c.startswith("F_")]
eq_cols = [c for c in all_daily.columns if c.startswith("E_")]

configs = {
    "Futures only (ORB+VWAP+ON)": fut_cols,
    "Equities only (ORB+OD+Pairs)": eq_cols,
    "ALL STREAMS COMBINED": list(all_daily.columns),
}

print(f"\n  {'Config':<35} {'Sharpe':>7} {'Sortino':>8} {'Ann.Ret':>8} {'MaxDD':>7} {'Ann.P&L':>10} {'Streams':>7}")
print(f"  {'-'*34} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*10} {'-'*7}")

for config_name, cols in configs.items():
    port = all_daily[cols].mean(axis=1)
    sh = port.mean() / port.std() * np.sqrt(252) if port.std() > 0 else 0
    ds = port[port < 0]
    sortino = port.mean() / ds.std() * np.sqrt(252) if len(ds) > 1 and ds.std() > 0 else 0
    ann_ret = port.mean() * 252
    cum = (1 + port).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    ann_pnl = ann_ret * capital
    print(f"  {config_name:<35} {sh:>7.2f} {sortino:>8.2f} {ann_ret:>+8.2%} {dd:>7.2%} ${ann_pnl:>+9,.0f} {len(cols):>7}")


# Annual projections
print(f"\n{'=' * 80}")
print("  ANNUAL PROJECTIONS ($100K account, equal-weight)")
print(f"{'=' * 80}")

combined = all_daily.mean(axis=1)
daily_mean = combined.mean()
daily_std = combined.std()
sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0

trades_per_day = 0
for name in all_daily.columns:
    active = (all_daily[name].abs() > 0).sum()
    trades_per_day += active / trading_days

cum = (1 + combined).cumprod()
max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

print(f"\n  Combined Sharpe:           {sharpe:.2f}")
print(f"  Annual return (mean):      {daily_mean * 252:+.2%} (${capital * daily_mean * 252:+,.0f})")
print(f"  Annual volatility:         {daily_std * np.sqrt(252):.2%}")
print(f"  Max drawdown (OOS):        {max_dd:.2%}")
print(f"  Est. trades per day:       {trades_per_day:.1f}")

# Monte Carlo
print(f"\n  --- Year-End Balance Scenarios ($100K start) ---")
np.random.seed(42)
annual_pnl = np.array([
    np.sum(np.random.choice(combined.values, size=252, replace=True))
    for _ in range(10000)
]) * capital

p5, p25, p50, p75, p95 = [np.percentile(annual_pnl, p) for p in [5, 25, 50, 75, 95]]

print(f"\n  {'Scenario':<25} {'P&L':>12} {'End Balance':>15} {'Probability':>12}")
print(f"  {'-'*24} {'-'*12} {'-'*15} {'-'*12}")
print(f"  {'Bad (5th pctile)':<25} ${p5:>+11,.0f} ${capital + p5:>14,.0f} {'5%':>12}")
print(f"  {'Below avg (25th)':<25} ${p25:>+11,.0f} ${capital + p25:>14,.0f} {'25%':>12}")
print(f"  {'Average (median)':<25} ${p50:>+11,.0f} ${capital + p50:>14,.0f} {'50%':>12}")
print(f"  {'Above avg (75th)':<25} ${p75:>+11,.0f} ${capital + p75:>14,.0f} {'75%':>12}")
print(f"  {'Great (95th pctile)':<25} ${p95:>+11,.0f} ${capital + p95:>14,.0f} {'95%':>12}")

prob_profit = (annual_pnl > 0).mean()
prob_20k = (annual_pnl > 20000).mean()
prob_loss_10k = (annual_pnl < -10000).mean()

print(f"\n  P(profitable year):        {prob_profit:.1%}")
print(f"  P(>$20K profit):           {prob_20k:.1%}")
print(f"  P(>$10K loss):             {prob_loss_10k:.1%}")

print(f"\n{'=' * 80}")
print("  PROJECTION COMPLETE")
print(f"{'=' * 80}")
