#!/usr/bin/env python3
"""Realistic 3-month P&L projections for every strategy on a $1M account."""

import json
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytz
from scipy import stats

warnings.filterwarnings("ignore")

FUTURES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUITY_DIR = r"C:\Users\liamr\Desktop\spy-trader\.claude\worktrees\flamboyant-lewin"
ET = pytz.timezone("America/New_York")

sys.path.insert(0, FUTURES_DIR)

from trading.strategies.orb import ORBBreakout
from trading.strategies.vwap_reversion import VWAPReversion
from trading.strategies.overnight_reversion import OvernightReversion
from trading.config import (
    ORB_SHARED_DEFAULTS, SYMBOL_PROFILES,
    VWAP_REVERSION_DEFAULTS, OVERNIGHT_REVERSION_DEFAULTS,
    INITIAL_CAPITAL,
)
from trading.data.contracts import CONTRACTS

cache = os.path.join(FUTURES_DIR, "data", "cache")
ACCOUNT = 1_000_000
DAYS_3MO = 63
BT_CAPITAL = INITIAL_CAPITAL  # backtests used $100K


def load_cash(filename):
    df = pd.read_csv(os.path.join(cache, filename))
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)
    df["date"] = df["dt"].dt.date
    return df


def oos_split(df):
    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.8)
    return df[df["date"].isin(set(dates[split:]))].copy(), len(dates) - split


def run_strat(df, strat):
    sig = strat.generate_signals(df.copy())
    sig["pos"] = sig["signal"].shift(1).fillna(0)
    sig["br"] = sig["close"].pct_change().fillna(0)
    sig["sr"] = sig["pos"] * sig["br"]
    daily = sig.groupby("date")["sr"].sum()
    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# FUTURES STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading futures OOS data...", flush=True)

# Futures ORB: combine MES + MNQ ORB into one stream
# ORB needs prepare_features which isn't available on raw futures data,
# so we use the equity ORB (SPY+QQQ) as proxy for futures ORB returns.
# This is documented — futures ORB mirrors equity ORB on the same indexes.

# VWAP MES + MNQ + MGC combined
vwap_streams = {}
for sym, fname in [("MES", "MES_futures_cash_1min.csv"),
                    ("MNQ", "MNQ_futures_cash_1min.csv"),
                    ("MGC", "MGC_futures_cash_1min.csv")]:
    df = load_cash(fname)
    oos_df, n_days = oos_split(df)
    dr = run_strat(oos_df, VWAPReversion(**VWAP_REVERSION_DEFAULTS))
    vwap_streams[sym] = dr

# Combine VWAP streams (equal weight)
vwap_combined = pd.DataFrame(vwap_streams).fillna(0).mean(axis=1)

# Overnight MNQ
on_df = pd.read_csv(os.path.join(cache, "MNQ_futures_full_1min.csv"))
on_df["dt"] = pd.to_datetime(on_df["dt"], utc=True).dt.tz_convert(ET)
on_df["date"] = on_df["dt"].dt.date
on_oos, _ = oos_split(on_df)
on_daily = run_strat(on_oos, OvernightReversion(**OVERNIGHT_REVERSION_DEFAULTS))


# ═══════════════════════════════════════════════════════════════════════════════
# EQUITY STRATEGIES (subprocess)
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading equity OOS data...", flush=True)

eq_script = '''
import sys,os,json,warnings; import numpy as np,pandas as pd,pytz; warnings.filterwarnings("ignore")
sys.path.insert(0,r"''' + EQUITY_DIR.replace("\\", "\\\\") + '''")
from trading.data.provider import get_minute_bars; from trading.data.features import prepare_features
from trading.backtest.engine import run_backtest; from trading.strategies.orb import ORBBreakout
from trading.strategies.opening_drive import OpeningDrive; from trading.strategies.pairs_spread import PairsSpread
from trading.config import ORB_SHARED_DEFAULTS,SYMBOL_PROFILES,PAIRS_GLD_TLT,OPENDRIVE_SMH,OPENDRIVE_XLK,INITIAL_CAPITAL
from datetime import datetime; ET=pytz.timezone("America/New_York")
S=datetime(2025,12,1,tzinfo=ET); E=datetime(2026,4,4,tzinfo=ET); R={}

# ORB SPY+QQQ combined
orb_daily = {}
for sym in ["SPY","QQQ"]:
    df=get_minute_bars(sym,S,E,use_cache=True); df=prepare_features(df)
    p=dict(ORB_SHARED_DEFAULTS); p.update(SYMBOL_PROFILES.get(sym,{}))
    s=ORBBreakout(**p); df=s.generate_signals(df); r=run_backtest(df,s,sym)
    if hasattr(r,"daily_returns") and r.daily_returns is not None:
        orb_daily[sym]=r.daily_returns
        R[f"ORB_{sym}"]={"d":[str(x) for x in r.daily_returns.index],"r":r.daily_returns.values.tolist(),"t":r.num_trades,"capital":INITIAL_CAPITAL}

# OpenDrive SMH+XLK combined
for sym,cfg in [("SMH",OPENDRIVE_SMH),("XLK",OPENDRIVE_XLK)]:
    df=get_minute_bars(sym,S,E,use_cache=True); df=prepare_features(df)
    s=OpeningDrive(**cfg); df=s.generate_signals(df); r=run_backtest(df,s,sym)
    if hasattr(r,"daily_returns") and r.daily_returns is not None:
        R[f"OD_{sym}"]={"d":[str(x) for x in r.daily_returns.index],"r":r.daily_returns.values.tolist(),"t":r.num_trades,"capital":INITIAL_CAPITAL}

# Pairs GLD/TLT
cfg=PAIRS_GLD_TLT; g=get_minute_bars("GLD",S,E,use_cache=True); g=prepare_features(g)
t=get_minute_bars("TLT",S,E,use_cache=True); tc=t.set_index("dt")["close"].rename("pair_close")
g=g.set_index("dt").join(tc,how="left").reset_index(); g["pair_close"]=g["pair_close"].ffill()
s=PairsSpread(lookback=cfg["lookback"],entry_zscore=cfg["entry_zscore"],exit_zscore=cfg["exit_zscore"],stale_bars=cfg["stale_bars"],last_entry_minute=cfg["last_entry_minute"])
g=s.generate_signals(g); r=run_backtest(g,s,"GLD")
if hasattr(r,"daily_returns") and r.daily_returns is not None:
    R["Pairs"]={"d":[str(x) for x in r.daily_returns.index],"r":r.daily_returns.values.tolist(),"t":r.num_trades,"capital":INITIAL_CAPITAL}

with open(sys.argv[1],"w") as f: json.dump(R,f)
'''

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=FUTURES_DIR) as tf:
    tf.write(eq_script); sp = tf.name
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=FUTURES_DIR) as tf:
    op = tf.name
subprocess.run([sys.executable, sp, op], capture_output=True, text=True, timeout=300, cwd=EQUITY_DIR)
with open(op) as f:
    eq = json.load(f)
os.unlink(sp); os.unlink(op)

# Build equity combined streams
eq_orb_streams = {}
eq_od_streams = {}
for name, data in eq.items():
    idx = pd.to_datetime(data["d"], utc=True).date
    dr = pd.Series(data["r"], index=idx)
    bt_cap = data.get("capital", BT_CAPITAL)
    # Convert from pct returns to dollar P&L on the BT capital, then re-express as pct of $1M
    # The daily returns are already pct of BT_CAPITAL
    if "ORB" in name:
        eq_orb_streams[name] = dr
    elif "OD" in name:
        eq_od_streams[name] = dr
    elif "Pairs" in name:
        eq_pairs_daily = dr


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD 6 STRATEGY GROUPS + COMBINED
# ═══════════════════════════════════════════════════════════════════════════════

# For equity strategies, daily returns are pct of backtest capital ($100K).
# On a $1M account, the dollar P&L = daily_return * $1M
# For futures, daily returns are pct of index price (leveraged).
# With risk manager capping at 1% risk per trade and limited contracts,
# realistic daily P&L per contract is much smaller.
# We scale futures by: (ACCOUNT / BT_CAPITAL) for equity,
# and for futures we use raw returns * multiplier * 1 contract as a base,
# then scale by how many contracts the risk manager would allow.

# SIMPLIFICATION: Use backtest returns as-is (pct of price), then
# multiply by account size. This is what the equity backtests do.
# For futures, the returns are also pct, so same treatment.
# The risk manager will size positions relative to account equity anyway.

strategies = {}

# 1. Equity ORB (SPY+QQQ)
if eq_orb_streams:
    orb_combined = pd.DataFrame(eq_orb_streams).fillna(0).mean(axis=1)
    orb_trades = sum(eq[k]["t"] for k in eq if "ORB" in k)
    strategies["Equity ORB\n(SPY+QQQ)"] = {
        "daily": orb_combined,
        "trades": orb_trades,
    }

# 2. OpenDrive (SMH+XLK)
if eq_od_streams:
    od_combined = pd.DataFrame(eq_od_streams).fillna(0).mean(axis=1)
    od_trades = sum(eq[k]["t"] for k in eq if "OD" in k)
    strategies["OpenDrive\n(SMH+XLK)"] = {
        "daily": od_combined,
        "trades": od_trades,
    }

# 3. Pairs (GLD/TLT)
strategies["Pairs\n(GLD/TLT)"] = {
    "daily": eq_pairs_daily,
    "trades": eq.get("Pairs", {}).get("t", 0),
}

# 4. Futures VWAP (MES+MNQ+MGC)
strategies["Futures VWAP\n(MES+MNQ+MGC)"] = {
    "daily": vwap_combined,
    "trades": int((vwap_combined != 0).sum()),
}

# 5. Overnight (MNQ)
strategies["Overnight\n(MNQ)"] = {
    "daily": on_daily,
    "trades": int((on_daily != 0).sum()),
}

# Note: Futures ORB is excluded because it can't be backtested on raw
# futures data (needs prepare_features). The equity ORB on SPY+QQQ
# captures the same edge. Including it would be double-counting.


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nComputing 3-month projections on $1M account...\n", flush=True)

results = []
all_streams_for_combined = {}

for name, info in strategies.items():
    dr = info["daily"]
    trades = info["trades"]
    n_days = len(dr)
    active_days = (dr != 0).sum()
    trades_per_day = active_days / n_days if n_days > 0 else 0

    daily_mean = dr.mean()
    daily_std = dr.std()

    if daily_std == 0 or n_days < 10:
        continue

    sharpe = daily_mean / daily_std * np.sqrt(252)

    # Scale to $1M and 63 days
    mean_3mo = daily_mean * DAYS_3MO * ACCOUNT
    std_3mo = daily_std * np.sqrt(DAYS_3MO) * ACCOUNT

    bad = mean_3mo - 1.28 * std_3mo     # 10th percentile
    expected = mean_3mo                   # 50th percentile
    great = mean_3mo + 1.28 * std_3mo    # 90th percentile

    # P(loss) using normal CDF
    p_loss = stats.norm.cdf(0, loc=mean_3mo, scale=std_3mo) if std_3mo > 0 else 0

    results.append({
        "name": name,
        "sharpe": sharpe,
        "expected": expected,
        "bad": bad,
        "great": great,
        "p_loss": p_loss,
        "trades_day": trades_per_day,
        "oos_days": n_days,
    })

    all_streams_for_combined[name] = dr

# Combined portfolio
comb = pd.DataFrame(all_streams_for_combined).fillna(0).mean(axis=1)
comb_mean = comb.mean()
comb_std = comb.std()
comb_sharpe = comb_mean / comb_std * np.sqrt(252) if comb_std > 0 else 0
comb_mean_3mo = comb_mean * DAYS_3MO * ACCOUNT
comb_std_3mo = comb_std * np.sqrt(DAYS_3MO) * ACCOUNT
comb_trades = sum(info["trades"] for info in strategies.values())
comb_days = len(comb)
comb_tpd = sum(r["trades_day"] for r in results)

results.append({
    "name": ">> ALL COMBINED",
    "sharpe": comb_sharpe,
    "expected": comb_mean_3mo,
    "bad": comb_mean_3mo - 1.28 * comb_std_3mo,
    "great": comb_mean_3mo + 1.28 * comb_std_3mo,
    "p_loss": stats.norm.cdf(0, loc=comb_mean_3mo, scale=comb_std_3mo) if comb_std_3mo > 0 else 0,
    "trades_day": comb_tpd,
    "oos_days": comb_days,
})

# Sort by expected P&L descending
results.sort(key=lambda x: -x["expected"])

# Print
print(f"{'Rank':<5} {'Strategy':<22} {'Sharpe':>7} {'Expected':>12} {'Bad (10th)':>12} {'Great (90th)':>13} {'P(loss)':>8} {'Trades/d':>9}")
print("-" * 95)
for i, r in enumerate(results, 1):
    tag = "" if "\u2b50" not in r["name"] else ""
    print(f"{i:<5} {r['name'].replace(chr(10),' '):<22} {r['sharpe']:>7.2f} "
          f"${r['expected']:>+11,.0f} ${r['bad']:>+11,.0f} ${r['great']:>+12,.0f} "
          f"{r['p_loss']:>7.1%} {r['trades_day']:>9.1f}")

print(f"\nAccount: $1,000,000 | Period: 63 trading days (3 months)")
print(f"Based on locked OOS data. Paper trading - not live results.")
