#!/usr/bin/env python3
"""Compute combined portfolio stats from locked OOS and send to Discord."""

import json
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytz
import requests

warnings.filterwarnings("ignore")

FUTURES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUITY_DIR = r"C:\Users\liamr\Desktop\spy-trader\.claude\worktrees\flamboyant-lewin"
ET = pytz.timezone("America/New_York")
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_PORTFOLIO", "")

sys.path.insert(0, FUTURES_DIR)

from trading.strategies.vwap_reversion import VWAPReversion
from trading.strategies.overnight_reversion import OvernightReversion
from trading.config import VWAP_REVERSION_DEFAULTS, OVERNIGHT_REVERSION_DEFAULTS
from trading.data.contracts import CONTRACTS

# ═══════════════════════════════════════════════════════════════════════════════
# FUTURES STREAMS
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading futures OOS...", flush=True)

cache = os.path.join(FUTURES_DIR, "data", "cache")
streams = {}

def load_cash(filename):
    df = pd.read_csv(os.path.join(cache, filename))
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)
    df["date"] = df["dt"].dt.date
    return df

def run_strategy(df, strat):
    sig = strat.generate_signals(df.copy())
    sig["pos"] = sig["signal"].shift(1).fillna(0)
    sig["br"] = sig["close"].pct_change().fillna(0)
    sig["sr"] = sig["pos"] * sig["br"]
    return sig.groupby("date")["sr"].sum()

def oos_split(df):
    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.8)
    return df[df["date"].isin(set(dates[split:]))].copy()

# VWAP MES
df = load_cash("MES_futures_cash_1min.csv")
oos = oos_split(df)
streams["VWAP MES"] = run_strategy(oos, VWAPReversion(**VWAP_REVERSION_DEFAULTS))
print(f"  VWAP MES: {len(streams['VWAP MES'])} days")

# VWAP MNQ
df = load_cash("MNQ_futures_cash_1min.csv")
oos = oos_split(df)
streams["VWAP MNQ"] = run_strategy(oos, VWAPReversion(**VWAP_REVERSION_DEFAULTS))
print(f"  VWAP MNQ: {len(streams['VWAP MNQ'])} days")

# VWAP MGC
df = load_cash("MGC_futures_cash_1min.csv")
oos = oos_split(df)
streams["VWAP MGC"] = run_strategy(oos, VWAPReversion(**VWAP_REVERSION_DEFAULTS))
print(f"  VWAP MGC: {len(streams['VWAP MGC'])} days")

# Overnight MNQ
on_df = pd.read_csv(os.path.join(cache, "MNQ_futures_full_1min.csv"))
on_df["dt"] = pd.to_datetime(on_df["dt"], utc=True).dt.tz_convert(ET)
on_df["date"] = on_df["dt"].dt.date
oos_on = oos_split(on_df)
streams["ON MNQ"] = run_strategy(oos_on, OvernightReversion(**OVERNIGHT_REVERSION_DEFAULTS))
print(f"  ON MNQ: {len(streams['ON MNQ'])} days")


# ═══════════════════════════════════════════════════════════════════════════════
# EQUITY STREAMS (subprocess to avoid import conflict)
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading equity OOS (subprocess)...", flush=True)

eq_script = '''
import sys, os, json, warnings
import numpy as np, pandas as pd, pytz
warnings.filterwarnings("ignore")
sys.path.insert(0, r"''' + EQUITY_DIR.replace("\\", "\\\\") + '''")

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
S = datetime(2025, 12, 1, tzinfo=ET)
E = datetime(2026, 4, 4, tzinfo=ET)
results = {}

for sym in ["SPY", "QQQ"]:
    df = get_minute_bars(sym, S, E, use_cache=True)
    df = prepare_features(df)
    p = dict(ORB_SHARED_DEFAULTS); p.update(SYMBOL_PROFILES.get(sym, {}))
    strat = ORBBreakout(**p)
    df = strat.generate_signals(df)
    r = run_backtest(df, strat, sym)
    if hasattr(r, "daily_returns") and r.daily_returns is not None:
        dr = r.daily_returns
        results[f"ORB {sym}"] = {"dates": [str(d) for d in dr.index], "returns": dr.values.tolist(), "trades": r.num_trades}

for sym, cfg in [("SMH", OPENDRIVE_SMH), ("XLK", OPENDRIVE_XLK)]:
    df = get_minute_bars(sym, S, E, use_cache=True)
    df = prepare_features(df)
    strat = OpeningDrive(**cfg)
    df = strat.generate_signals(df)
    r = run_backtest(df, strat, sym)
    if hasattr(r, "daily_returns") and r.daily_returns is not None:
        dr = r.daily_returns
        results[f"OD {sym}"] = {"dates": [str(d) for d in dr.index], "returns": dr.values.tolist(), "trades": r.num_trades}

cfg = PAIRS_GLD_TLT
gld = get_minute_bars("GLD", S, E, use_cache=True); gld = prepare_features(gld)
tlt = get_minute_bars("TLT", S, E, use_cache=True)
tc = tlt.set_index("dt")["close"].rename("pair_close")
gld = gld.set_index("dt").join(tc, how="left").reset_index()
gld["pair_close"] = gld["pair_close"].ffill()
strat = PairsSpread(lookback=cfg["lookback"], entry_zscore=cfg["entry_zscore"],
    exit_zscore=cfg["exit_zscore"], stale_bars=cfg["stale_bars"], last_entry_minute=cfg["last_entry_minute"])
gld = strat.generate_signals(gld)
r = run_backtest(gld, strat, "GLD")
if hasattr(r, "daily_returns") and r.daily_returns is not None:
    dr = r.daily_returns
    results["Pairs GLD/TLT"] = {"dates": [str(d) for d in dr.index], "returns": dr.values.tolist(), "trades": r.num_trades}

with open(sys.argv[1], "w") as f:
    json.dump(results, f)
'''

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=FUTURES_DIR) as tf:
    tf.write(eq_script)
    sp = tf.name
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=FUTURES_DIR) as tf:
    op = tf.name

try:
    r = subprocess.run([sys.executable, sp, op], capture_output=True, text=True, timeout=300, cwd=EQUITY_DIR)
    if r.returncode != 0:
        print(f"Equity error: {r.stderr[:300]}")
    with open(op) as f:
        eq_results = json.load(f)
finally:
    os.unlink(sp)
    if os.path.exists(op):
        os.unlink(op)

for name, data in eq_results.items():
    idx = pd.to_datetime(data["dates"], utc=True).date
    streams[name] = pd.Series(data["returns"], index=idx)
    print(f"  {name}: {data['trades']} trades")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE PORTFOLIO STATS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing portfolio stats...", flush=True)

all_daily = pd.DataFrame(streams).fillna(0)
port = all_daily.mean(axis=1)  # equal weight
capital = 100_000

# Basic metrics
daily_mean = port.mean()
daily_std = port.std()
sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0
ds = port[port < 0]
sortino = daily_mean / ds.std() * np.sqrt(252) if len(ds) > 1 and ds.std() > 0 else 0
cum = (1 + port).cumprod()
max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
ann_ret = daily_mean * 252
calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

# Alpha/beta vs SPY
try:
    spy_data = eq_results.get("ORB SPY", {})
    if spy_data:
        spy_idx = pd.to_datetime(spy_data["dates"], utc=True).date
        spy_daily = pd.Series(spy_data["returns"], index=spy_idx)
        merged = pd.DataFrame({"port": port, "spy": spy_daily}).dropna()
        if len(merged) > 10 and merged["spy"].var() > 0:
            beta = merged["spy"].cov(merged["port"]) / merged["spy"].var()
            alpha = (merged["port"].mean() - beta * merged["spy"].mean()) * 252
        else:
            alpha, beta = ann_ret, 0
    else:
        alpha, beta = ann_ret, 0
except Exception:
    alpha, beta = ann_ret, 0

# Trade stats
total_trades = sum(
    (all_daily[col].abs() > 0).sum() for col in all_daily.columns
)
trading_days = len(all_daily)
trades_per_day = total_trades / trading_days if trading_days > 0 else 0

active = port[port != 0]
wins = active[active > 0]
losses = active[active < 0]
win_rate = len(wins) / len(active) if len(active) > 0 else 0
profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0
avg_win = wins.mean() * capital if len(wins) > 0 else 0
avg_loss = losses.mean() * capital if len(losses) > 0 else 0
total_pnl = port.sum() * capital

# Per-strategy stats
strat_stats = []
for name in sorted(all_daily.columns):
    dr = all_daily[name]
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    pnl_pct = dr.sum()
    pnl_share = pnl_pct / port.sum() * 100 if port.sum() != 0 else 0
    strat_stats.append((name, sh, pnl_share))

# Correlation
corr = all_daily.corr()

# Monte Carlo
np.random.seed(42)
mc_returns = np.array([
    np.sum(np.random.choice(port.values, size=252, replace=True))
    for _ in range(10000)
])
mc_100k = mc_returns * capital
mc_1m = mc_returns * 1_000_000

p10_100k = np.percentile(mc_100k, 10)
p50_100k = np.percentile(mc_100k, 50)
p90_100k = np.percentile(mc_100k, 90)
p10_1m = np.percentile(mc_1m, 10)
p50_1m = np.percentile(mc_1m, 50)
p90_1m = np.percentile(mc_1m, 90)


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT & SEND TO DISCORD
# ═══════════════════════════════════════════════════════════════════════════════
print("Sending to Discord...", flush=True)

# Build correlation matrix string
abbr = {}
for n in corr.columns:
    a = n.replace("VWAP ", "V.").replace("ORB ", "O.").replace("Pairs ", "P.").replace("ON ", "ON.").replace("OD ", "D.")
    abbr[n] = a[:7]

corr_lines = []
header = "       " + " ".join(f"{abbr[n]:>7}" for n in corr.columns)
corr_lines.append(header)
for n in corr.columns:
    row = f"{abbr[n]:<7}" + " ".join(f"{corr.loc[n, n2]:>+7.2f}" for n2 in corr.columns)
    corr_lines.append(row)
corr_str = "\n".join(corr_lines)

# Strategy breakdown string
strat_lines = []
for name, sh, share in sorted(strat_stats, key=lambda x: -x[2]):
    strat_lines.append(f"{name:<15} Sharpe {sh:>5.2f} | {share:>5.1f}% of P&L")
strat_str = "\n".join(strat_lines)

msg = {
    "embeds": [
        {
            "title": "\U0001f4ca Combined Portfolio Statistics (Locked OOS)",
            "color": 0x2ECC71,
            "fields": [
                {"name": "Sharpe", "value": f"**{sharpe:.2f}**", "inline": True},
                {"name": "Sortino", "value": f"**{sortino:.2f}**", "inline": True},
                {"name": "Calmar", "value": f"**{calmar:.1f}**", "inline": True},
                {"name": "Ann. Return", "value": f"{ann_ret:+.1%}", "inline": True},
                {"name": "Alpha vs SPY", "value": f"{alpha:+.1%}", "inline": True},
                {"name": "Beta vs SPY", "value": f"{beta:.3f}", "inline": True},
                {"name": "Max Drawdown", "value": f"{max_dd:.2%} (${max_dd * capital:,.0f})", "inline": True},
                {"name": "OOS Days", "value": f"{trading_days}", "inline": True},
                {"name": "Streams", "value": f"{len(all_daily.columns)}", "inline": True},
            ],
        },
        {
            "title": "\U0001f4b9 Trade Statistics",
            "color": 0x3498DB,
            "fields": [
                {"name": "Total Trades", "value": f"{total_trades}", "inline": True},
                {"name": "Trades/Day", "value": f"{trades_per_day:.1f}", "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.1%}", "inline": True},
                {"name": "Profit Factor", "value": f"{profit_factor:.2f}", "inline": True},
                {"name": "Avg Win", "value": f"${avg_win:+,.0f}", "inline": True},
                {"name": "Avg Loss", "value": f"${avg_loss:+,.0f}", "inline": True},
                {"name": "Total OOS P&L", "value": f"**${total_pnl:+,.0f}** on $100K", "inline": False},
            ],
        },
        {
            "title": "\U0001f4b0 Year-End Projections",
            "color": 0x3498DB,
            "fields": [
                {"name": "$100K Account", "value": (
                    f"Bad (10th):  **${capital + p10_100k:>10,.0f}**\n"
                    f"Average:      **${capital + p50_100k:>10,.0f}**\n"
                    f"Great (90th): **${capital + p90_100k:>10,.0f}**"
                ), "inline": True},
                {"name": "$1M Account", "value": (
                    f"Bad (10th):  **${1_000_000 + p10_1m:>13,.0f}**\n"
                    f"Average:      **${1_000_000 + p50_1m:>13,.0f}**\n"
                    f"Great (90th): **${1_000_000 + p90_1m:>13,.0f}**"
                ), "inline": True},
            ],
        },
        {
            "title": "\U0001f3af Per-Strategy Contribution",
            "color": 0x9B59B6,
            "description": f"```\n{strat_str}\n```",
        },
        {
            "title": "\U0001f517 Correlation Matrix",
            "color": 0x9B59B6,
            "description": f"```\n{corr_str}\n```",
            "footer": {"text": "Based on locked OOS period. Paper trading \u2014 not live results."},
        },
    ]
}

r = requests.post(WEBHOOK, json=msg, timeout=15)
print(f"Discord: HTTP {r.status_code}")
if r.status_code != 204:
    print(r.text[:300])
else:
    print("Sent successfully.")
