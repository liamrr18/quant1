#!/usr/bin/env python3
"""4-scenario 3-month projection comparison."""

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

FUTURES_DIR = "/root/futures_trader"
EQUITY_DIR = "/root/flamboyant-lewin"
ET = pytz.timezone("America/New_York")

sys.path.insert(0, FUTURES_DIR)

from trading.strategies.orb import ORBBreakout as FORB
from trading.strategies.vwap_reversion import VWAPReversion
from trading.strategies.overnight_reversion import OvernightReversion
from trading.config import (
    ORB_SHARED_DEFAULTS as F_ORB_DEFAULTS, SYMBOL_PROFILES as F_PROFILES,
    VWAP_REVERSION_DEFAULTS, OVERNIGHT_REVERSION_DEFAULTS,
)
from trading.data.contracts import CONTRACTS

DAYS_3MO = 63
MONTH_DAYS = 21
N_SIMS = 10000

# ─── Contract specs ──────────────────────────────────────────────────────
MARGIN = {"MES": 1500, "MNQ": 1800, "MGC": 1000}
MULT = {"MES": 5, "MNQ": 2, "MGC": 10}
COMM_RT = 0.62  # per contract round trip

# ─── Load futures OOS data ───────────────────────────────────────────────
cache = f"{FUTURES_DIR}/data/cache"


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

    # VWAP stream
    vwap_strat = VWAPReversion(**VWAP_REVERSION_DEFAULTS)
    futures_streams[f"VWAP_{sym}"] = run_strat(oos_df, vwap_strat)

    # ORB stream (skip MGC - no ORB)
    if sym != "MGC":
        orb_params = dict(F_ORB_DEFAULTS)
        orb_params.update(F_PROFILES.get(sym, {}))
        orb_strat = FORB(**orb_params)
        futures_streams[f"ORB_{sym}"] = run_strat(oos_df, orb_strat)

# Overnight MNQ
on_path = os.path.join(cache, "MNQ_futures_full_1min.csv")
if os.path.exists(on_path):
    on_df = pd.read_csv(on_path)
    on_df["dt"] = pd.to_datetime(on_df["dt"], utc=True).dt.tz_convert(ET)
    on_df["date"] = on_df["dt"].dt.date
    on_oos = oos(on_df)
    on_strat = OvernightReversion(**OVERNIGHT_REVERSION_DEFAULTS)
    futures_streams["ON_MNQ"] = run_strat(on_oos, on_strat)


# ─── Load equity OOS via subprocess (separate venv to avoid import conflict) ──
print("Loading equity OOS data...", flush=True)

eq_script = '''
import sys, json, pandas as pd, pytz, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "''' + EQUITY_DIR + '''")
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

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/root") as tf:
    tf.write(eq_script); sp = tf.name
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/root") as tf:
    op = tf.name

r = subprocess.run([sys.executable, sp, op], capture_output=True, text=True, timeout=300, cwd=EQUITY_DIR)
with open(op) as f:
    eq = json.load(f)
os.unlink(sp); os.unlink(op)

equity_streams = {}
for name, data in eq.items():
    idx = pd.to_datetime(data["d"], utc=True).date
    equity_streams[name] = pd.Series(data["r"], index=idx)

# ─── Helper: project from daily returns array ────────────────────────────

def project(daily_dollar_pnl, starting_balance, n_days=DAYS_3MO, n_sims=N_SIMS, seed=42):
    """daily_dollar_pnl: 1D array of per-day dollar P&L (already account-size adjusted)."""
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
    # Approximate max drawdown from median path
    median_cumul = np.median(np.array([np.cumsum(np.random.choice(daily_dollar_pnl, size=n_days, replace=True))
                                        for _ in range(1000)]), axis=0)
    dd = np.min(median_cumul - np.maximum.accumulate(median_cumul))
    result["max_dd"] = float(dd)
    result["worst_day"] = float(np.percentile(daily_dollar_pnl, 1))
    result["bad"] = starting_balance + float(np.percentile(sims, 10))
    result["avg"] = starting_balance + float(np.percentile(sims, 50))
    result["great"] = starting_balance + float(np.percentile(sims, 90))
    return result


# ─── Helper: convert per-contract % daily return to $ P&L at given size ──
def scale_futures_stream_to_dollars(daily_ret_series, symbol, contracts):
    """Convert daily return (% of underlying) to dollar P&L per contract * count.
    daily_ret = (end_price - start_price) / start_price
    dollar_pnl = daily_ret * avg_price * multiplier * contracts
    Approximate avg_price from contract specs.
    """
    # Approximate typical price for each contract
    typical_price = {"MES": 7000, "MNQ": 26000, "MGC": 4800}[symbol]
    dollar_per_contract_per_day = daily_ret_series.values * typical_price * MULT[symbol]
    return dollar_per_contract_per_day * contracts


def equity_stream_to_dollars(daily_ret, account_capital_share):
    """Equity strategy returns are already % of backtest capital ($100K).
    Scale to dollar P&L at given capital allocation.
    """
    return daily_ret.values * account_capital_share


# ─── Scenario definitions ────────────────────────────────────────────────

def scenario_1_1m_all():
    """1M account, all 7 strategies."""
    # Futures: size by account / margin ratio. On $1M with 2x margin:
    # Can hold lots of contracts. Risk manager targets 1% per trade = $10K risk.
    # VWAP: 0.3% stop × price × mult. Budget $10K / ($7000*0.003*5) = 95 MES contracts (capped at 30).
    account = 1_000_000
    streams_dollars = []

    # Sizing: use risk manager logic (simplified): 1% risk per trade
    # But cap at MAX_CONTRACTS. For $1M:
    # MES: min(95, 30) = 30. MNQ: min($10K/$212, 20) = 20. MGC: min($10K/$144, 5) = 5.
    # Still, strategy only holds 1 position at a time per stream so total contracts at once is manageable.

    sizes = {"MES_ORB": 30, "MNQ_ORB": 20, "VWAP_MES": 30, "VWAP_MNQ": 20, "VWAP_MGC": 5, "ON_MNQ": 1}
    trade_count = 0
    for key, size in sizes.items():
        sym = key.split("_")[-1]
        stream_name = {"MES_ORB": "ORB_MES", "MNQ_ORB": "ORB_MNQ",
                       "VWAP_MES": "VWAP_MES", "VWAP_MNQ": "VWAP_MNQ",
                       "VWAP_MGC": "VWAP_MGC", "ON_MNQ": "ON_MNQ"}[key]
        stream = futures_streams.get(stream_name)
        if stream is None:
            continue
        dollars = scale_futures_stream_to_dollars(stream, sym, size)
        # Subtract commissions per trade (~0.5 trades/day per stream)
        active_days = (stream != 0).sum()
        total_days = len(stream)
        trades_per_day = active_days / total_days if total_days > 0 else 0
        daily_comm = trades_per_day * COMM_RT * size
        dollars = dollars - daily_comm
        streams_dollars.append(dollars)
        trade_count += trades_per_day

    # Equity streams — % of $100K backtest cap, scale to 20% allocation of $1M = $200K each
    per_strat_capital = 200_000
    for name in ["ORB_SPY", "ORB_QQQ", "OD_SMH", "OD_XLK", "Pairs"]:
        s = equity_streams.get(name)
        if s is None:
            continue
        dollars = equity_stream_to_dollars(s, per_strat_capital)
        streams_dollars.append(dollars)
        active = (s != 0).sum()
        trade_count += active / len(s) if len(s) > 0 else 0

    # Align lengths
    min_len = min(len(d) for d in streams_dollars)
    combined = np.sum([d[:min_len] for d in streams_dollars], axis=0)
    commissions = 0
    # Approximate total futures commission: trades_per_day * 63 * ~80 contracts * $0.62
    commissions = trade_count * 63 * 50 * COMM_RT / 10  # rough aggregate
    return {
        "name": "1M all 7 strategies",
        "starting": account,
        "strategies": "ORB(MES+MNQ), VWAP(MES+MNQ+MGC), ON(MNQ), E-ORB(SPY+QQQ), OD(SMH+XLK), Pairs(GLD/TLT)",
        "trades_per_day": trade_count,
        "combined_daily": combined,
        "commissions_approx": trade_count * 63 * 80 * COMM_RT,  # very rough
    }


def scenario_2_1m_futures_only():
    account = 1_000_000
    sizes = {"ORB_MES": 30, "ORB_MNQ": 20, "VWAP_MES": 30, "VWAP_MNQ": 20, "VWAP_MGC": 5, "ON_MNQ": 1}
    streams_dollars = []
    trade_count = 0
    for key, size in sizes.items():
        sym = key.split("_")[-1]
        stream = futures_streams.get(key)
        if stream is None:
            continue
        dollars = scale_futures_stream_to_dollars(stream, sym, size)
        active_days = (stream != 0).sum()
        total_days = len(stream)
        tpd = active_days / total_days if total_days > 0 else 0
        trade_count += tpd
        daily_comm = tpd * COMM_RT * size
        dollars = dollars - daily_comm
        streams_dollars.append(dollars)

    min_len = min(len(d) for d in streams_dollars)
    combined = np.sum([d[:min_len] for d in streams_dollars], axis=0)
    return {
        "name": "1M futures only",
        "starting": account,
        "strategies": "ORB(MES+MNQ), VWAP(MES+MNQ+MGC), ON(MNQ)",
        "trades_per_day": trade_count,
        "combined_daily": combined,
        "commissions_approx": trade_count * 63 * 106 * COMM_RT,  # sum of sizes
    }


def scenario_3_5k_futures():
    """$5k futures. Need to check margin. 1 MES+1 MNQ+1 MGC = $4300 margin.
    With 2x safety = $8600 > $5k. So can't hold all 3 at once.
    With $5k/2 = $2500 buffer needed, so $5k - $2500 = $2500 available for margin.
    1 MES ($1500) fits alone. 1 MNQ ($1800) fits alone. 1 MGC ($1000) fits alone.
    But strategies generally take 1 position at a time per stream.

    Risk limit: 1% = $50 per trade. VWAP stop ~$106-$159 -> BLOCKED for every symbol.
    ORB ~$141-$212 -> BLOCKED.
    Only Overnight (fixed 1 contract, no risk-based sizing) works.
    """
    account = 5000
    # Nothing but ON_MNQ passes the risk sizer
    streams_dollars = []
    tpd = 0
    stream = futures_streams.get("ON_MNQ")
    if stream is not None:
        dollars = scale_futures_stream_to_dollars(stream, "MNQ", 1)
        active_days = (stream != 0).sum()
        total_days = len(stream)
        tpd = active_days / total_days if total_days > 0 else 0
        daily_comm = tpd * COMM_RT
        dollars = dollars - daily_comm
        streams_dollars.append(dollars)

    if streams_dollars:
        combined = streams_dollars[0]
    else:
        combined = np.zeros(100)
    return {
        "name": "5k futures-attempt",
        "starting": account,
        "strategies": "ON(MNQ) only — ORB and VWAP blocked by 1% risk limit",
        "trades_per_day": tpd,
        "combined_daily": combined,
        "commissions_approx": tpd * 63 * 1 * COMM_RT,
        "blocked_note": "ORB stops ~$141/ct, VWAP stops ~$106-$159/ct. Risk budget = $50. Only ON (no sizer) fits.",
    }


def scenario_4_5k_best():
    """$5k best single strategy. ON_MNQ has Sharpe 1.65, trades just 1 contract,
    proven positive. That's the best single viable futures strategy."""
    account = 5000
    stream = futures_streams.get("ON_MNQ")
    dollars = scale_futures_stream_to_dollars(stream, "MNQ", 1)
    active_days = (stream != 0).sum()
    total_days = len(stream)
    tpd = active_days / total_days if total_days > 0 else 0
    daily_comm = tpd * COMM_RT
    dollars = dollars - daily_comm
    return {
        "name": "5k best single (ON_MNQ)",
        "starting": account,
        "strategies": "ON(MNQ) only, 1 contract",
        "trades_per_day": tpd,
        "combined_daily": dollars,
        "commissions_approx": tpd * 63 * 1 * COMM_RT,
    }


# ─── Run all 4 ─────────────────────────────────────────────────────────

scenarios = [scenario_1_1m_all(), scenario_2_1m_futures_only(),
             scenario_3_5k_futures(), scenario_4_5k_best()]

results = []
for s in scenarios:
    proj = project(s["combined_daily"], s["starting"])
    s["proj"] = proj
    results.append(s)

# ─── Print per-scenario detail ─────────────────────────────────────────
for s in results:
    p = s["proj"]
    print("=" * 80)
    print(f"SCENARIO: {s['name'].upper()}")
    print(f"Starting: ${s['starting']:,} | Strategies: {s['strategies']}")
    print(f"Trades/day: {s['trades_per_day']:.1f}")
    print(f"Daily mean P&L: ${p['daily_mean']:+,.0f} | Std: ${p['daily_std']:,.0f}")
    if s.get("blocked_note"):
        print(f"NOTE: {s['blocked_note']}")
    print()
    print(f"{'Month':<8} {'Bad (10%)':>15} {'Expected (50%)':>18} {'Great (90%)':>15}")
    for m in [1, 2, 3]:
        mp = p[f"m{m}"]
        print(f"M{m:<7} ${mp['bad']:>14,.0f} ${mp['avg']:>17,.0f} ${mp['great']:>14,.0f}")
    print()
    print(f"P(loss over 3mo): {p['p_loss']:.1%}")
    print(f"P(account halved): {p['p_halved']:.1%}")
    print(f"Est. max drawdown: ${p['max_dd']:,.0f} ({p['max_dd']/s['starting']*100:.1f}%)")
    print(f"Worst single day (1%ile): ${p['worst_day']:,.0f}")
    print(f"Approx commissions over 3mo: ${s['commissions_approx']:,.0f}")
    if p['avg'] - s['starting'] != 0:
        print(f"Commissions as % of gross P&L: {abs(s['commissions_approx'] / (p['avg'] - s['starting'])) * 100:.1f}%")
    print(f"3-MONTH RESULT: Bad ${p['bad']:,.0f} | Expected ${p['avg']:,.0f} | Great ${p['great']:,.0f}")
    print()

# ─── Comparison table ──────────────────────────────────────────────────
print("=" * 120)
print("COMPARISON TABLE — 3 MONTH OUTCOMES")
print("=" * 120)
print(f"{'Scenario':<30} {'Start':>10} {'Strats':>7} {'Tr/d':>6} {'Expected P&L':>14} {'Bad':>12} {'Great':>14} {'P(Loss)':>8} {'MaxDD':>10} {'Comms':>10}")
print("-" * 120)
for s in results:
    p = s["proj"]
    expected_pnl = p["avg"] - s["starting"]
    bad_pnl = p["bad"] - s["starting"]
    great_pnl = p["great"] - s["starting"]
    n_strats = len(s["strategies"].split(","))
    print(f"{s['name']:<30} ${s['starting']:>9,} {n_strats:>7} {s['trades_per_day']:>6.1f} "
          f"${expected_pnl:>+13,.0f} ${bad_pnl:>+11,.0f} ${great_pnl:>+13,.0f} "
          f"{p['p_loss']:>7.1%} ${abs(p['max_dd']):>9,.0f} ${s['commissions_approx']:>9,.0f}")

print()
print("Done.")
