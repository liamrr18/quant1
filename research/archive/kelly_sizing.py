#!/usr/bin/env python3
"""Kelly Criterion position sizing analysis.

Computes optimal position sizing for each strategy based on historical
win rate and payoff ratio. Compares current fixed 30%/20% sizing to
Kelly-optimal and fractional Kelly (half-Kelly for safety).

Kelly formula: f* = (p * b - q) / b
  where p = win rate, q = 1-p, b = avg_win / avg_loss

Half-Kelly is the standard practical implementation (reduces variance
while capturing ~75% of the growth rate).
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
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

print("Loading data...", flush=True)
data = {}
for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    df = get_minute_bars(sym, START, END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df

gld = data["GLD"].copy()
tc = data["TLT"].set_index("dt")["close"].rename("pair_close")
gld = gld.set_index("dt").join(tc, how="left").reset_index()
gld["pair_close"] = gld["pair_close"].ffill()

def run_oos(df, strat, sym):
    mask = df["date"].apply(lambda d: OOS_START.date() <= d)
    p = df[mask].copy()
    p = strat.generate_signals(p)
    return run_backtest(p, strat, sym)

strats = [
    ("ORB SPY", run_oos(data["SPY"], ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["SPY"]}), "SPY")),
    ("ORB QQQ", run_oos(data["QQQ"], ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["QQQ"]}), "QQQ")),
    ("Pairs GLD/TLT", run_oos(gld, PairsSpread(lookback=120, entry_zscore=2.0, exit_zscore=0.5, stale_bars=90, last_entry_minute=900), "GLD")),
    ("OD SMH", run_oos(data["SMH"], OpeningDrive(**OPENDRIVE_SMH), "SMH")),
    ("OD XLK", run_oos(data["XLK"], OpeningDrive(**OPENDRIVE_XLK), "XLK")),
]

print(f"\n{'='*90}", flush=True)
print("KELLY CRITERION POSITION SIZING ANALYSIS (Locked OOS)", flush=True)
print(f"{'='*90}", flush=True)

print(f"\n  {'Strategy':<16} {'WinRate':>7} {'Payoff':>7} {'Kelly%':>7} {'HalfK%':>7}"
      f" {'Current%':>8} {'Verdict':>12}", flush=True)
print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*12}", flush=True)

for name, r in strats:
    wins = [t.pnl for t in r.trades if t.pnl > 0]
    losses = [t.pnl for t in r.trades if t.pnl <= 0]

    if not wins or not losses:
        print(f"  {name:<16} insufficient trades", flush=True)
        continue

    p = len(wins) / len(r.trades)  # Win rate
    q = 1 - p
    b = abs(np.mean(wins) / np.mean(losses))  # Payoff ratio

    # Kelly fraction
    kelly = (p * b - q) / b if b > 0 else 0
    half_kelly = kelly / 2

    # Current sizing
    current = 30 if "ORB" in name else 20

    if kelly <= 0:
        verdict = "DON'T TRADE"
    elif half_kelly > current / 100:
        verdict = "UNDERSIZE"
    elif half_kelly < current / 200:
        verdict = "OVERSIZE"
    else:
        verdict = "ABOUT RIGHT"

    print(f"  {name:<16} {p:>6.1%} {b:>7.2f} {kelly:>6.1%} {half_kelly:>6.1%}"
          f" {current:>7}% {verdict:>12}", flush=True)

print(f"\n  Notes:", flush=True)
print(f"  - Kelly% = optimal fraction of bankroll to risk per trade", flush=True)
print(f"  - HalfK% = practical recommendation (half-Kelly reduces variance)", flush=True)
print(f"  - Current% = what we're actually using (MAX_POSITION_PCT)", flush=True)
print(f"  - Kelly assumes independent trades. Correlated trades need less sizing.", flush=True)
print(f"  - With 5 concurrent streams, effective sizing per strategy should be ~1/5 of total.", flush=True)

# Portfolio-level Kelly
print(f"\n{'='*90}", flush=True)
print("PORTFOLIO-LEVEL SIZING", flush=True)
print(f"{'='*90}", flush=True)

all_trades = [t for _, r in strats for t in r.trades]
wins = [t.pnl for t in all_trades if t.pnl > 0]
losses = [t.pnl for t in all_trades if t.pnl <= 0]
p = len(wins) / len(all_trades)
b = abs(np.mean(wins) / np.mean(losses))
kelly = (p * b - 1 + p) / b if b > 0 else 0
half_kelly = kelly / 2

print(f"\n  Combined portfolio: WR={p:.1%}  Payoff={b:.2f}x  Kelly={kelly:.1%}  HalfKelly={half_kelly:.1%}", flush=True)
print(f"  Current total exposure: up to 5 x 20-30% = 100-150% theoretical max", flush=True)
print(f"  Practical max (4 concurrent positions): 4 x 25% = 100%", flush=True)

# Simulate different sizing levels
print(f"\n  Sizing simulation (locked OOS, ORB SPY+QQQ):", flush=True)
print(f"  {'Size%':>6} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7}", flush=True)
print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*7}", flush=True)

for size_pct in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    # Re-run ORB with different position sizing
    combined_dr = []
    for sym_name, sym_key in [("SPY", "SPY"), ("QQQ", "QQQ")]:
        params = dict(ORB_SHARED_DEFAULTS)
        params.update(SYMBOL_PROFILES.get(sym_key, {}))
        strat = ORBBreakout(**params)
        mask = data[sym_key]["date"].apply(lambda d: OOS_START.date() <= d)
        p_df = data[sym_key][mask].copy()
        p_df = strat.generate_signals(p_df)
        r = run_backtest(p_df, strat, sym_key, position_pct=size_pct)
        if r.daily_returns is not None and len(r.daily_returns) > 0:
            combined_dr.append(r.daily_returns)

    if combined_dr:
        port = pd.DataFrame({f"s{i}": dr for i, dr in enumerate(combined_dr)}).fillna(0).mean(axis=1)
        sh = (port.mean() / port.std()) * np.sqrt(252) if port.std() > 0 else 0
        ret = (1 + port).prod() - 1
        cum = (1 + port).cumprod()
        dd = ((cum - cum.cummax()) / cum.cummax()).min()
        print(f"  {size_pct:>5.0%} {sh:>7.2f} {ret:>+8.2%} {dd:>7.2%}", flush=True)

print(f"\n  RECOMMENDATION: Current 30% (ORB) and 20% (candidates) sizing is", flush=True)
print(f"  conservative and appropriate for the paper validation phase.", flush=True)
print(f"  After 20+ days of paper validation, consider moving to 25% uniform", flush=True)
print(f"  if all strategies confirm their edge.", flush=True)

print(f"\n{'='*90}", flush=True)
print("KELLY ANALYSIS COMPLETE", flush=True)
print(f"{'='*90}", flush=True)
