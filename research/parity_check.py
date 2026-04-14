#!/usr/bin/env python3
"""Verify production strategy modules match research backtest results."""

import sys, os, io, warnings
from datetime import datetime
import pytz, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.backtest.engine import run_backtest
from trading.strategies.pairs_spread import PairsSpread
from trading.strategies.opening_drive import OpeningDrive
from trading.config import PAIRS_GLD_TLT, OPENDRIVE_SMH, OPENDRIVE_XLK

ET = pytz.timezone("America/New_York")
START = datetime(2025, 1, 2, tzinfo=ET)
END = datetime(2026, 4, 4, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

print("Loading data...", flush=True)
data = {}
for sym in ["GLD", "TLT", "SMH", "XLK"]:
    df = get_minute_bars(sym, START, END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"  {sym}: {len(df)} bars", flush=True)

def run_oos(df, strat, sym):
    mask = df["date"].apply(lambda d: OOS_START.date() <= d)
    p = df[mask].copy()
    p = strat.generate_signals(p)
    r = run_backtest(p, strat, sym)
    dr = r.daily_returns
    sh = (dr.mean()/dr.std())*np.sqrt(252) if len(dr) > 5 and dr.std() > 0 else 0
    return sh, r.total_return, r.num_trades, r.profit_factor

# ── Test 1: Pairs GLD/TLT ──
print("\n=== Pairs GLD/TLT ===", flush=True)
gld = data["GLD"].copy()
tlt_close = data["TLT"].set_index("dt")["close"].rename("pair_close")
gld = gld.set_index("dt").join(tlt_close, how="left").reset_index()
gld["pair_close"] = gld["pair_close"].ffill()

strat = PairsSpread(
    lookback=PAIRS_GLD_TLT["lookback"],
    entry_zscore=PAIRS_GLD_TLT["entry_zscore"],
    exit_zscore=PAIRS_GLD_TLT["exit_zscore"],
    stale_bars=PAIRS_GLD_TLT["stale_bars"],
    last_entry_minute=PAIRS_GLD_TLT["last_entry_minute"],
)
sh, ret, trades, pf = run_oos(gld, strat, "GLD")
print(f"  Production module: Sharpe={sh:.2f}  Ret={ret:+.2%}  T={trades}  PF={pf:.2f}", flush=True)
print(f"  Research target:   Sharpe=4.86     Ret=+5.31%  T=228    PF=1.66", flush=True)
print(f"  Match: {'YES' if abs(sh - 4.86) < 1.0 else 'CHECK'}", flush=True)

# ── Test 2: OpenDrive SMH ──
print("\n=== OpenDrive SMH ===", flush=True)
strat = OpeningDrive(**OPENDRIVE_SMH)
sh, ret, trades, pf = run_oos(data["SMH"], strat, "SMH")
print(f"  Production module: Sharpe={sh:.2f}  Ret={ret:+.2%}  T={trades}  PF={pf:.2f}", flush=True)
print(f"  Research target:   Sharpe=3.87     Ret=+5.83%  T=86     PF=1.91", flush=True)
print(f"  Match: {'YES' if abs(sh - 3.87) < 1.0 else 'CHECK'}", flush=True)

# ── Test 3: OpenDrive XLK ──
print("\n=== OpenDrive XLK ===", flush=True)
strat = OpeningDrive(**OPENDRIVE_XLK)
sh, ret, trades, pf = run_oos(data["XLK"], strat, "XLK")
print(f"  Production module: Sharpe={sh:.2f}  Ret={ret:+.2%}  T={trades}  PF={pf:.2f}", flush=True)
print(f"  Research target:   Sharpe=3.26     Ret=+2.55%  T=84     PF=1.63", flush=True)
print(f"  Match: {'YES' if abs(sh - 3.26) < 1.0 else 'CHECK'}", flush=True)

print("\nParity check complete.", flush=True)
