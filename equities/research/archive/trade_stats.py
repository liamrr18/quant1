#!/usr/bin/env python3
import sys, os, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from datetime import datetime
import pytz, numpy as np
from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.backtest.engine import run_backtest
from trading.strategies.orb import ORBBreakout
from trading.strategies.pairs_spread import PairsSpread
from trading.strategies.opening_drive import OpeningDrive
from trading.config import *

ET = pytz.timezone("America/New_York")
S = datetime(2025,12,1,tzinfo=ET); E = datetime(2026,4,4,tzinfo=ET)
B = datetime(2025,1,2,tzinfo=ET)

print("Loading...", flush=True)
data = {}
for sym in ["SPY","QQQ","GLD","TLT","SMH","XLK"]:
    df = get_minute_bars(sym, B, E, use_cache=True)
    df = prepare_features(df)
    data[sym] = df

def oos(df, strat, sym):
    p = df[df["date"].apply(lambda d: S.date() <= d)].copy()
    p = strat.generate_signals(p)
    return run_backtest(p, strat, sym)

gld = data["GLD"].copy()
tc = data["TLT"].set_index("dt")["close"].rename("pair_close")
gld = gld.set_index("dt").join(tc, how="left").reset_index()
gld["pair_close"] = gld["pair_close"].ffill()

results = [
    ("ORB SPY", oos(data["SPY"], ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["SPY"]}), "SPY")),
    ("ORB QQQ", oos(data["QQQ"], ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["QQQ"]}), "QQQ")),
    ("Pairs GLD/TLT", oos(gld, PairsSpread(lookback=120, entry_zscore=2.0, exit_zscore=0.5, stale_bars=90, last_entry_minute=900), "GLD")),
    ("OD SMH", oos(data["SMH"], OpeningDrive(**OPENDRIVE_SMH), "SMH")),
    ("OD XLK", oos(data["XLK"], OpeningDrive(**OPENDRIVE_XLK), "XLK")),
]

print(flush=True)
print(f"{'Strategy':<16} {'AvgWin':>8} {'AvgLoss':>9} {'BigWin':>8} {'BigLoss':>9} {'WR':>6} {'W':>4} {'L':>4} {'Payoff':>7}", flush=True)
print(f"{'-'*15} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*6} {'-'*4} {'-'*4} {'-'*7}", flush=True)

all_t = []
for name, r in results:
    wins = [t.pnl for t in r.trades if t.pnl > 0]
    losses = [t.pnl for t in r.trades if t.pnl <= 0]
    aw = np.mean(wins) if wins else 0
    al = np.mean(losses) if losses else 0
    bw = max(wins) if wins else 0
    bl = min(losses) if losses else 0
    wr = len(wins)/len(r.trades)*100 if r.trades else 0
    po = abs(aw/al) if al != 0 else 0
    print(f"{name:<16} ${aw:>7.2f} ${al:>8.2f} ${bw:>7.2f} ${bl:>8.2f} {wr:>5.1f}% {len(wins):>4} {len(losses):>4} {po:>6.2f}x", flush=True)
    all_t.extend(r.trades)

wins = [t.pnl for t in all_t if t.pnl > 0]
losses = [t.pnl for t in all_t if t.pnl <= 0]
aw = np.mean(wins); al = np.mean(losses)
print(f"\n{'COMBINED':<16} ${aw:>7.2f} ${al:>8.2f} ${max(wins):>7.2f} ${min(losses):>8.2f} {len(wins)/len(all_t)*100:>5.1f}% {len(wins):>4} {len(losses):>4} {abs(aw/al):>6.2f}x", flush=True)
print(f"\nTotal: {len(all_t)} trades over ~84 OOS days ({len(all_t)/84:.1f}/day)", flush=True)
