#!/usr/bin/env python3
"""Research: overnight/pre-market edge analysis.

Questions:
1. Is there a systematic overnight return (close-to-open) in our instruments?
2. Does the overnight gap predict intraday direction?
3. Could we capture edge by holding positions overnight?
4. Is the pre-market session (4AM-9:30AM) tradeable on Alpaca?

This is exploratory research — no strategy implementation unless edge is found.
"""

import sys, os, io, warnings
from datetime import datetime
import pytz, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features

ET = pytz.timezone("America/New_York")
START = datetime(2025, 1, 2, tzinfo=ET)
END = datetime(2026, 4, 14, tzinfo=ET)

print("Loading data...", flush=True)
data = {}
for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    df = get_minute_bars(sym, START, END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"  {sym}: {len(df)} bars", flush=True)


print(f"\n{'='*90}", flush=True)
print("OVERNIGHT EDGE ANALYSIS", flush=True)
print(f"{'='*90}", flush=True)

# ── 1. Overnight returns (close-to-open) ──
print(f"\n  1. OVERNIGHT RETURNS (prev close -> today open)", flush=True)
print(f"  {'Symbol':<6} {'Mean':>8} {'Std':>8} {'Sharpe':>7} {'%Pos':>6} {'Count':>6}", flush=True)
print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*6}", flush=True)

for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    df = data[sym]
    daily = df.groupby("date").agg(
        day_open=("open", "first"),
        day_close=("close", "last"),
    )
    overnight_ret = (daily["day_open"].shift(-1) / daily["day_close"] - 1).dropna() * 100
    # Actually: this morning's open / yesterday's close
    overnight_ret = (daily["day_open"] / daily["day_close"].shift(1) - 1).dropna() * 100

    if len(overnight_ret) < 20:
        continue

    mean = overnight_ret.mean()
    std = overnight_ret.std()
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    pct_pos = (overnight_ret > 0).mean() * 100

    print(f"  {sym:<6} {mean:>+7.3f}% {std:>7.3f}% {sharpe:>7.2f} {pct_pos:>5.1f}% {len(overnight_ret):>6}", flush=True)


# ── 2. Gap predicting intraday direction ──
print(f"\n  2. GAP PREDICTING INTRADAY DIRECTION", flush=True)
print(f"  (Does gap direction predict same-day close direction?)", flush=True)
print(f"  {'Symbol':<6} {'GapUp->DayUp':>12} {'GapDn->DayDn':>12} {'Overall':>8}", flush=True)
print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*8}", flush=True)

for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    df = data[sym]
    daily = df.groupby("date").agg(
        day_open=("open", "first"),
        day_close=("close", "last"),
    )
    prev_close = daily["day_close"].shift(1)
    gap = (daily["day_open"] / prev_close - 1).dropna()
    intraday = (daily["day_close"] / daily["day_open"] - 1)

    # Align
    aligned = pd.DataFrame({"gap": gap, "intraday": intraday}).dropna()
    if len(aligned) < 20:
        continue

    gap_up = aligned[aligned["gap"] > 0]
    gap_dn = aligned[aligned["gap"] < 0]

    # Gap up -> intraday positive?
    gu_correct = (gap_up["intraday"] > 0).mean() * 100 if len(gap_up) > 0 else 0
    # Gap down -> intraday negative?
    gd_correct = (gap_dn["intraday"] < 0).mean() * 100 if len(gap_dn) > 0 else 0
    # Overall: gap direction = intraday direction?
    overall = ((aligned["gap"] > 0) == (aligned["intraday"] > 0)).mean() * 100

    print(f"  {sym:<6} {gu_correct:>11.1f}% {gd_correct:>11.1f}% {overall:>7.1f}%", flush=True)


# ── 3. Gap reversal tendency ──
print(f"\n  3. GAP REVERSAL TENDENCY", flush=True)
print(f"  (What % of gaps get filled during the day?)", flush=True)
print(f"  {'Symbol':<6} {'GapUp Fill%':>11} {'GapDn Fill%':>11} {'Avg Gap%':>8}", flush=True)
print(f"  {'-'*5} {'-'*11} {'-'*11} {'-'*8}", flush=True)

for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    df = data[sym]
    daily = df.groupby("date").agg(
        day_open=("open", "first"),
        day_close=("close", "last"),
        day_low=("low", "min"),
        day_high=("high", "max"),
    )
    prev_close = daily["day_close"].shift(1)
    gap_pct = ((daily["day_open"] / prev_close - 1) * 100)
    aligned = pd.DataFrame({
        "gap": gap_pct,
        "day_low": daily["day_low"],
        "day_high": daily["day_high"],
        "prev_close": prev_close,
    }).dropna()

    gap_up = aligned[aligned["gap"] > 0.1]  # Gap up > 0.1%
    gap_dn = aligned[aligned["gap"] < -0.1]

    # Gap up filled = day_low reached prev_close
    gu_filled = (gap_up["day_low"] <= gap_up["prev_close"]).mean() * 100 if len(gap_up) > 0 else 0
    # Gap down filled = day_high reached prev_close
    gd_filled = (gap_dn["day_high"] >= gap_dn["prev_close"]).mean() * 100 if len(gap_dn) > 0 else 0
    avg_gap = aligned["gap"].abs().mean()

    print(f"  {sym:<6} {gu_filled:>10.1f}% {gd_filled:>10.1f}% {avg_gap:>7.2f}%", flush=True)


# ── 4. Multi-timeframe: daily trend vs intraday ──
print(f"\n  4. MULTI-TIMEFRAME: DAILY TREND vs INTRADAY RETURNS", flush=True)
print(f"  (Does 5-day trend direction predict next-day intraday return?)", flush=True)
print(f"  {'Symbol':<6} {'TrendUp->DayUp':>14} {'TrendDn->DayDn':>14} {'EdgeBps':>8}", flush=True)
print(f"  {'-'*5} {'-'*14} {'-'*14} {'-'*8}", flush=True)

for sym in ["SPY", "QQQ", "GLD", "TLT", "SMH", "XLK"]:
    df = data[sym]
    daily = df.groupby("date").agg(
        day_open=("open", "first"),
        day_close=("close", "last"),
    )
    # 5-day return as trend indicator
    trend_5d = daily["day_close"].pct_change(5)
    # Next day intraday return
    intraday_ret = (daily["day_close"] / daily["day_open"] - 1)

    aligned = pd.DataFrame({
        "trend": trend_5d.shift(1),  # Use YESTERDAY's 5d trend
        "intraday": intraday_ret,
    }).dropna()

    trend_up = aligned[aligned["trend"] > 0]
    trend_dn = aligned[aligned["trend"] < 0]

    tu_correct = (trend_up["intraday"] > 0).mean() * 100 if len(trend_up) > 0 else 0
    td_correct = (trend_dn["intraday"] < 0).mean() * 100 if len(trend_dn) > 0 else 0

    # Edge in bps: avg intraday return when trading with trend
    with_trend = []
    for _, row in aligned.iterrows():
        if row["trend"] > 0:
            with_trend.append(row["intraday"])
        else:
            with_trend.append(-row["intraday"])  # Short when trend is down
    edge_bps = np.mean(with_trend) * 10000 if with_trend else 0

    print(f"  {sym:<6} {tu_correct:>13.1f}% {td_correct:>13.1f}% {edge_bps:>+7.1f}", flush=True)

# ── Summary ──
print(f"\n{'='*90}", flush=True)
print("RESEARCH CONCLUSIONS", flush=True)
print(f"{'='*90}", flush=True)
print(f"""
  1. OVERNIGHT RETURNS: Check Sharpe column above. If any instrument shows
     consistent overnight Sharpe > 1.0, there may be an overnight hold edge.
     However, overnight positions have gap risk and can't be stopped intraday.

  2. GAP PREDICTION: If gap direction predicts intraday direction > 55% of
     the time, our Opening Drive strategy is already capturing this edge.
     If gaps REVERSE > 55%, our Pairs strategy may benefit from gap-fade.

  3. GAP FILL: High fill rates (>60%) support mean-reversion strategies.
     Low fill rates support gap-continuation strategies.

  4. MULTI-TIMEFRAME: If 5-day trend predicts next-day direction > 52%,
     there's a potential filter: only take ORB/OpenDrive trades in the
     direction of the higher-timeframe trend. This would reduce trades
     but potentially improve win rate.

  ACTION ITEMS:
  - If overnight Sharpe > 1.0: research overnight hold strategy
  - If gap prediction > 55%: our strategies already exploit this
  - If multi-TF edge > 3 bps: test as trade direction filter
  - None of these should be implemented without locked OOS confirmation
""", flush=True)
