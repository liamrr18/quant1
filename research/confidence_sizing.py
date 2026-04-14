#!/usr/bin/env python3
"""Experiment 16: Confidence-weighted position sizing overlay.

For each strategy, score trades at entry using features available at that
moment. Bucket into LOW / MED / HIGH confidence. Test whether higher
confidence = better trade outcomes on locked OOS.

If predictive: size HIGH trades at 1.5x base, MED at 1.0x, LOW at 0.5x.
If not predictive: REJECT the overlay.

Confidence features (available at entry time, no lookahead):
  ORB: range_pct, ATR percentile, relative volume, gap_pct, time of entry
  OpenDrive: drive_pct magnitude, ATR percentile, gap alignment, rel volume
  Pairs: Z-score magnitude, spread volatility rank, ATR of primary

Methodology:
  1. Run backtest, tag each trade with entry-time features
  2. Build confidence score from feature terciles
  3. Bucket trades into LOW/MED/HIGH
  4. Compare win rate, avg PnL, Sharpe per bucket
  5. Re-run backtest with tiered sizing on locked OOS
  6. Parameter sensitivity on score thresholds
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


def get_period(df, start_dt, end_dt):
    return df[df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Run backtests and extract entry-time features for each trade
# ═══════════════════════════════════════════════════════════════════════════════

def tag_trades_orb(df, strat, sym):
    """Run backtest and tag each trade with entry-time features."""
    df2 = strat.generate_signals(df.copy())
    r = run_backtest(df2, strat, sym)

    tagged = []
    for t in r.trades:
        # Find the bar at entry time
        entry_bars = df2[df2["dt"] == t.entry_time]
        if len(entry_bars) == 0:
            # Try closest bar
            idx = df2["dt"].searchsorted(t.entry_time)
            if idx >= len(df2):
                idx = len(df2) - 1
            entry_bars = df2.iloc[[idx]]

        bar = entry_bars.iloc[0]

        features = {
            "atr_pctl": bar.get("atr_percentile", 50) if pd.notna(bar.get("atr_percentile", np.nan)) else 50,
            "rel_vol": bar.get("rel_volume", 1.0) if pd.notna(bar.get("rel_volume", np.nan)) else 1.0,
            "range_pct": (bar.get("or_high", 0) - bar.get("or_low", 0)) / bar.get("or_low", 1) * 100
                         if bar.get("or_low", 0) > 0 else 0,
            "abs_gap": abs(bar.get("gap_pct", 0)) if pd.notna(bar.get("gap_pct", np.nan)) else 0,
            "minute": bar.get("minute_of_day", 600),
        }

        tagged.append({
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "direction": t.direction,
            "exit_reason": t.exit_reason,
            **features,
        })

    return pd.DataFrame(tagged), r


def tag_trades_opendrive(df, strat, sym):
    """Tag OpenDrive trades with entry features."""
    df2 = strat.generate_signals(df.copy())
    r = run_backtest(df2, strat, sym)

    tagged = []
    for t in r.trades:
        idx = df2["dt"].searchsorted(t.entry_time)
        if idx >= len(df2):
            idx = len(df2) - 1
        bar = df2.iloc[idx]

        # Estimate drive magnitude from open-to-entry price
        day_bars = df2[df2["date"] == bar["date"]]
        day_open = day_bars.iloc[0]["open"] if len(day_bars) > 0 else bar["open"]
        drive_pct = abs((t.entry_price - day_open) / day_open * 100) if day_open > 0 else 0

        features = {
            "atr_pctl": bar.get("atr_percentile", 50) if pd.notna(bar.get("atr_percentile", np.nan)) else 50,
            "rel_vol": bar.get("rel_volume", 1.0) if pd.notna(bar.get("rel_volume", np.nan)) else 1.0,
            "drive_pct": drive_pct,
            "abs_gap": abs(bar.get("gap_pct", 0)) if pd.notna(bar.get("gap_pct", np.nan)) else 0,
        }

        tagged.append({"pnl": t.pnl, "pnl_pct": t.pnl_pct, **features})

    return pd.DataFrame(tagged), r


def tag_trades_pairs(df, strat, sym):
    """Tag Pairs trades with entry features."""
    df2 = strat.generate_signals(df.copy())
    r = run_backtest(df2, strat, sym)

    # Precompute spread z-score
    spread = np.log(df2["close"]) - np.log(df2["pair_close"])
    spread_mean = spread.rolling(120, min_periods=20).mean()
    spread_std = spread.rolling(120, min_periods=20).std()
    zscore = ((spread - spread_mean) / spread_std.replace(0, np.nan)).abs()

    tagged = []
    for t in r.trades:
        idx = df2["dt"].searchsorted(t.entry_time)
        if idx >= len(df2):
            idx = len(df2) - 1
        bar = df2.iloc[idx]

        features = {
            "atr_pctl": bar.get("atr_percentile", 50) if pd.notna(bar.get("atr_percentile", np.nan)) else 50,
            "abs_zscore": zscore.iloc[idx] if idx < len(zscore) and pd.notna(zscore.iloc[idx]) else 2.0,
            "rel_vol": bar.get("rel_volume", 1.0) if pd.notna(bar.get("rel_volume", np.nan)) else 1.0,
        }

        tagged.append({"pnl": t.pnl, "pnl_pct": t.pnl_pct, **features})

    return pd.DataFrame(tagged), r


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Build confidence scores and bucket
# ═══════════════════════════════════════════════════════════════════════════════

def score_and_bucket(trades_df, feature_cols, period_label):
    """Score each trade by feature tercile ranks, bucket into LOW/MED/HIGH."""
    if len(trades_df) < 15:
        return None

    df = trades_df.copy()

    # Score = sum of tercile ranks (1/2/3) across features
    # Higher feature value = higher confidence (more vol, bigger move, etc.)
    df["score"] = 0
    for col in feature_cols:
        if col not in df.columns:
            continue
        vals = df[col].fillna(df[col].median())
        # Rank into terciles
        try:
            df[f"{col}_tercile"] = pd.qcut(vals, 3, labels=[1, 2, 3], duplicates="drop").astype(int)
        except ValueError:
            df[f"{col}_tercile"] = 2  # All same value
        df["score"] += df[f"{col}_tercile"]

    # Bucket by score terciles
    try:
        df["bucket"] = pd.qcut(df["score"], 3, labels=["LOW", "MED", "HIGH"], duplicates="drop")
    except ValueError:
        # Not enough unique scores for 3 buckets
        median_score = df["score"].median()
        df["bucket"] = df["score"].apply(
            lambda s: "LOW" if s < median_score else ("HIGH" if s > median_score else "MED")
        )

    return df


def report_buckets(df, label):
    """Print performance by confidence bucket."""
    if df is None:
        print(f"  {label}: insufficient trades for bucketing", flush=True)
        return None

    print(f"\n  {label}:", flush=True)
    print(f"  {'Bucket':<6} {'Trades':>6} {'WinRate':>7} {'AvgPnL':>9} {'AvgWin':>9} {'AvgLoss':>9}"
          f" {'PF':>5} {'TotalPnL':>10}", flush=True)
    print(f"  {'-'*5} {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*5} {'-'*10}", flush=True)

    bucket_stats = {}
    for bucket in ["LOW", "MED", "HIGH"]:
        b = df[df["bucket"] == bucket]
        if len(b) == 0:
            continue

        wins = b[b["pnl"] > 0]
        losses = b[b["pnl"] <= 0]
        wr = len(wins) / len(b) if len(b) > 0 else 0
        avg_pnl = b["pnl"].mean()
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
        gross_win = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.001
        pf = gross_win / gross_loss
        total_pnl = b["pnl"].sum()

        print(f"  {bucket:<6} {len(b):>6} {wr:>6.1%} ${avg_pnl:>8.2f} ${avg_win:>8.2f}"
              f" ${avg_loss:>8.2f} {pf:>5.2f} ${total_pnl:>9.2f}", flush=True)

        bucket_stats[bucket] = {
            "trades": len(b), "wr": wr, "avg_pnl": avg_pnl,
            "pf": pf, "total_pnl": total_pnl,
        }

    # Is HIGH actually better than LOW?
    if "HIGH" in bucket_stats and "LOW" in bucket_stats:
        h = bucket_stats["HIGH"]
        l = bucket_stats["LOW"]
        if h["avg_pnl"] > l["avg_pnl"] and h["wr"] >= l["wr"] - 0.05:
            print(f"  --> PREDICTIVE: HIGH avg ${h['avg_pnl']:.2f} > LOW avg ${l['avg_pnl']:.2f}", flush=True)
            return True
        else:
            print(f"  --> NOT PREDICTIVE: HIGH avg ${h['avg_pnl']:.2f} vs LOW avg ${l['avg_pnl']:.2f}", flush=True)
            return False

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Run analysis on DEV and OOS separately
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}", flush=True)
print("EXPERIMENT 16: CONFIDENCE-WEIGHTED SIZING OVERLAY", flush=True)
print(f"{'='*90}", flush=True)

STRATEGIES = {
    "ORB_SPY": {
        "strat": ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["SPY"]}),
        "df": data["SPY"], "sym": "SPY",
        "tag_fn": tag_trades_orb,
        "features": ["atr_pctl", "rel_vol", "range_pct", "abs_gap"],
    },
    "ORB_QQQ": {
        "strat": ORBBreakout(**{**ORB_SHARED_DEFAULTS, **SYMBOL_PROFILES["QQQ"]}),
        "df": data["QQQ"], "sym": "QQQ",
        "tag_fn": tag_trades_orb,
        "features": ["atr_pctl", "rel_vol", "range_pct", "abs_gap"],
    },
    "Pairs_GLD_TLT": {
        "strat": PairsSpread(lookback=120, entry_zscore=2.0, exit_zscore=0.5, stale_bars=90, last_entry_minute=900),
        "df": gld_full, "sym": "GLD",
        "tag_fn": tag_trades_pairs,
        "features": ["atr_pctl", "abs_zscore", "rel_vol"],
    },
    "OD_SMH": {
        "strat": OpeningDrive(**OPENDRIVE_SMH),
        "df": data["SMH"], "sym": "SMH",
        "tag_fn": tag_trades_opendrive,
        "features": ["atr_pctl", "rel_vol", "drive_pct", "abs_gap"],
    },
    "OD_XLK": {
        "strat": OpeningDrive(**OPENDRIVE_XLK),
        "df": data["XLK"], "sym": "XLK",
        "tag_fn": tag_trades_opendrive,
        "features": ["atr_pctl", "rel_vol", "drive_pct", "abs_gap"],
    },
}

dev_predictive = {}
oos_predictive = {}

for name, cfg in STRATEGIES.items():
    print(f"\n{'='*90}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*90}", flush=True)

    strat = cfg["strat"]
    features = cfg["features"]

    # DEV period
    df_dev = get_period(cfg["df"], START, DEV_END)
    trades_dev, _ = cfg["tag_fn"](df_dev, strat, cfg["sym"])
    bucketed_dev = score_and_bucket(trades_dev, features, "dev")
    dev_result = report_buckets(bucketed_dev, f"{name} DEV")
    dev_predictive[name] = dev_result

    # OOS period
    df_oos = get_period(cfg["df"], OOS_START, END)
    trades_oos, _ = cfg["tag_fn"](df_oos, strat, cfg["sym"])
    bucketed_oos = score_and_bucket(trades_oos, features, "oos")
    oos_result = report_buckets(bucketed_oos, f"{name} LOCKED OOS")
    oos_predictive[name] = oos_result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Sizing simulation for predictive strategies
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}", flush=True)
print("SIZING SIMULATION (locked OOS)", flush=True)
print("LOW=0.5x base, MED=1.0x base, HIGH=1.5x base", flush=True)
print(f"{'='*90}", flush=True)

# For strategies where confidence IS predictive on BOTH dev and OOS,
# simulate tiered sizing
for name, cfg in STRATEGIES.items():
    dev_p = dev_predictive.get(name)
    oos_p = oos_predictive.get(name)

    if not (dev_p is True and oos_p is True):
        status = "dev" if dev_p is not True else "oos"
        print(f"\n  {name}: SKIP (not predictive on {status})", flush=True)
        continue

    print(f"\n  {name}: PREDICTIVE on both periods — simulating tiered sizing", flush=True)

    # Get OOS bucketed trades
    df_oos = get_period(cfg["df"], OOS_START, END)
    trades_oos, r_base = cfg["tag_fn"](df_oos, cfg["strat"], cfg["sym"])
    bucketed = score_and_bucket(trades_oos, cfg["features"], "oos")

    if bucketed is None:
        continue

    # Uniform sizing PnL
    uniform_pnl = bucketed["pnl"].sum()
    uniform_trades = len(bucketed)

    # Tiered sizing PnL (multiply PnL by size factor)
    size_map = {"LOW": 0.5, "MED": 1.0, "HIGH": 1.5}
    bucketed["sized_pnl"] = bucketed.apply(
        lambda row: row["pnl"] * size_map.get(row["bucket"], 1.0), axis=1
    )
    tiered_pnl = bucketed["sized_pnl"].sum()

    delta = tiered_pnl - uniform_pnl
    pct_improve = (tiered_pnl / uniform_pnl - 1) * 100 if uniform_pnl != 0 else 0

    print(f"    Uniform sizing PnL:  ${uniform_pnl:+.2f}", flush=True)
    print(f"    Tiered sizing PnL:   ${tiered_pnl:+.2f}", flush=True)
    print(f"    Delta:               ${delta:+.2f} ({pct_improve:+.1f}%)", flush=True)

    # Check concentration: does HIGH bucket have > 50% of capital?
    high_count = len(bucketed[bucketed["bucket"] == "HIGH"])
    high_pct = high_count / len(bucketed) * 100
    print(f"    HIGH bucket: {high_count} trades ({high_pct:.0f}% of total)", flush=True)

    if delta > 0 and high_pct < 50:
        print(f"    --> ACCEPT: tiered sizing improves PnL without concentration", flush=True)
    elif delta > 0 and high_pct >= 50:
        print(f"    --> REJECT: improves PnL but HIGH bucket is >50% — concentration risk", flush=True)
    else:
        print(f"    --> REJECT: tiered sizing does not improve PnL", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Parameter sensitivity on score thresholds
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}", flush=True)
print("PARAMETER SENSITIVITY: different size multipliers", flush=True)
print(f"{'='*90}", flush=True)

for name, cfg in STRATEGIES.items():
    dev_p = dev_predictive.get(name)
    oos_p = oos_predictive.get(name)
    if not (dev_p is True and oos_p is True):
        continue

    df_oos = get_period(cfg["df"], OOS_START, END)
    trades_oos, _ = cfg["tag_fn"](df_oos, cfg["strat"], cfg["sym"])
    bucketed = score_and_bucket(trades_oos, cfg["features"], "oos")
    if bucketed is None:
        continue

    uniform_pnl = bucketed["pnl"].sum()
    print(f"\n  {name} (uniform PnL: ${uniform_pnl:+.2f}):", flush=True)
    print(f"  {'LOW mult':>8} {'MED mult':>8} {'HIGH mult':>9} {'Tiered PnL':>11} {'Delta':>8} {'Better?':>7}", flush=True)

    for low_m in [0.25, 0.50, 0.75]:
        for high_m in [1.25, 1.50, 1.75, 2.00]:
            sized = bucketed["pnl"].copy()
            sized[bucketed["bucket"] == "LOW"] *= low_m
            sized[bucketed["bucket"] == "HIGH"] *= high_m
            total = sized.sum()
            delta = total - uniform_pnl
            better = "YES" if delta > 0 else "no"
            print(f"  {low_m:>8.2f} {1.0:>8.2f} {high_m:>9.2f} ${total:>10.2f} ${delta:>7.2f} {better:>7}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}", flush=True)
print("FINAL VERDICT", flush=True)
print(f"{'='*90}", flush=True)

any_accepted = False
for name in STRATEGIES:
    dev_p = dev_predictive.get(name)
    oos_p = oos_predictive.get(name)

    if dev_p is True and oos_p is True:
        status = "PREDICTIVE (both periods)"
        any_accepted = True
    elif dev_p is True:
        status = "dev only (overfit)"
    elif oos_p is True:
        status = "oos only (unstable)"
    else:
        status = "NOT PREDICTIVE"

    print(f"  {name:<18} {status}", flush=True)

if not any_accepted:
    print(f"\n  CONCLUSION: No strategy shows reliable confidence prediction on both", flush=True)
    print(f"  dev and locked OOS periods. The confidence overlay is REJECTED.", flush=True)
    print(f"  Continue with uniform position sizing.", flush=True)
else:
    print(f"\n  CONCLUSION: See sizing simulation above for accepted strategies.", flush=True)
    print(f"  Only implement tiered sizing for strategies that passed both", flush=True)
    print(f"  periods AND the sensitivity check.", flush=True)

print(f"\n{'='*90}", flush=True)
print("EXPERIMENT 16 COMPLETE", flush=True)
print(f"{'='*90}", flush=True)
