#!/usr/bin/env python3
"""Test potential improvements with walk-forward OOS validation.

Tests:
1. Chop proxy filter: OR_range / ATR ratio (no look-ahead)
2. Time-decay exit: exit stale trades after N minutes without profit
3. Gap size minimum filter
4. Combinations

Each test uses walk-forward (60/20/20) to validate OOS.
"""

import sys, os, logging
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.orb import ORBBreakout
from trading.backtest.engine import run_backtest
from trading.backtest.walkforward import walk_forward

ET = pytz.timezone("America/New_York")
start = datetime(2025, 1, 2, tzinfo=ET)
end = datetime(2026, 4, 4, tzinfo=ET)

# ── Load data once ──
data = {}
for sym in ["SPY", "QQQ"]:
    df = get_minute_bars(sym, start, end, use_cache=True)
    df = prepare_features(df)
    data[sym] = df

def test_strategy(strat, label):
    """Run a strategy and return combined metrics."""
    results = []
    for sym in ["SPY", "QQQ"]:
        df = data[sym].copy()
        df = strat.generate_signals(df)
        r = run_backtest(df, strat, sym)
        results.append(r)

    total_trades = sum(r.num_trades for r in results)
    if total_trades == 0:
        print(f"  {label}: NO TRADES")
        return

    avg_sharpe = np.mean([r.sharpe_ratio for r in results])
    avg_pf = np.mean([r.profit_factor for r in results])
    avg_wr = np.mean([r.win_rate for r in results])
    total_pnl = sum(r.equity_curve.iloc[-1] - 100_000 for r in results)

    print(f"  {label:45s}: Sharpe={avg_sharpe:.2f} PF={avg_pf:.2f} "
          f"WR={avg_wr*100:.1f}% Trades={total_trades:4d} PnL=${total_pnl:+,.0f}")
    return results

def test_walkforward(strat, label):
    """Run walk-forward OOS validation."""
    wf_results = []
    for sym in ["SPY", "QQQ"]:
        df = data[sym].copy()
        df = strat.generate_signals(df)
        wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)
        wf_results.append(wf)

    if not wf_results:
        print(f"  {label} OOS: NO DATA")
        return

    avg_oos_sharpe = np.mean([w.sharpe_ratio for w in wf_results])
    total_oos_trades = sum(w.total_trades for w in wf_results)
    avg_oos_pf = np.mean([w.profit_factor for w in wf_results])
    avg_oos_wr = np.mean([w.win_rate for w in wf_results])
    total_return = np.mean([w.total_return for w in wf_results])
    print(f"  {label:45s}: OOS Sharpe={avg_oos_sharpe:.2f} PF={avg_oos_pf:.2f} "
          f"WR={avg_oos_wr*100:.1f}% Trades={total_oos_trades:4d} Ret={total_return*100:+.2f}%")
    return avg_oos_sharpe


# ═══════════════════════════════════════════════════════════
print("=" * 90)
print("BASELINE: Current improved system")
print("=" * 90)
baseline = ORBBreakout(range_minutes=15, target_multiple=1.5,
                       min_atr_percentile=25, min_breakout_volume=1.2,
                       last_entry_minute=15*60)
test_strategy(baseline, "Baseline (current)")
test_walkforward(baseline, "Baseline (current)")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 1: CHOP PROXY FILTER (OR_range / ATR)")
print("  If OR already consumed most of expected daily range, skip the day")
print("=" * 90)

# First, analyze the chop proxy distribution
print("\n  Analyzing chop proxy (or_range_pct / atr_pct) distribution...")
for sym in ["SPY", "QQQ"]:
    df = data[sym]
    daily = df.groupby("date").agg(
        or_high=("or_high", "first"),
        or_low=("or_low", "first"),
        atr_pct=("atr_pct", "first"),
    ).dropna()
    daily["or_range_pct"] = (daily["or_high"] - daily["or_low"]) / daily["or_low"] * 100
    daily["chop_proxy"] = daily["or_range_pct"] / daily["atr_pct"].replace(0, np.nan)
    print(f"  {sym} chop_proxy: mean={daily['chop_proxy'].mean():.2f} "
          f"median={daily['chop_proxy'].median():.2f} "
          f"p25={daily['chop_proxy'].quantile(0.25):.2f} "
          f"p75={daily['chop_proxy'].quantile(0.75):.2f}")

# To implement chop filter, we need to add the feature to the dataframe
# and modify ORB to use it. Let's test by pre-filtering signals.

from trading.strategies.orb import ORBBreakout as _ORB

class ORBWithChopFilter(_ORB):
    """ORB with chop proxy filter."""
    name = "orb_chop"

    def __init__(self, max_chop_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_chop_ratio = max_chop_ratio

    def get_params(self):
        d = super().get_params()
        d["max_chop"] = self.max_chop_ratio
        return d

    def generate_signals(self, df):
        df = df.copy()

        # Compute chop proxy per day: OR_range_pct / atr_pct
        daily = df.groupby("date").agg(
            or_high=("or_high", "first"),
            or_low=("or_low", "first"),
            atr_pct=("atr_pct", "first"),
        )
        daily["or_range_pct"] = (daily["or_high"] - daily["or_low"]) / daily["or_low"] * 100
        daily["chop_proxy"] = daily["or_range_pct"] / daily["atr_pct"].replace(0, np.nan)
        df["chop_proxy"] = df["date"].map(daily["chop_proxy"])

        # Generate base signals
        df = super().generate_signals(df)

        # Zero out signals on choppy days
        choppy = df["chop_proxy"] > self.max_chop_ratio
        df.loc[choppy, "signal"] = 0
        return df


for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    strat = ORBWithChopFilter(
        max_chop_ratio=threshold,
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60,
    )
    test_strategy(strat, f"Chop filter <= {threshold}")

# WF on best chop thresholds
print("\n  Walk-forward validation of chop filters:")
for threshold in [0.40, 0.45, 0.50]:
    strat = ORBWithChopFilter(
        max_chop_ratio=threshold,
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60,
    )
    test_walkforward(strat, f"Chop filter <= {threshold}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 2: TIME-DECAY EXIT (exit stale trades)")
print("  If trade is not profitable after N minutes, exit")
print("=" * 90)

class ORBWithTimeDecay(_ORB):
    """ORB with time-decay exit for stale trades."""
    name = "orb_timedecay"

    def __init__(self, stale_minutes=120, **kwargs):
        super().__init__(**kwargs)
        self.stale_minutes = stale_minutes

    def get_params(self):
        d = super().get_params()
        d["stale_min"] = self.stale_minutes
        return d

    def generate_signals(self, df):
        df = df.copy()
        df["signal"] = 0

        if "or_high" not in df.columns:
            return df

        has_atr_filter = self.min_atr_percentile > 0 and "atr_percentile" in df.columns
        has_vol_filter = self.min_breakout_volume > 0 and "rel_volume" in df.columns
        has_late_cutoff = self.last_entry_minute > 0

        or_end = 9 * 60 + 30 + self.range_minutes
        signals = np.zeros(len(df))
        position = 0
        entry_price = 0.0
        entry_bar = 0
        or_high = 0.0
        or_low = 0.0
        target = 0.0
        stop = 0.0
        current_date = None
        day_skipped = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                or_high = row["or_high"] if pd.notna(row["or_high"]) else 0
                or_low = row["or_low"] if pd.notna(row["or_low"]) else 0

                if has_atr_filter:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_percentile:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            if row["minute_of_day"] < or_end:
                signals[i] = 0
                continue

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            if or_high <= 0 or or_low <= 0:
                signals[i] = 0
                continue

            range_width = or_high - or_low
            range_pct = range_width / or_low

            if range_pct < self.min_range_pct or range_pct > self.max_range_pct:
                signals[i] = 0
                position = 0
                continue

            if position == 0:
                if has_late_cutoff and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                if has_vol_filter:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_breakout_volume:
                        signals[i] = 0
                        continue

                if row["close"] > or_high:
                    position = 1
                    entry_price = row["close"]
                    entry_bar = i
                    target = entry_price + range_width * self.target_multiple
                    stop = or_low
                elif row["close"] < or_low:
                    position = -1
                    entry_price = row["close"]
                    entry_bar = i
                    target = entry_price - range_width * self.target_multiple
                    stop = or_high

            elif position == 1:
                if row["close"] >= target or row["close"] <= stop:
                    position = 0
                # Time decay: exit if not profitable after N minutes
                elif (i - entry_bar) >= self.stale_minutes and row["close"] <= entry_price:
                    position = 0

            elif position == -1:
                if row["close"] <= target or row["close"] >= stop:
                    position = 0
                elif (i - entry_bar) >= self.stale_minutes and row["close"] >= entry_price:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


for stale_min in [60, 90, 120, 180]:
    strat = ORBWithTimeDecay(
        stale_minutes=stale_min,
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60,
    )
    test_strategy(strat, f"Time decay exit {stale_min}m")

print("\n  Walk-forward validation of time decay:")
for stale_min in [60, 90, 120]:
    strat = ORBWithTimeDecay(
        stale_minutes=stale_min,
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60,
    )
    test_walkforward(strat, f"Time decay exit {stale_min}m")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 3: GAP SIZE MINIMUM FILTER")
print("  Larger gaps = more directional days")
print("=" * 90)

class ORBWithGapFilter(_ORB):
    """ORB with minimum gap size filter."""
    name = "orb_gap"

    def __init__(self, min_gap_pct=0.2, **kwargs):
        super().__init__(**kwargs)
        self.min_gap_pct = min_gap_pct

    def get_params(self):
        d = super().get_params()
        d["min_gap"] = self.min_gap_pct
        return d

    def generate_signals(self, df):
        df = super().generate_signals(df)

        if "gap_pct" in df.columns:
            small_gap = df["gap_pct"].abs() < self.min_gap_pct
            df.loc[small_gap, "signal"] = 0
        return df


for min_gap in [0.1, 0.15, 0.2, 0.3]:
    strat = ORBWithGapFilter(
        min_gap_pct=min_gap,
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60,
    )
    test_strategy(strat, f"Min gap >= {min_gap}%")

print("\n  Walk-forward validation of gap filters:")
for min_gap in [0.1, 0.15, 0.2]:
    strat = ORBWithGapFilter(
        min_gap_pct=min_gap,
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60,
    )
    test_walkforward(strat, f"Min gap >= {min_gap}%")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 4: COMBINED (best from above)")
print("=" * 90)

class ORBCombined(_ORB):
    """ORB with chop filter + time decay + gap filter."""
    name = "orb_combined"

    def __init__(self, max_chop_ratio=0.5, stale_minutes=120, min_gap_pct=0.0, **kwargs):
        super().__init__(**kwargs)
        self.max_chop_ratio = max_chop_ratio
        self.stale_minutes = stale_minutes
        self.min_gap_pct = min_gap_pct

    def get_params(self):
        d = super().get_params()
        if self.max_chop_ratio < 99:
            d["max_chop"] = self.max_chop_ratio
        if self.stale_minutes < 999:
            d["stale_min"] = self.stale_minutes
        if self.min_gap_pct > 0:
            d["min_gap"] = self.min_gap_pct
        return d

    def generate_signals(self, df):
        df = df.copy()

        # Compute chop proxy
        if self.max_chop_ratio < 99:
            daily = df.groupby("date").agg(
                or_high=("or_high", "first"),
                or_low=("or_low", "first"),
                atr_pct=("atr_pct", "first"),
            )
            daily["or_range_pct"] = (daily["or_high"] - daily["or_low"]) / daily["or_low"] * 100
            daily["chop_proxy"] = daily["or_range_pct"] / daily["atr_pct"].replace(0, np.nan)
            df["chop_proxy"] = df["date"].map(daily["chop_proxy"])

        # Generate base ORB signals (with time-decay built in)
        df["signal"] = 0
        if "or_high" not in df.columns:
            return df

        has_atr_filter = self.min_atr_percentile > 0 and "atr_percentile" in df.columns
        has_vol_filter = self.min_breakout_volume > 0 and "rel_volume" in df.columns
        has_late_cutoff = self.last_entry_minute > 0
        has_chop = self.max_chop_ratio < 99
        has_gap = self.min_gap_pct > 0
        has_stale = self.stale_minutes < 999

        or_end = 9 * 60 + 30 + self.range_minutes
        signals = np.zeros(len(df))
        position = 0
        entry_price = 0.0
        entry_bar = 0
        or_high = 0.0
        or_low = 0.0
        target = 0.0
        stop = 0.0
        current_date = None
        day_skipped = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                or_high = row["or_high"] if pd.notna(row["or_high"]) else 0
                or_low = row["or_low"] if pd.notna(row["or_low"]) else 0

                if has_atr_filter:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_percentile:
                        day_skipped = True

                if has_chop and not day_skipped:
                    cp = row.get("chop_proxy", 0) if "chop_proxy" in df.columns else df.iloc[i].get("chop_proxy", 0)
                    if pd.notna(cp) and cp > self.max_chop_ratio:
                        day_skipped = True

                if has_gap and not day_skipped:
                    gap = row.get("gap_pct", 0)
                    if pd.notna(gap) and abs(gap) < self.min_gap_pct:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            if row["minute_of_day"] < or_end:
                signals[i] = 0
                continue

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            if or_high <= 0 or or_low <= 0:
                signals[i] = 0
                continue

            range_width = or_high - or_low
            range_pct = range_width / or_low

            if range_pct < self.min_range_pct or range_pct > self.max_range_pct:
                signals[i] = 0
                position = 0
                continue

            if position == 0:
                if has_late_cutoff and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                if has_vol_filter:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_breakout_volume:
                        signals[i] = 0
                        continue

                if row["close"] > or_high:
                    position = 1
                    entry_price = row["close"]
                    entry_bar = i
                    target = entry_price + range_width * self.target_multiple
                    stop = or_low
                elif row["close"] < or_low:
                    position = -1
                    entry_price = row["close"]
                    entry_bar = i
                    target = entry_price - range_width * self.target_multiple
                    stop = or_high

            elif position == 1:
                if row["close"] >= target or row["close"] <= stop:
                    position = 0
                elif has_stale and (i - entry_bar) >= self.stale_minutes and row["close"] <= entry_price:
                    position = 0

            elif position == -1:
                if row["close"] <= target or row["close"] >= stop:
                    position = 0
                elif has_stale and (i - entry_bar) >= self.stale_minutes and row["close"] >= entry_price:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


# Test combinations
combos = [
    {"max_chop_ratio": 0.45, "stale_minutes": 999, "min_gap_pct": 0.0},
    {"max_chop_ratio": 0.45, "stale_minutes": 90, "min_gap_pct": 0.0},
    {"max_chop_ratio": 0.45, "stale_minutes": 999, "min_gap_pct": 0.15},
    {"max_chop_ratio": 0.45, "stale_minutes": 90, "min_gap_pct": 0.15},
    {"max_chop_ratio": 0.50, "stale_minutes": 90, "min_gap_pct": 0.10},
    {"max_chop_ratio": 99, "stale_minutes": 90, "min_gap_pct": 0.15},
]

for combo in combos:
    label = f"chop={combo['max_chop_ratio']} stale={combo['stale_minutes']} gap={combo['min_gap_pct']}"
    strat = ORBCombined(
        **combo,
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60,
    )
    test_strategy(strat, label)

print("\n  Walk-forward validation of combinations:")
for combo in combos:
    label = f"chop={combo['max_chop_ratio']} stale={combo['stale_minutes']} gap={combo['min_gap_pct']}"
    strat = ORBCombined(
        **combo,
        range_minutes=15, target_multiple=1.5,
        min_atr_percentile=25, min_breakout_volume=1.2,
        last_entry_minute=15*60,
    )
    test_walkforward(strat, label)

print("\n" + "=" * 90)
print("RESEARCH COMPLETE")
print("=" * 90)
