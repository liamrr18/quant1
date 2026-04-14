#!/usr/bin/env python3
"""Phase 2, Experiment 12: Screen new strategy families.

Tests strategy concepts that are structurally different from ORB:
1. Gap Fade: trade against overnight gaps (mean-reverting)
2. Late-Day Momentum: ride afternoon directional continuation
3. ORB Fade: fade failed breakouts (mean-reverting)

All tested on SPY and QQQ dev period (Jan-Nov 2025).
Uses same walk-forward and cost model as baseline.
"""

import sys, os, logging, warnings
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.base import Strategy
from trading.backtest.engine import run_backtest
from trading.backtest.walkforward import walk_forward
from trading.config import ORB_SHARED_DEFAULTS, SYMBOL_PROFILES

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)


# ═══════════════════════════════════════════════════════════
# Strategy 1: Gap Fade
# ═══════════════════════════════════════════════════════════

class GapFade(Strategy):
    """Fade overnight gaps that are likely to fill.

    Logic: Large overnight gaps tend to mean-revert toward the previous close
    during the trading day. Enter in the opposite direction of the gap after
    the opening range forms, targeting the gap fill level.

    Entry: After opening range, if gap > threshold, enter short (gap up) or
           long (gap down). Requires gap not already filled.
    Exit: Gap fills to previous close, fixed stop, stale exit, or EOD.
    """
    name = "gap_fade"

    def __init__(self, min_gap_pct=0.3, max_gap_pct=2.0,
                 stop_multiple=1.5, stale_bars=120,
                 wait_minutes=15, entry_confirm_bars=3):
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.stop_multiple = stop_multiple  # Stop at gap * this multiple from entry
        self.stale_bars = stale_bars        # Give up after this many bars
        self.wait_minutes = wait_minutes    # Wait after open before entry
        self.entry_confirm_bars = entry_confirm_bars  # Bars of adverse move before entry

    def get_params(self):
        return {
            "min_gap": self.min_gap_pct, "max_gap": self.max_gap_pct,
            "stop_mult": self.stop_multiple, "stale": self.stale_bars,
            "wait": self.wait_minutes,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "gap_pct" not in df.columns or "prev_close" not in df.columns:
            return df

        or_end = 9 * 60 + 30 + self.wait_minutes
        signals = np.zeros(len(df))
        position = 0
        entry_price = 0.0
        entry_bar = 0
        target = 0.0
        stop = 0.0
        current_date = None
        day_gap = 0.0
        prev_close_val = 0.0
        day_skipped = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                day_gap = row.get("gap_pct", 0)
                prev_close_val = row.get("prev_close", 0)

                if pd.isna(day_gap) or pd.isna(prev_close_val) or prev_close_val <= 0:
                    day_skipped = True
                elif abs(day_gap) < self.min_gap_pct or abs(day_gap) > self.max_gap_pct:
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

            if position == 0:
                # Gap up -> short (fade), gap down -> long (fade)
                if day_gap > 0 and row["close"] > prev_close_val:
                    # Gap up, price still above prev close -> short to fade
                    position = -1
                    entry_price = row["close"]
                    entry_bar = i
                    target = prev_close_val  # Target = gap fill
                    gap_size = abs(day_gap / 100 * prev_close_val)
                    stop = entry_price + gap_size * self.stop_multiple

                elif day_gap < 0 and row["close"] < prev_close_val:
                    # Gap down, price still below prev close -> long to fade
                    position = 1
                    entry_price = row["close"]
                    entry_bar = i
                    target = prev_close_val
                    gap_size = abs(day_gap / 100 * prev_close_val)
                    stop = entry_price - gap_size * self.stop_multiple

            elif position == 1:
                # Long: target (gap fill) or stop
                if row["close"] >= target or row["close"] <= stop:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            elif position == -1:
                # Short: target (gap fill) or stop
                if row["close"] <= target or row["close"] >= stop:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


# ═══════════════════════════════════════════════════════════
# Strategy 2: ORB Fade (failed breakout reversal)
# ═══════════════════════════════════════════════════════════

class ORBFade(Strategy):
    """Fade failed ORB breakouts.

    Logic: When a breakout fails (price returns inside the range after breaking
    out), this signals a false breakout. Enter in the opposite direction
    targeting the other side of the range.

    This is anti-correlated with ORB by design.
    """
    name = "orb_fade"

    def __init__(self, range_minutes=15, target_multiple=1.0,
                 min_range_pct=0.001, max_range_pct=0.008,
                 fail_bars=15, stale_bars=120,
                 min_atr_percentile=0, min_breakout_volume=0):
        self.range_minutes = range_minutes
        self.target_multiple = target_multiple
        self.min_range_pct = min_range_pct
        self.max_range_pct = max_range_pct
        self.fail_bars = fail_bars          # Must fail within this many bars
        self.stale_bars = stale_bars
        self.min_atr_percentile = min_atr_percentile
        self.min_breakout_volume = min_breakout_volume

    def get_params(self):
        return {
            "range_min": self.range_minutes, "target_mult": self.target_multiple,
            "fail_bars": self.fail_bars, "stale": self.stale_bars,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "or_high" not in df.columns or "or_low" not in df.columns:
            return df

        or_end = 9 * 60 + 30 + self.range_minutes
        signals = np.zeros(len(df))
        position = 0
        entry_price = 0.0
        entry_bar = 0
        target = 0.0
        stop = 0.0
        or_high = 0.0
        or_low = 0.0
        range_width = 0.0
        current_date = None
        day_skipped = False

        # Tracking breakout state
        broke_high = False
        broke_low = False
        breakout_bar = 0
        breakout_dir = 0  # 1 = broke high, -1 = broke low

        has_atr_filter = self.min_atr_percentile > 0 and "atr_percentile" in df.columns

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                broke_high = False
                broke_low = False
                breakout_dir = 0
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
                continue

            mid = (or_high + or_low) / 2

            if position == 0:
                # Track breakouts
                if not broke_high and row["close"] > or_high:
                    broke_high = True
                    breakout_bar = i
                    breakout_dir = 1
                elif not broke_low and row["close"] < or_low:
                    broke_low = True
                    breakout_bar = i
                    breakout_dir = -1

                # Detect failed breakout: price returned inside range
                if breakout_dir == 1 and broke_high and row["close"] < mid:
                    if (i - breakout_bar) <= self.fail_bars:
                        # Failed upside breakout -> go short
                        position = -1
                        entry_price = row["close"]
                        entry_bar = i
                        target = or_low - range_width * self.target_multiple
                        stop = or_high + range_width * 0.5
                        breakout_dir = 0  # Reset

                elif breakout_dir == -1 and broke_low and row["close"] > mid:
                    if (i - breakout_bar) <= self.fail_bars:
                        # Failed downside breakout -> go long
                        position = 1
                        entry_price = row["close"]
                        entry_bar = i
                        target = or_high + range_width * self.target_multiple
                        stop = or_low - range_width * 0.5
                        breakout_dir = 0

            elif position == 1:
                if row["close"] >= target or row["close"] <= stop:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            elif position == -1:
                if row["close"] <= target or row["close"] >= stop:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


# ═══════════════════════════════════════════════════════════
# Strategy 3: Late Day Momentum
# ═══════════════════════════════════════════════════════════

class LateDayMomentum(Strategy):
    """Capture afternoon momentum continuation.

    Logic: If the stock has moved consistently from late morning to early
    afternoon, enter in the direction of the move and ride to close.
    Uses VWAP direction as a confirmation filter.
    """
    name = "late_momentum"

    def __init__(self, entry_hour=13, lookback_bars=60,
                 min_move_pct=0.15, stale_bars=90):
        self.entry_hour = entry_hour  # Enter at this hour ET
        self.lookback_bars = lookback_bars
        self.min_move_pct = min_move_pct  # Min directional move to enter
        self.stale_bars = stale_bars

    def get_params(self):
        return {
            "entry_hour": self.entry_hour, "lookback": self.lookback_bars,
            "min_move": self.min_move_pct,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        entry_minute = self.entry_hour * 60
        signals = np.zeros(len(df))
        position = 0
        entry_price = 0.0
        entry_bar = 0
        stop = 0.0
        current_date = None

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            if position == 0 and row["minute_of_day"] == entry_minute:
                # Check directional move over lookback
                if i < self.lookback_bars:
                    signals[i] = 0
                    continue

                lookback_close = float(df.iloc[i - self.lookback_bars]["close"])
                current_close = row["close"]
                if lookback_close <= 0:
                    signals[i] = 0
                    continue

                move_pct = (current_close - lookback_close) / lookback_close * 100

                if move_pct > self.min_move_pct:
                    position = 1
                    entry_price = current_close
                    entry_bar = i
                    stop = entry_price * (1 - self.min_move_pct / 100 * 2)
                elif move_pct < -self.min_move_pct:
                    position = -1
                    entry_price = current_close
                    entry_bar = i
                    stop = entry_price * (1 + self.min_move_pct / 100 * 2)

            elif position == 1:
                if row["close"] <= stop:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            elif position == -1:
                if row["close"] >= stop:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


# ═══════════════════════════════════════════════════════════
# Evaluation harness
# ═══════════════════════════════════════════════════════════

def eval_walkforward(sym, df, strat, label=""):
    df = strat.generate_signals(df)
    wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)

    all_daily = []
    for oos_r in wf.oos_results:
        dr = oos_r.daily_returns
        if dr is not None and len(dr) > 0:
            all_daily.append(dr)

    if not all_daily:
        return None

    combined = pd.concat(all_daily).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    n = len(combined)
    if n < 10 or combined.std() == 0:
        return None

    sharpe = (combined.mean() / combined.std()) * np.sqrt(252)
    total_ret = (1 + combined).prod() - 1
    cum = (1 + combined).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    window_sharpes = [r.sharpe_ratio for r in wf.oos_results]
    pos_windows = sum(1 for s in window_sharpes if s > 0)

    return {
        "sym": sym, "label": label, "sharpe": sharpe, "return": total_ret,
        "max_dd": max_dd, "trades": wf.total_trades,
        "win_rate": wf.win_rate, "pf": wf.profit_factor,
        "windows_pos": f"{pos_windows}/{len(window_sharpes)}",
        "daily_returns": combined,
    }


# ═══════════════════════════════════════════════════════════
# Run all strategy screens
# ═══════════════════════════════════════════════════════════

print("Loading data...")
data = {}
for sym in ["SPY", "QQQ"]:
    df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"  {sym}: {len(df)} bars")

# Get baseline ORB results for comparison
print("\n" + "=" * 100)
print("BASELINE: Production ORB profiles (dev period walk-forward)")
print("=" * 100)

from trading.strategies.orb import ORBBreakout

baseline_results = {}
for sym in ["SPY", "QQQ"]:
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(SYMBOL_PROFILES.get(sym, {}))
    strat = ORBBreakout(**params)
    df_dev = data[sym].loc[data[sym]["dt"] <= DEV_END].copy()
    r = eval_walkforward(sym, df_dev, strat, f"ORB-{sym}")
    baseline_results[sym] = r
    if r:
        print(f"  {sym} ORB: Sharpe={r['sharpe']:.2f}  Ret={r['return']:+.2%}"
              f"  T={r['trades']}  WR={r['win_rate']:.1%}  Win={r['windows_pos']}")


# ── Strategy 1: Gap Fade variants ──
print("\n" + "=" * 100)
print("STRATEGY 1: GAP FADE (mean-revert overnight gaps)")
print("=" * 100)

gap_fade_variants = [
    ("gap>=0.3%, stop 1.5x", {"min_gap_pct": 0.3, "stop_multiple": 1.5}),
    ("gap>=0.3%, stop 2.0x", {"min_gap_pct": 0.3, "stop_multiple": 2.0}),
    ("gap>=0.5%, stop 1.5x", {"min_gap_pct": 0.5, "stop_multiple": 1.5}),
    ("gap>=0.5%, stop 2.0x", {"min_gap_pct": 0.5, "stop_multiple": 2.0}),
    ("gap>=0.3%, stale 60", {"min_gap_pct": 0.3, "stop_multiple": 1.5, "stale_bars": 60}),
    ("gap>=0.3%, stale 180", {"min_gap_pct": 0.3, "stop_multiple": 1.5, "stale_bars": 180}),
    ("gap>=0.2%, stop 1.5x", {"min_gap_pct": 0.2, "stop_multiple": 1.5}),
]

for sym in ["SPY", "QQQ"]:
    print(f"\n  {sym}:")
    df_dev = data[sym].loc[data[sym]["dt"] <= DEV_END].copy()
    for label, kwargs in gap_fade_variants:
        strat = GapFade(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<30} S={r['sharpe']:>6.2f}  Ret={r['return']:>+7.2%}"
                  f"  DD={r['max_dd']:>6.2%}  T={r['trades']:>4}  WR={r['win_rate']:.1%}"
                  f"  PF={r['pf']:.2f}  Win={r['windows_pos']}")
        else:
            print(f"    {label:<30} -> no valid data")


# ── Strategy 2: ORB Fade variants ──
print("\n" + "=" * 100)
print("STRATEGY 2: ORB FADE (failed breakout reversal)")
print("=" * 100)

orb_fade_variants = [
    ("fail 15b, target 1.0x", {"fail_bars": 15, "target_multiple": 1.0}),
    ("fail 15b, target 0.5x", {"fail_bars": 15, "target_multiple": 0.5}),
    ("fail 30b, target 1.0x", {"fail_bars": 30, "target_multiple": 1.0}),
    ("fail 30b, target 0.5x", {"fail_bars": 30, "target_multiple": 0.5}),
    ("fail 15b, stale 60", {"fail_bars": 15, "target_multiple": 1.0, "stale_bars": 60}),
    ("fail 15b + ATR25", {"fail_bars": 15, "target_multiple": 1.0, "min_atr_percentile": 25}),
]

for sym in ["SPY", "QQQ"]:
    print(f"\n  {sym}:")
    df_dev = data[sym].loc[data[sym]["dt"] <= DEV_END].copy()
    for label, kwargs in orb_fade_variants:
        strat = ORBFade(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<30} S={r['sharpe']:>6.2f}  Ret={r['return']:>+7.2%}"
                  f"  DD={r['max_dd']:>6.2%}  T={r['trades']:>4}  WR={r['win_rate']:.1%}"
                  f"  PF={r['pf']:.2f}  Win={r['windows_pos']}")
        else:
            print(f"    {label:<30} -> no valid data")


# ── Strategy 3: Late Day Momentum ──
print("\n" + "=" * 100)
print("STRATEGY 3: LATE DAY MOMENTUM (afternoon continuation)")
print("=" * 100)

late_mom_variants = [
    ("entry 13:00, look 60", {"entry_hour": 13, "lookback_bars": 60, "min_move_pct": 0.15}),
    ("entry 13:00, look 90", {"entry_hour": 13, "lookback_bars": 90, "min_move_pct": 0.15}),
    ("entry 13:00, move 0.2%", {"entry_hour": 13, "lookback_bars": 60, "min_move_pct": 0.20}),
    ("entry 14:00, look 60", {"entry_hour": 14, "lookback_bars": 60, "min_move_pct": 0.15}),
    ("entry 14:00, look 90", {"entry_hour": 14, "lookback_bars": 90, "min_move_pct": 0.15}),
    ("entry 12:00, look 60", {"entry_hour": 12, "lookback_bars": 60, "min_move_pct": 0.15}),
]

for sym in ["SPY", "QQQ"]:
    print(f"\n  {sym}:")
    df_dev = data[sym].loc[data[sym]["dt"] <= DEV_END].copy()
    for label, kwargs in late_mom_variants:
        strat = LateDayMomentum(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<30} S={r['sharpe']:>6.2f}  Ret={r['return']:>+7.2%}"
                  f"  DD={r['max_dd']:>6.2%}  T={r['trades']:>4}  WR={r['win_rate']:.1%}"
                  f"  PF={r['pf']:.2f}  Win={r['windows_pos']}")
        else:
            print(f"    {label:<30} -> no valid data")


# ── Portfolio combination test ──
print("\n" + "=" * 100)
print("PORTFOLIO COMBINATION (any strategy with Sharpe > 0.5 on dev)")
print("=" * 100)

# Collect all results with positive Sharpe
all_strat_results = {}

# Re-run best variants of each strategy type to get daily returns
for sym in ["SPY", "QQQ"]:
    df_dev = data[sym].loc[data[sym]["dt"] <= DEV_END].copy()

    # Gap Fade best
    for label, kwargs in gap_fade_variants:
        strat = GapFade(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r and r["sharpe"] > 0.3:
            key = f"GapFade-{sym}-{label}"
            all_strat_results[key] = r

    # ORB Fade best
    for label, kwargs in orb_fade_variants:
        strat = ORBFade(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r and r["sharpe"] > 0.3:
            key = f"ORBFade-{sym}-{label}"
            all_strat_results[key] = r

    # Late Mom best
    for label, kwargs in late_mom_variants:
        strat = LateDayMomentum(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r and r["sharpe"] > 0.3:
            key = f"LateMom-{sym}-{label}"
            all_strat_results[key] = r

# Baseline portfolio
baseline_daily = {}
for sym in ["SPY", "QQQ"]:
    if baseline_results.get(sym):
        baseline_daily[sym] = baseline_results[sym]["daily_returns"]

if len(baseline_daily) == 2:
    port_base = pd.DataFrame(baseline_daily).fillna(0).mean(axis=1)
    base_sharpe = (port_base.mean() / port_base.std()) * np.sqrt(252) if port_base.std() > 0 else 0
    print(f"\n  Baseline SPY+QQQ ORB portfolio: Sharpe={base_sharpe:.2f}")

    for key, r in sorted(all_strat_results.items(), key=lambda x: -x[1]["sharpe"]):
        combined = dict(baseline_daily)
        # Add as a new "symbol" in the portfolio
        combined[key] = r["daily_returns"]
        port_df = pd.DataFrame(combined).fillna(0)
        port_ret = port_df.mean(axis=1)

        if len(port_ret) < 10 or port_ret.std() == 0:
            continue

        new_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252)
        delta = new_sharpe - base_sharpe

        # Correlation with baseline
        corr = r["daily_returns"].corr(port_base)

        print(f"  + {key:<45} portfolio={new_sharpe:.2f} (delta={delta:+.2f})"
              f"  corr(base)={corr:.2f}  standalone={r['sharpe']:.2f}")

print("\n" + "=" * 100)
print("STRATEGY SCREENING COMPLETE")
print("=" * 100)
