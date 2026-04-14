#!/usr/bin/env python3
"""Phase 5, Experiment 14: Broad screen of genuinely new strategy families.

Tests 5 strategy concepts that are structurally different from ORB:
1. Mean Reversion on Extreme Moves (VWAP Z-score fade)
2. VWAP Trend Following (dynamic trend anchor)
3. Volatility Compression Breakout (Bollinger squeeze)
4. Intraday Momentum Score (multi-factor composite)
5. Gap Continuation (overnight information event)

Tested across: SPY, QQQ, GLD, XLE, XLK, SMH, TLT
Dev period only (Jan-Nov 2025). Walk-forward OOS validation.
Same cost model: $0.01/share slippage, $0.00 commission, 30% position sizing.
"""

import sys, os, io, logging, warnings
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

# Force unbuffered output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)
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

UNIVERSE = ["SPY", "QQQ", "GLD", "XLE", "XLK", "SMH", "TLT"]


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Mean Reversion on Extreme Intraday Moves
# ═══════════════════════════════════════════════════════════════════════════════

class MeanRevExtreme(Strategy):
    """Fade extreme intraday moves using VWAP deviation Z-score.

    Structural difference from ORB: Bets AGAINST momentum. Uses dynamic VWAP
    levels (not fixed opening range). Can trigger at any time, not just after
    opening range. Anti-correlated with ORB by design.

    Entry: When VWAP deviation Z-score exceeds threshold, enter opposite direction.
           Requires RSI confirmation (oversold for longs, overbought for shorts).
    Exit: Price reverts toward VWAP (deviation drops below exit threshold),
          fixed stop, stale exit, or EOD.
    """
    name = "mean_rev_extreme"

    def __init__(self, entry_zscore=2.0, exit_zscore=0.5,
                 rsi_oversold=30, rsi_overbought=70,
                 min_atr_pctl=0, stale_bars=90,
                 cooldown_bars=15, last_entry_minute=900):
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_atr_pctl = min_atr_pctl
        self.stale_bars = stale_bars
        self.cooldown_bars = cooldown_bars
        self.last_entry_minute = last_entry_minute

    def get_params(self):
        return {
            "entry_z": self.entry_zscore, "exit_z": self.exit_zscore,
            "rsi_os": self.rsi_oversold, "rsi_ob": self.rsi_overbought,
            "stale": self.stale_bars, "cooldown": self.cooldown_bars,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "vwap_dev" not in df.columns or "rsi" not in df.columns:
            return df

        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df))
        position = 0
        entry_bar = 0
        current_date = None
        day_skipped = False
        bars_since_exit = 999

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                bars_since_exit = 999
                if has_atr:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            # Need at least 30 min of data for VWAP to be meaningful
            if row["minute_of_day"] < 10 * 60:
                signals[i] = 0
                continue

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            vwap_dev = row.get("vwap_dev", 0)
            rsi = row.get("rsi", 50)
            if pd.isna(vwap_dev) or pd.isna(rsi):
                signals[i] = position
                continue

            if position == 0:
                bars_since_exit += 1

                if bars_since_exit < self.cooldown_bars:
                    signals[i] = 0
                    continue

                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                # Extreme low: VWAP dev very negative + RSI oversold -> go long
                if vwap_dev < -self.entry_zscore and rsi < self.rsi_oversold:
                    position = 1
                    entry_bar = i

                # Extreme high: VWAP dev very positive + RSI overbought -> go short
                elif vwap_dev > self.entry_zscore and rsi > self.rsi_overbought:
                    position = -1
                    entry_bar = i

            elif position == 1:
                # Exit long: reverted to VWAP or stale
                if vwap_dev >= -self.exit_zscore:
                    position = 0
                    bars_since_exit = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0
                    bars_since_exit = 0

            elif position == -1:
                # Exit short: reverted to VWAP or stale
                if vwap_dev <= self.exit_zscore:
                    position = 0
                    bars_since_exit = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0
                    bars_since_exit = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: VWAP Trend Following
# ═══════════════════════════════════════════════════════════════════════════════

class VWAPTrend(Strategy):
    """Follow intraday trend using VWAP as dynamic anchor.

    Structural difference from ORB: Uses continuously-updating VWAP as trend
    reference (not fixed opening range). Requires sustained position above/below
    VWAP with volume confirmation. Can trigger at any time.

    Entry: Price has been above VWAP for N consecutive bars with above-average
           volume -> go long (and vice versa for short).
    Exit: Price crosses back through VWAP, or stale/EOD.
    """
    name = "vwap_trend"

    def __init__(self, confirm_bars=10, min_rel_vol=1.0,
                 min_vwap_dist_pct=0.05, stale_bars=120,
                 last_entry_minute=900, min_atr_pctl=0):
        self.confirm_bars = confirm_bars
        self.min_rel_vol = min_rel_vol
        self.min_vwap_dist_pct = min_vwap_dist_pct  # Min % distance from VWAP
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl

    def get_params(self):
        return {
            "confirm": self.confirm_bars, "min_vol": self.min_rel_vol,
            "min_dist": self.min_vwap_dist_pct, "stale": self.stale_bars,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "intraday_vwap" not in df.columns:
            return df

        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        has_vol = self.min_rel_vol > 0 and "rel_volume" in df.columns
        signals = np.zeros(len(df))
        position = 0
        entry_bar = 0
        bars_above = 0
        bars_below = 0
        current_date = None
        day_skipped = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                bars_above = 0
                bars_below = 0
                if has_atr:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            # Wait for VWAP to stabilize
            if row["minute_of_day"] < 10 * 60:
                signals[i] = 0
                continue

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            vwap = row.get("intraday_vwap", 0)
            if pd.isna(vwap) or vwap <= 0:
                signals[i] = position
                continue

            price = row["close"]
            dist_pct = (price - vwap) / vwap * 100

            # Track consecutive bars above/below VWAP
            if price > vwap:
                bars_above += 1
                bars_below = 0
            elif price < vwap:
                bars_below += 1
                bars_above = 0
            else:
                bars_above = 0
                bars_below = 0

            if position == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                vol_ok = True
                if has_vol:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_rel_vol:
                        vol_ok = False

                # Long: sustained above VWAP with volume
                if (bars_above >= self.confirm_bars
                        and dist_pct >= self.min_vwap_dist_pct and vol_ok):
                    position = 1
                    entry_bar = i

                # Short: sustained below VWAP with volume
                elif (bars_below >= self.confirm_bars
                      and dist_pct <= -self.min_vwap_dist_pct and vol_ok):
                    position = -1
                    entry_bar = i

            elif position == 1:
                # Exit: price crosses below VWAP
                if price < vwap:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            elif position == -1:
                # Exit: price crosses above VWAP
                if price > vwap:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Volatility Compression Breakout (Bollinger Squeeze)
# ═══════════════════════════════════════════════════════════════════════════════

class VolCompression(Strategy):
    """Trade breakouts from intraday volatility compression.

    Structural difference from ORB: Uses dynamic Bollinger bandwidth compression
    (not a fixed time-based range). Can trigger at any time of day when vol
    compresses then expands. Measures relative bandwidth, not absolute levels.

    Entry: Bollinger bandwidth drops below percentile threshold (squeeze), then
           price breaks above upper band (long) or below lower band (short).
    Exit: Price returns inside bands, or stale/EOD.
    """
    name = "vol_compression"

    def __init__(self, bb_period=20, bb_std=2.0, squeeze_lookback=60,
                 squeeze_pctl=20, stale_bars=60,
                 last_entry_minute=900, min_atr_pctl=0):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback  # Bars to rank bandwidth
        self.squeeze_pctl = squeeze_pctl  # Bandwidth must be below this percentile
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl

    def get_params(self):
        return {
            "bb_per": self.bb_period, "bb_std": self.bb_std,
            "sq_look": self.squeeze_lookback, "sq_pctl": self.squeeze_pctl,
            "stale": self.stale_bars,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
            return df

        # Compute bandwidth
        bb_mid = df["bb_mid"]
        bandwidth = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)
        # Rolling percentile of bandwidth
        bw_pctl = bandwidth.rolling(self.squeeze_lookback, min_periods=20).rank(pct=True) * 100

        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df))
        position = 0
        entry_bar = 0
        was_squeezed = False
        current_date = None
        day_skipped = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                was_squeezed = False
                if has_atr:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            if row["minute_of_day"] < 10 * 60:
                signals[i] = 0
                continue

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            bw_p = bw_pctl.iloc[i] if i < len(bw_pctl) else 50
            if pd.isna(bw_p):
                signals[i] = position
                continue

            bb_up = row.get("bb_upper", 0)
            bb_lo = row.get("bb_lower", 0)
            if pd.isna(bb_up) or pd.isna(bb_lo) or bb_up <= 0:
                signals[i] = position
                continue

            # Track squeeze state
            if bw_p <= self.squeeze_pctl:
                was_squeezed = True

            if position == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    was_squeezed = False  # Reset for next day
                    continue

                if was_squeezed and bw_p > self.squeeze_pctl:
                    # Squeeze released - enter on breakout direction
                    if row["close"] > bb_up:
                        position = 1
                        entry_bar = i
                        was_squeezed = False
                    elif row["close"] < bb_lo:
                        position = -1
                        entry_bar = i
                        was_squeezed = False

            elif position == 1:
                # Exit: price falls back below mid band or stale
                mid = row.get("bb_mid", 0)
                if pd.notna(mid) and row["close"] < mid:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            elif position == -1:
                mid = row.get("bb_mid", 0)
                if pd.notna(mid) and row["close"] > mid:
                    position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: Intraday Momentum Score (Multi-Factor Composite)
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumScore(Strategy):
    """Multi-factor momentum scoring system.

    Structural difference from ORB: Uses composite scoring from multiple
    independent factors rather than a single breakout event. No fixed range,
    no time anchor. Continuously recalculates. More robust to single-factor
    noise.

    Factors scored [-1, 0, +1] each:
    - EMA alignment (fast vs slow)
    - Price vs VWAP
    - RSI direction (above/below midline)
    - Volume confirmation

    Entry: Composite score reaches threshold (e.g., 3 of 4 bullish).
    Exit: Score drops to 0 or flips, or stale/EOD.
    """
    name = "momentum_score"

    def __init__(self, entry_threshold=3, exit_threshold=1,
                 rsi_bull=55, rsi_bear=45, min_rel_vol=1.0,
                 stale_bars=120, last_entry_minute=900,
                 min_atr_pctl=0, cooldown_bars=10):
        self.entry_threshold = entry_threshold  # Min factors aligned to enter
        self.exit_threshold = exit_threshold    # Exit when score drops to this
        self.rsi_bull = rsi_bull
        self.rsi_bear = rsi_bear
        self.min_rel_vol = min_rel_vol
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl
        self.cooldown_bars = cooldown_bars

    def get_params(self):
        return {
            "entry_thr": self.entry_threshold, "exit_thr": self.exit_threshold,
            "rsi_b": self.rsi_bull, "rsi_br": self.rsi_bear,
            "stale": self.stale_bars, "cooldown": self.cooldown_bars,
        }

    def _compute_score(self, row):
        """Compute composite momentum score from multiple factors."""
        score = 0

        # Factor 1: EMA alignment (fast > slow = bullish)
        ema8 = row.get("ema_8", 0)
        ema21 = row.get("ema_21", 0)
        if pd.notna(ema8) and pd.notna(ema21) and ema21 > 0:
            if ema8 > ema21:
                score += 1
            elif ema8 < ema21:
                score -= 1

        # Factor 2: Price vs VWAP
        vwap = row.get("intraday_vwap", 0)
        if pd.notna(vwap) and vwap > 0:
            if row["close"] > vwap:
                score += 1
            elif row["close"] < vwap:
                score -= 1

        # Factor 3: RSI direction
        rsi = row.get("rsi", 50)
        if pd.notna(rsi):
            if rsi > self.rsi_bull:
                score += 1
            elif rsi < self.rsi_bear:
                score -= 1

        # Factor 4: Volume confirmation
        rv = row.get("rel_volume", 1.0)
        if pd.notna(rv) and rv >= self.min_rel_vol:
            # Volume confirms the direction (adds +1 if bullish, -1 if bearish)
            if score > 0:
                score += 1
            elif score < 0:
                score -= 1

        return score

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df))
        position = 0
        entry_bar = 0
        current_date = None
        day_skipped = False
        bars_since_exit = 999

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                bars_since_exit = 999
                if has_atr:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            if row["minute_of_day"] < 10 * 60:
                signals[i] = 0
                continue

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            score = self._compute_score(row)

            if position == 0:
                bars_since_exit += 1

                if bars_since_exit < self.cooldown_bars:
                    signals[i] = 0
                    continue

                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                if score >= self.entry_threshold:
                    position = 1
                    entry_bar = i
                elif score <= -self.entry_threshold:
                    position = -1
                    entry_bar = i

            elif position == 1:
                # Exit: score drops to exit threshold or below, or stale
                if score <= self.exit_threshold:
                    position = 0
                    bars_since_exit = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0
                    bars_since_exit = 0

            elif position == -1:
                if score >= -self.exit_threshold:
                    position = 0
                    bars_since_exit = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars:
                    position = 0
                    bars_since_exit = 0

            signals[i] = position

        df["signal"] = signals.astype(int)
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: Gap Continuation
# ═══════════════════════════════════════════════════════════════════════════════

class GapContinuation(Strategy):
    """Trade in the direction of gaps that hold after open.

    Structural difference from ORB: Trades overnight information events, not
    intraday range breakouts. The edge comes from gaps that reflect genuine
    news/sentiment shifts. Opposite of Gap Fade.

    Entry: Large gap that does NOT fill in the first N minutes. Enter in gap
           direction after confirmation period.
    Exit: Target (gap extension), trailing stop, stale, or EOD.
    """
    name = "gap_continuation"

    def __init__(self, min_gap_pct=0.3, max_gap_pct=2.5,
                 confirm_minutes=15, target_pct=0.3,
                 stop_pct=0.15, stale_bars=180,
                 last_entry_minute=720, min_atr_pctl=0):
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.confirm_minutes = confirm_minutes  # Gap must hold this long
        self.target_pct = target_pct  # Target = entry + this % move
        self.stop_pct = stop_pct  # Stop = entry - this % move
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl

    def get_params(self):
        return {
            "min_gap": self.min_gap_pct, "max_gap": self.max_gap_pct,
            "confirm": self.confirm_minutes, "target": self.target_pct,
            "stop": self.stop_pct, "stale": self.stale_bars,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        if "gap_pct" not in df.columns or "prev_close" not in df.columns:
            return df

        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        confirm_end = 9 * 60 + 30 + self.confirm_minutes
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
        gap_held = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]

            if date != current_date:
                position = 0
                current_date = date
                day_skipped = False
                gap_held = False
                day_gap = row.get("gap_pct", 0)
                prev_close_val = row.get("prev_close", 0)

                if pd.isna(day_gap) or pd.isna(prev_close_val) or prev_close_val <= 0:
                    day_skipped = True
                elif abs(day_gap) < self.min_gap_pct or abs(day_gap) > self.max_gap_pct:
                    day_skipped = True

                if has_atr and not day_skipped:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl:
                        day_skipped = True

            if day_skipped:
                signals[i] = 0
                continue

            if row["minute_of_day"] > 15 * 60 + 30:
                position = 0
                signals[i] = 0
                continue

            # During confirmation period: check if gap is holding
            if row["minute_of_day"] < confirm_end:
                # Check if gap has been filled during confirmation
                if day_gap > 0 and row["low"] <= prev_close_val:
                    day_skipped = True  # Gap filled, skip day
                elif day_gap < 0 and row["high"] >= prev_close_val:
                    day_skipped = True
                signals[i] = 0
                continue

            # Right at confirmation time: mark gap as held
            if not gap_held and row["minute_of_day"] >= confirm_end:
                gap_held = True

            if position == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0
                    continue

                if gap_held:
                    if day_gap > 0:
                        # Gap up held -> go long (continuation)
                        position = 1
                        entry_price = row["close"]
                        entry_bar = i
                        target = entry_price * (1 + self.target_pct / 100)
                        stop = entry_price * (1 - self.stop_pct / 100)
                        gap_held = False  # Only enter once
                    elif day_gap < 0:
                        # Gap down held -> go short (continuation)
                        position = -1
                        entry_price = row["close"]
                        entry_bar = i
                        target = entry_price * (1 - self.target_pct / 100)
                        stop = entry_price * (1 + self.stop_pct / 100)
                        gap_held = False

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


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def eval_walkforward(sym, df, strat, label=""):
    """Run walk-forward and return aggregated metrics."""
    df = strat.generate_signals(df)
    try:
        wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)
    except ValueError:
        return None

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

    # Sortino
    downside = combined[combined < 0]
    downside_std = downside.std() if len(downside) > 1 else combined.std()
    sortino = (combined.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    window_sharpes = [r.sharpe_ratio for r in wf.oos_results]
    pos_windows = sum(1 for s in window_sharpes if s > 0)

    return {
        "sym": sym, "label": label, "sharpe": sharpe, "sortino": sortino,
        "return": total_ret, "max_dd": max_dd, "trades": wf.total_trades,
        "win_rate": wf.win_rate, "pf": wf.profit_factor,
        "windows_pos": f"{pos_windows}/{len(window_sharpes)}",
        "daily_returns": combined,
    }


def eval_period(sym, df, strat, start_date, end_date, label=""):
    """Run backtest on a specific date range and return metrics + daily returns."""
    dates = sorted(df["date"].unique())
    mask = df["date"].apply(lambda d: start_date.date() <= d <= end_date.date()
                            if hasattr(start_date, 'date') else True)
    period_df = df[mask].copy()
    if len(period_df) < 100:
        return None

    period_df = strat.generate_signals(period_df)
    result = run_backtest(period_df, strat, sym)

    dr = result.daily_returns
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return None

    sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
    total_ret = result.total_return
    cum = (1 + dr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    downside = dr[dr < 0]
    downside_std = downside.std() if len(downside) > 1 else dr.std()
    sortino = (dr.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    return {
        "sym": sym, "label": label, "sharpe": sharpe, "sortino": sortino,
        "return": total_ret, "max_dd": max_dd, "trades": result.num_trades,
        "win_rate": result.win_rate, "pf": result.profit_factor,
        "daily_returns": dr,
    }


def compute_alpha_beta(strat_returns, bench_returns):
    """Compute alpha and beta via OLS regression."""
    aligned = pd.DataFrame({"strat": strat_returns, "bench": bench_returns}).dropna()
    if len(aligned) < 10:
        return 0, 0
    x = aligned["bench"]
    y = aligned["strat"]
    beta = x.cov(y) / x.var() if x.var() > 0 else 0
    alpha = (y.mean() - beta * x.mean()) * 252
    return alpha, beta


def print_result(r, prefix="  "):
    """Print a single result row."""
    if r is None:
        print(f"{prefix}-> no valid data", flush=True)
        return
    win = r.get("windows_pos", "")
    win_str = f"  Win={win}" if win else ""
    print(f"{prefix}S={r['sharpe']:>6.2f}  Sort={r['sortino']:>5.2f}"
          f"  Ret={r['return']:>+7.2%}  DD={r['max_dd']:>6.2%}"
          f"  T={r['trades']:>4}  WR={r['win_rate']:.1%}"
          f"  PF={r['pf']:.2f}{win_str}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 100, flush=True)
print("EXPERIMENT 14: BROAD NEW EDGE SCREEN", flush=True)
print("5 strategy families x 7 instruments, dev period walk-forward", flush=True)
print("=" * 100, flush=True)
print("\nLoading data...", flush=True)

data = {}
for sym in UNIVERSE:
    try:
        df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
        df = prepare_features(df)
        data[sym] = df
        n_days = df["date"].nunique()
        print(f"  {sym}: {len(df)} bars, {n_days} days", flush=True)
    except Exception as e:
        print(f"  {sym}: FAILED - {e}", flush=True)

# Load SPY benchmark returns for alpha/beta
spy_bench = data["SPY"].groupby("date")["close"].last().pct_change().dropna()

# Dev period filter
def get_dev(sym):
    df = data[sym]
    return df.loc[df["dt"] <= DEV_END].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE: Production ORB profiles
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100, flush=True)
print("BASELINE: Production ORB (SPY+QQQ) on dev period", flush=True)
print("=" * 100, flush=True)

from trading.strategies.orb import ORBBreakout

baseline_daily = {}
for sym in ["SPY", "QQQ"]:
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(SYMBOL_PROFILES.get(sym, {}))
    strat = ORBBreakout(**params)
    r = eval_walkforward(sym, get_dev(sym), strat, f"ORB-{sym}")
    if r:
        print(f"  {sym}: ", end="", flush=True)
        print_result(r, "")
        baseline_daily[sym] = r["daily_returns"]

# Portfolio baseline
if len(baseline_daily) >= 2:
    port_base = pd.DataFrame(baseline_daily).fillna(0).mean(axis=1)
    base_sharpe = (port_base.mean() / port_base.std()) * np.sqrt(252) if port_base.std() > 0 else 0
    base_ret = (1 + port_base).prod() - 1
    print(f"\n  Portfolio SPY+QQQ ORB: Sharpe={base_sharpe:.2f}  Return={base_ret:+.2%}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN ALL STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

all_results = []  # (strategy_name, sym, variant_label, result_dict)

# ─── Strategy 1: Mean Reversion Extreme ───

print("\n" + "=" * 100, flush=True)
print("STRATEGY 1: MEAN REVERSION ON EXTREME MOVES (VWAP Z-score fade)", flush=True)
print("=" * 100, flush=True)

mr_variants = [
    ("z2.0 rsi30/70",        {"entry_zscore": 2.0, "exit_zscore": 0.5, "rsi_oversold": 30, "rsi_overbought": 70}),
    ("z2.0 rsi25/75",        {"entry_zscore": 2.0, "exit_zscore": 0.5, "rsi_oversold": 25, "rsi_overbought": 75}),
    ("z1.5 rsi30/70",        {"entry_zscore": 1.5, "exit_zscore": 0.3, "rsi_oversold": 30, "rsi_overbought": 70}),
    ("z2.5 rsi30/70",        {"entry_zscore": 2.5, "exit_zscore": 0.5, "rsi_oversold": 30, "rsi_overbought": 70}),
    ("z2.0 stale60",         {"entry_zscore": 2.0, "exit_zscore": 0.5, "stale_bars": 60}),
    ("z2.0 stale120",        {"entry_zscore": 2.0, "exit_zscore": 0.5, "stale_bars": 120}),
    ("z2.0 noRSI",           {"entry_zscore": 2.0, "exit_zscore": 0.5, "rsi_oversold": 0, "rsi_overbought": 100}),
    ("z2.0 atr25",           {"entry_zscore": 2.0, "exit_zscore": 0.5, "min_atr_pctl": 25}),
]

for sym in UNIVERSE:
    if sym not in data:
        continue
    print(f"\n  {sym}:", flush=True)
    df_dev = get_dev(sym)
    for label, kwargs in mr_variants:
        strat = MeanRevExtreme(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<25}", end="", flush=True)
            print_result(r, "")
            all_results.append(("MeanRev", sym, label, r))
        else:
            print(f"    {label:<25} -> no valid data", flush=True)


# ─── Strategy 2: VWAP Trend ───

print("\n" + "=" * 100, flush=True)
print("STRATEGY 2: VWAP TREND FOLLOWING", flush=True)
print("=" * 100, flush=True)

vt_variants = [
    ("conf10 vol1.0 dist0.05", {"confirm_bars": 10, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}),
    ("conf10 vol0.0 dist0.05", {"confirm_bars": 10, "min_rel_vol": 0.0, "min_vwap_dist_pct": 0.05}),
    ("conf5 vol1.0 dist0.05",  {"confirm_bars": 5, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}),
    ("conf15 vol1.0 dist0.05", {"confirm_bars": 15, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}),
    ("conf10 vol1.0 dist0.10", {"confirm_bars": 10, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.10}),
    ("conf10 vol1.2 dist0.05", {"confirm_bars": 10, "min_rel_vol": 1.2, "min_vwap_dist_pct": 0.05}),
    ("conf10 stale60",         {"confirm_bars": 10, "min_rel_vol": 1.0, "stale_bars": 60}),
    ("conf10 atr25",           {"confirm_bars": 10, "min_rel_vol": 1.0, "min_atr_pctl": 25}),
]

for sym in UNIVERSE:
    if sym not in data:
        continue
    print(f"\n  {sym}:", flush=True)
    df_dev = get_dev(sym)
    for label, kwargs in vt_variants:
        strat = VWAPTrend(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<25}", end="", flush=True)
            print_result(r, "")
            all_results.append(("VWAPTrend", sym, label, r))
        else:
            print(f"    {label:<25} -> no valid data", flush=True)


# ─── Strategy 3: Volatility Compression ───

print("\n" + "=" * 100, flush=True)
print("STRATEGY 3: VOLATILITY COMPRESSION BREAKOUT (Bollinger Squeeze)", flush=True)
print("=" * 100, flush=True)

vc_variants = [
    ("sq20 stale60",          {"squeeze_pctl": 20, "stale_bars": 60}),
    ("sq20 stale90",          {"squeeze_pctl": 20, "stale_bars": 90}),
    ("sq15 stale60",          {"squeeze_pctl": 15, "stale_bars": 60}),
    ("sq25 stale60",          {"squeeze_pctl": 25, "stale_bars": 60}),
    ("sq20 stale120",         {"squeeze_pctl": 20, "stale_bars": 120}),
    ("sq20 bb_per30",         {"squeeze_pctl": 20, "stale_bars": 60, "bb_period": 30}),
    ("sq20 bb_std2.5",        {"squeeze_pctl": 20, "stale_bars": 60, "bb_std": 2.5}),
    ("sq20 atr25",            {"squeeze_pctl": 20, "stale_bars": 60, "min_atr_pctl": 25}),
]

for sym in UNIVERSE:
    if sym not in data:
        continue
    print(f"\n  {sym}:", flush=True)
    df_dev = get_dev(sym)
    for label, kwargs in vc_variants:
        strat = VolCompression(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<25}", end="", flush=True)
            print_result(r, "")
            all_results.append(("VolComp", sym, label, r))
        else:
            print(f"    {label:<25} -> no valid data", flush=True)


# ─── Strategy 4: Momentum Score ───

print("\n" + "=" * 100, flush=True)
print("STRATEGY 4: INTRADAY MOMENTUM SCORE (multi-factor composite)", flush=True)
print("=" * 100, flush=True)

ms_variants = [
    ("thr3 exit1",            {"entry_threshold": 3, "exit_threshold": 1}),
    ("thr3 exit0",            {"entry_threshold": 3, "exit_threshold": 0}),
    ("thr4 exit1",            {"entry_threshold": 4, "exit_threshold": 1}),
    ("thr3 exit1 stale60",    {"entry_threshold": 3, "exit_threshold": 1, "stale_bars": 60}),
    ("thr3 exit1 stale180",   {"entry_threshold": 3, "exit_threshold": 1, "stale_bars": 180}),
    ("thr3 exit1 cd20",       {"entry_threshold": 3, "exit_threshold": 1, "cooldown_bars": 20}),
    ("thr3 exit1 vol1.2",     {"entry_threshold": 3, "exit_threshold": 1, "min_rel_vol": 1.2}),
    ("thr3 exit1 atr25",      {"entry_threshold": 3, "exit_threshold": 1, "min_atr_pctl": 25}),
]

for sym in UNIVERSE:
    if sym not in data:
        continue
    print(f"\n  {sym}:", flush=True)
    df_dev = get_dev(sym)
    for label, kwargs in ms_variants:
        strat = MomentumScore(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<25}", end="", flush=True)
            print_result(r, "")
            all_results.append(("MomScore", sym, label, r))
        else:
            print(f"    {label:<25} -> no valid data", flush=True)


# ─── Strategy 5: Gap Continuation ───

print("\n" + "=" * 100, flush=True)
print("STRATEGY 5: GAP CONTINUATION (overnight information event)", flush=True)
print("=" * 100, flush=True)

gc_variants = [
    ("gap0.3 tgt0.3 stp0.15",  {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15}),
    ("gap0.3 tgt0.2 stp0.10",  {"min_gap_pct": 0.3, "target_pct": 0.2, "stop_pct": 0.10}),
    ("gap0.3 tgt0.4 stp0.20",  {"min_gap_pct": 0.3, "target_pct": 0.4, "stop_pct": 0.20}),
    ("gap0.5 tgt0.3 stp0.15",  {"min_gap_pct": 0.5, "target_pct": 0.3, "stop_pct": 0.15}),
    ("gap0.3 confirm30",       {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "confirm_minutes": 30}),
    ("gap0.3 stale90",         {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "stale_bars": 90}),
    ("gap0.3 atr25",           {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "min_atr_pctl": 25}),
    ("gap0.2 tgt0.3 stp0.15",  {"min_gap_pct": 0.2, "target_pct": 0.3, "stop_pct": 0.15}),
]

for sym in UNIVERSE:
    if sym not in data:
        continue
    print(f"\n  {sym}:", flush=True)
    df_dev = get_dev(sym)
    for label, kwargs in gc_variants:
        strat = GapContinuation(**kwargs)
        r = eval_walkforward(sym, df_dev, strat, label)
        if r:
            print(f"    {label:<25}", end="", flush=True)
            print_result(r, "")
            all_results.append(("GapCont", sym, label, r))
        else:
            print(f"    {label:<25} -> no valid data", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY: TOP CANDIDATES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100, flush=True)
print("TOP CANDIDATES (Sharpe > 0.5 on dev walk-forward)", flush=True)
print("=" * 100, flush=True)

# Sort by Sharpe
candidates = [(s, sym, lab, r) for s, sym, lab, r in all_results if r["sharpe"] > 0.5]
candidates.sort(key=lambda x: -x[3]["sharpe"])

if not candidates:
    print("\n  NO candidates with Sharpe > 0.5 found.", flush=True)
else:
    print(f"\n  {'Strategy':<15} {'Sym':<5} {'Variant':<25} {'Sharpe':>7} {'Sortino':>8}"
          f" {'Return':>8} {'MaxDD':>7} {'Trades':>6} {'WR':>5} {'PF':>5}", flush=True)
    print(f"  {'-'*14} {'-'*4} {'-'*24} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*5} {'-'*5}", flush=True)
    for strat_name, sym, label, r in candidates:
        alpha, beta = compute_alpha_beta(r["daily_returns"], spy_bench)
        print(f"  {strat_name:<15} {sym:<5} {label:<25} {r['sharpe']:>7.2f} {r['sortino']:>8.2f}"
              f" {r['return']:>+8.2%} {r['max_dd']:>7.2%} {r['trades']:>6}"
              f" {r['win_rate']:>5.1%} {r['pf']:>5.2f}"
              f"  a={alpha:+.1%} b={beta:.3f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO COMBINATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100, flush=True)
print("PORTFOLIO COMBINATION (adding each candidate to SPY+QQQ ORB baseline)", flush=True)
print("=" * 100, flush=True)

if len(baseline_daily) >= 2 and candidates:
    port_base = pd.DataFrame(baseline_daily).fillna(0).mean(axis=1)
    base_sharpe = (port_base.mean() / port_base.std()) * np.sqrt(252)

    print(f"\n  Baseline SPY+QQQ ORB: Sharpe={base_sharpe:.2f}", flush=True)

    combo_results = []
    for strat_name, sym, label, r in candidates:
        combined = dict(baseline_daily)
        key = f"{strat_name}-{sym}-{label}"
        combined[key] = r["daily_returns"]
        port_df = pd.DataFrame(combined).fillna(0)
        port_ret = port_df.mean(axis=1)

        if len(port_ret) < 10 or port_ret.std() == 0:
            continue

        new_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252)
        delta = new_sharpe - base_sharpe
        corr = r["daily_returns"].corr(port_base)

        combo_results.append((key, new_sharpe, delta, corr, r["sharpe"]))
        print(f"  + {key:<55} port={new_sharpe:.2f} (d={delta:+.2f})"
              f"  corr={corr:.2f}  solo={r['sharpe']:.2f}", flush=True)

    # Best combo
    if combo_results:
        combo_results.sort(key=lambda x: -x[1])
        best = combo_results[0]
        print(f"\n  BEST COMBO: {best[0]} -> portfolio Sharpe {best[1]:.2f} (delta {best[2]:+.2f})", flush=True)

print("\n" + "=" * 100, flush=True)
print("EXPERIMENT 14 COMPLETE", flush=True)
print("=" * 100, flush=True)
