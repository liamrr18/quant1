#!/usr/bin/env python3
"""Experiment 15: Wave 2 strategy search — 4 more genuinely different families.

1. Prior-Day Level Reversal — fade at yesterday's high/low/close
2. Opening Drive Continuation — first 5-min move predicts the day
3. Pairs Spread Mean Reversion — trade SPY/QQQ relative value spread
4. Intraday Range Expansion — narrow-range bar clusters predict breakouts

Uses the same parallel walk-forward pipeline as Experiment 14.
"""

import sys, os, io, json, time, logging, warnings, tempfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pytz
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)

from trading.strategies.base import Strategy

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

UNIVERSE = ["SPY", "QQQ", "GLD", "XLE", "XLK", "SMH", "TLT"]
DATA_CACHE_DIR = Path(tempfile.gettempdir()) / "spy_trader_screen"
RESULTS_PATH = Path("research/wave2_results.json")

t0 = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Prior-Day Level Reversal
# ═══════════════════════════════════════════════════════════════════════════════

class PriorDayReversal(Strategy):
    """Fade price at previous day's high, low, or close.

    Structural difference from ORB: Uses PREVIOUS session levels as
    support/resistance, not current-day opening range. Exploits the
    tendency of price to bounce off prior-session key levels.

    Entry: Price touches prior high/low and RSI shows exhaustion -> fade.
    Exit: Target (partial retracement), stop (breakout beyond level), or stale/EOD.
    """
    name = "prior_day_reversal"

    def __init__(self, touch_pct=0.05, target_pct=0.15, stop_pct=0.10,
                 stale_bars=90, use_rsi=True, rsi_extreme=35,
                 last_entry_minute=900, min_atr_pctl=0):
        self.touch_pct = touch_pct  # % distance from level to count as "touch"
        self.target_pct = target_pct
        self.stop_pct = stop_pct
        self.stale_bars = stale_bars
        self.use_rsi = use_rsi
        self.rsi_extreme = rsi_extreme  # RSI below this for longs, above (100-this) for shorts
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl

    def get_params(self):
        return {"touch": self.touch_pct, "tgt": self.target_pct,
                "stp": self.stop_pct, "stale": self.stale_bars, "rsi": self.use_rsi}

    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        # Need prev day high/low
        daily_high = df.groupby("date")["high"].transform("max")
        daily_low = df.groupby("date")["low"].transform("min")
        dates = sorted(df["date"].unique())
        prev_high_map = {}; prev_low_map = {}; prev_close_map = {}
        for idx, d in enumerate(dates):
            if idx > 0:
                prev_d = dates[idx-1]
                mask = df["date"] == prev_d
                prev_high_map[d] = df.loc[mask, "high"].max()
                prev_low_map[d] = df.loc[mask, "low"].min()
                prev_close_map[d] = df.loc[mask, "close"].iloc[-1]

        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        has_rsi = self.use_rsi and "rsi" in df.columns
        signals = np.zeros(len(df)); pos = 0; ep = 0.0; ebar = 0
        tgt = 0.0; stp = 0.0; cdate = None; skip = False
        ph = 0.0; pl = 0.0

        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False
                ph = prev_high_map.get(d, 0)
                pl = prev_low_map.get(d, 0)
                if ph <= 0 or pl <= 0: skip = True
                if has_atr and not skip:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True

            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue

            price = row["close"]

            if pos == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; continue

                # Near prior high -> short (reversal)
                if ph > 0 and abs(price - ph) / ph * 100 <= self.touch_pct:
                    rsi_ok = True
                    if has_rsi:
                        rsi = row.get("rsi", 50)
                        if pd.isna(rsi) or rsi < (100 - self.rsi_extreme): rsi_ok = False
                    if rsi_ok and price >= ph * (1 - self.touch_pct/100):
                        pos = -1; ep = price; ebar = i
                        tgt = ep * (1 - self.target_pct / 100)
                        stp = ep * (1 + self.stop_pct / 100)

                # Near prior low -> long (reversal)
                elif pl > 0 and abs(price - pl) / pl * 100 <= self.touch_pct:
                    rsi_ok = True
                    if has_rsi:
                        rsi = row.get("rsi", 50)
                        if pd.isna(rsi) or rsi > self.rsi_extreme: rsi_ok = False
                    if rsi_ok and price <= pl * (1 + self.touch_pct/100):
                        pos = 1; ep = price; ebar = i
                        tgt = ep * (1 + self.target_pct / 100)
                        stp = ep * (1 - self.stop_pct / 100)

            elif pos == 1:
                if price >= tgt or price <= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if price <= tgt or price >= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0

            signals[i] = pos
        df["signal"] = signals.astype(int); return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: Opening Drive Continuation
# ═══════════════════════════════════════════════════════════════════════════════

class OpeningDrive(Strategy):
    """Trade in the direction of the first N-minute move after open.

    Structural difference from ORB: Trades the DIRECTION of the first move,
    not a breakout from a range. Enters immediately after the drive window,
    not waiting for a breakout. Uses gap + initial momentum as signal.

    Entry: If price moved > threshold in first N minutes from open, enter
           in that direction. Requires gap alignment.
    Exit: Fixed target/stop based on drive size, or stale/EOD.
    """
    name = "opening_drive"

    def __init__(self, drive_minutes=5, min_drive_pct=0.10,
                 target_multiple=2.0, stop_multiple=1.0,
                 require_gap_align=False, min_gap_pct=0.0,
                 stale_bars=120, last_entry_minute=720, min_atr_pctl=0):
        self.drive_minutes = drive_minutes
        self.min_drive_pct = min_drive_pct
        self.target_multiple = target_multiple
        self.stop_multiple = stop_multiple
        self.require_gap_align = require_gap_align
        self.min_gap_pct = min_gap_pct
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl

    def get_params(self):
        return {"drive_min": self.drive_minutes, "min_drive": self.min_drive_pct,
                "tgt_mult": self.target_multiple, "stp_mult": self.stop_multiple,
                "stale": self.stale_bars}

    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        drive_end = 9 * 60 + 30 + self.drive_minutes
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns

        signals = np.zeros(len(df)); pos = 0; ep = 0.0; ebar = 0
        tgt = 0.0; stp = 0.0; cdate = None; skip = False
        day_open = 0.0; drive_dir = 0; drive_size = 0.0; entered = False

        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; drive_dir = 0
                drive_size = 0.0; entered = False
                day_open = row["open"]
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
                # Gap alignment check
                if self.require_gap_align and "gap_pct" in df.columns:
                    gp = row.get("gap_pct", 0)
                    if pd.isna(gp) or abs(gp) < self.min_gap_pct: skip = True

            if skip: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue

            # During drive window: track the move
            if row["minute_of_day"] <= drive_end:
                if day_open > 0:
                    move_pct = (row["close"] - day_open) / day_open * 100
                    if abs(move_pct) > abs(drive_size):
                        drive_size = move_pct
                signals[i] = 0; continue

            # Just after drive window: determine direction
            if drive_dir == 0 and not entered:
                if drive_size >= self.min_drive_pct:
                    drive_dir = 1  # Bullish drive
                elif drive_size <= -self.min_drive_pct:
                    drive_dir = -1  # Bearish drive
                else:
                    skip = True  # Drive too small
                    signals[i] = 0; continue

                # Gap alignment
                if self.require_gap_align and "gap_pct" in df.columns:
                    gp = row.get("gap_pct", 0)
                    if pd.notna(gp):
                        if (drive_dir > 0 and gp < 0) or (drive_dir < 0 and gp > 0):
                            skip = True; signals[i] = 0; continue

            if pos == 0 and not entered and drive_dir != 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; continue

                entered = True
                pos = drive_dir
                ep = row["close"]; ebar = i
                abs_drive = abs(drive_size / 100 * ep)
                if pos == 1:
                    tgt = ep + abs_drive * self.target_multiple
                    stp = ep - abs_drive * self.stop_multiple
                else:
                    tgt = ep - abs_drive * self.target_multiple
                    stp = ep + abs_drive * self.stop_multiple

            elif pos == 1:
                if row["close"] >= tgt or row["close"] <= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if row["close"] <= tgt or row["close"] >= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0

            signals[i] = pos
        df["signal"] = signals.astype(int); return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Pairs Spread Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════════

class PairsSpread(Strategy):
    """Trade mean reversion of the intraday spread between two correlated ETFs.

    Structural difference: Cross-sectional, not single-instrument. Trades
    relative value, not absolute direction. Market-neutral by construction.

    NOTE: This strategy requires a SECOND instrument's data to be merged.
    The 'pair_col' column must be added before calling generate_signals.
    Signal: long = spread too low (buy this, hedge would short pair),
            short = spread too high (sell this, hedge would buy pair).
    """
    name = "pairs_spread"

    def __init__(self, lookback=60, entry_zscore=2.0, exit_zscore=0.5,
                 stale_bars=90, last_entry_minute=900):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute

    def get_params(self):
        return {"look": self.lookback, "z_in": self.entry_zscore,
                "z_out": self.exit_zscore, "stale": self.stale_bars}

    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "pair_close" not in df.columns: return df

        # Compute log spread
        spread = np.log(df["close"]) - np.log(df["pair_close"])
        spread_mean = spread.rolling(self.lookback, min_periods=20).mean()
        spread_std = spread.rolling(self.lookback, min_periods=20).std()
        zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

        signals = np.zeros(len(df)); pos = 0; ebar = 0
        cdate = None

        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d

            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue

            z = zscore.iloc[i] if i < len(zscore) else 0
            if pd.isna(z): signals[i] = pos; continue

            if pos == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; continue
                # Spread too low -> long this instrument (spread will revert up)
                if z < -self.entry_zscore: pos = 1; ebar = i
                # Spread too high -> short this instrument
                elif z > self.entry_zscore: pos = -1; ebar = i
            elif pos == 1:
                if z >= -self.exit_zscore: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if z <= self.exit_zscore: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0

            signals[i] = pos
        df["signal"] = signals.astype(int); return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: Intraday Range Expansion (NR pattern)
# ═══════════════════════════════════════════════════════════════════════════════

class RangeExpansion(Strategy):
    """Trade breakouts after narrow-range bar clusters.

    Structural difference from ORB: Pattern-based, not time-based. Uses
    bar-level range compression (NR = narrowest range in N bars) to
    identify coiled setups. Can trigger at any time of day.

    Entry: After N consecutive narrowing-range bars, enter on breakout
           of the narrow cluster's range.
    Exit: Target (multiple of cluster range), stop (opposite side), or stale/EOD.
    """
    name = "range_expansion"

    def __init__(self, nr_lookback=7, min_narrow_bars=3,
                 target_multiple=2.0, stop_multiple=0.5,
                 stale_bars=60, last_entry_minute=900, min_atr_pctl=0):
        self.nr_lookback = nr_lookback
        self.min_narrow_bars = min_narrow_bars
        self.target_multiple = target_multiple
        self.stop_multiple = stop_multiple
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl

    def get_params(self):
        return {"nr_look": self.nr_lookback, "narrow": self.min_narrow_bars,
                "tgt_mult": self.target_multiple, "stale": self.stale_bars}

    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0

        # Bar range
        bar_range = df["high"] - df["low"]
        # Rolling min range (NR detection)
        min_range = bar_range.rolling(self.nr_lookback, min_periods=3).min()
        # Is current bar the narrowest in lookback?
        is_nr = bar_range <= min_range * 1.05  # 5% tolerance

        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); pos = 0; ep = 0.0; ebar = 0
        tgt = 0.0; stp = 0.0; cdate = None; skip = False
        narrow_count = 0; cluster_high = 0.0; cluster_low = 999999.0

        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; narrow_count = 0
                cluster_high = 0.0; cluster_low = 999999.0
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True

            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue

            nr = is_nr.iloc[i] if i < len(is_nr) else False
            if pd.isna(nr): nr = False

            if pos == 0:
                if nr:
                    narrow_count += 1
                    cluster_high = max(cluster_high, row["high"])
                    cluster_low = min(cluster_low, row["low"])
                else:
                    # Check if we had enough narrow bars and now expanding
                    if narrow_count >= self.min_narrow_bars:
                        if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                            narrow_count = 0; signals[i] = 0; continue

                        cr = cluster_high - cluster_low
                        if cr > 0:
                            if row["close"] > cluster_high:
                                pos = 1; ep = row["close"]; ebar = i
                                tgt = ep + cr * self.target_multiple
                                stp = ep - cr * self.stop_multiple
                            elif row["close"] < cluster_low:
                                pos = -1; ep = row["close"]; ebar = i
                                tgt = ep - cr * self.target_multiple
                                stp = ep + cr * self.stop_multiple
                    narrow_count = 0; cluster_high = 0.0; cluster_low = 999999.0

            elif pos == 1:
                if row["close"] >= tgt or row["close"] <= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if row["close"] <= tgt or row["close"] >= stp: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0

            signals[i] = pos
        df["signal"] = signals.astype(int); return df


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT GRIDS
# ═══════════════════════════════════════════════════════════════════════════════

GRID = {
    "PDRev": {
        "cls": PriorDayReversal,
        "variants": [
            ("touch0.05_tgt0.15",  {"touch_pct": 0.05, "target_pct": 0.15, "stop_pct": 0.10}),
            ("touch0.10_tgt0.15",  {"touch_pct": 0.10, "target_pct": 0.15, "stop_pct": 0.10}),
            ("touch0.05_tgt0.20",  {"touch_pct": 0.05, "target_pct": 0.20, "stop_pct": 0.15}),
            ("touch0.05_tgt0.10",  {"touch_pct": 0.05, "target_pct": 0.10, "stop_pct": 0.05}),
            ("noRSI",              {"touch_pct": 0.05, "target_pct": 0.15, "stop_pct": 0.10, "use_rsi": False}),
            ("stale60",            {"touch_pct": 0.05, "target_pct": 0.15, "stop_pct": 0.10, "stale_bars": 60}),
            ("stale180",           {"touch_pct": 0.05, "target_pct": 0.15, "stop_pct": 0.10, "stale_bars": 180}),
            ("atr25",              {"touch_pct": 0.05, "target_pct": 0.15, "stop_pct": 0.10, "min_atr_pctl": 25}),
        ],
    },
    "OpenDrive": {
        "cls": OpeningDrive,
        "variants": [
            ("5m_tgt2x_stp1x",    {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 2.0, "stop_multiple": 1.0}),
            ("5m_tgt1.5x",        {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 1.5, "stop_multiple": 1.0}),
            ("5m_tgt3x",          {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 3.0, "stop_multiple": 1.0}),
            ("10m_tgt2x",         {"drive_minutes": 10, "min_drive_pct": 0.15, "target_multiple": 2.0, "stop_multiple": 1.0}),
            ("5m_drive0.15",      {"drive_minutes": 5, "min_drive_pct": 0.15, "target_multiple": 2.0, "stop_multiple": 1.0}),
            ("5m_drive0.20",      {"drive_minutes": 5, "min_drive_pct": 0.20, "target_multiple": 2.0, "stop_multiple": 1.0}),
            ("5m_gap_align",      {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 2.0, "stop_multiple": 1.0, "require_gap_align": True, "min_gap_pct": 0.2}),
            ("5m_atr25",          {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 2.0, "stop_multiple": 1.0, "min_atr_pctl": 25}),
        ],
    },
    "RangeExp": {
        "cls": RangeExpansion,
        "variants": [
            ("nr7_3bars_tgt2x",   {"nr_lookback": 7, "min_narrow_bars": 3, "target_multiple": 2.0}),
            ("nr7_3bars_tgt1.5x", {"nr_lookback": 7, "min_narrow_bars": 3, "target_multiple": 1.5}),
            ("nr7_3bars_tgt3x",   {"nr_lookback": 7, "min_narrow_bars": 3, "target_multiple": 3.0}),
            ("nr10_3bars",        {"nr_lookback": 10, "min_narrow_bars": 3, "target_multiple": 2.0}),
            ("nr7_5bars",         {"nr_lookback": 7, "min_narrow_bars": 5, "target_multiple": 2.0}),
            ("nr7_stale90",       {"nr_lookback": 7, "min_narrow_bars": 3, "target_multiple": 2.0, "stale_bars": 90}),
            ("nr7_stale120",      {"nr_lookback": 7, "min_narrow_bars": 3, "target_multiple": 2.0, "stale_bars": 120}),
            ("nr7_atr25",         {"nr_lookback": 7, "min_narrow_bars": 3, "target_multiple": 2.0, "min_atr_pctl": 25}),
        ],
    },
}

# Pairs strategy handled separately (needs two instruments merged)
PAIRS_VARIANTS = [
    ("z2.0_look60",   {"lookback": 60, "entry_zscore": 2.0, "exit_zscore": 0.5}),
    ("z2.0_look120",  {"lookback": 120, "entry_zscore": 2.0, "exit_zscore": 0.5}),
    ("z1.5_look60",   {"lookback": 60, "entry_zscore": 1.5, "exit_zscore": 0.3}),
    ("z2.5_look60",   {"lookback": 60, "entry_zscore": 2.5, "exit_zscore": 0.5}),
    ("z2.0_stale60",  {"lookback": 60, "entry_zscore": 2.0, "exit_zscore": 0.5, "stale_bars": 60}),
    ("z2.0_stale120", {"lookback": 60, "entry_zscore": 2.0, "exit_zscore": 0.5, "stale_bars": 120}),
]

PAIRS = [("SPY", "QQQ"), ("QQQ", "SPY"), ("GLD", "TLT"), ("XLK", "SMH")]


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _worker(args):
    strat_name, vlabel, cls_name, kwargs, sym, pkl_path, pair_pkl = args

    import sys, os, warnings, logging
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if project_root not in sys.path: sys.path.insert(0, project_root)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)

    from trading.backtest.walkforward import walk_forward as wf_func

    df = pd.read_pickle(pkl_path)

    # For pairs strategy, merge pair data
    if pair_pkl:
        pair_df = pd.read_pickle(pair_pkl)
        pair_close = pair_df.set_index("dt")["close"].rename("pair_close")
        df = df.set_index("dt").join(pair_close, how="left").reset_index()
        df["pair_close"] = df["pair_close"].ffill()

    cls_map = {
        "PriorDayReversal": PriorDayReversal,
        "OpeningDrive": OpeningDrive,
        "PairsSpread": PairsSpread,
        "RangeExpansion": RangeExpansion,
    }
    cls = cls_map[cls_name]
    strat = cls(**kwargs)
    df = strat.generate_signals(df)

    try:
        wf = wf_func(df, strat, sym, train_days=60, test_days=20, step_days=20)
    except Exception as e:
        return {"strat": strat_name, "variant": vlabel, "sym": sym,
                "sharpe": None, "trades": 0, "error": str(e)}

    all_daily = []
    for r in wf.oos_results:
        dr = r.daily_returns
        if dr is not None and len(dr) > 0: all_daily.append(dr)
    if not all_daily:
        return {"strat": strat_name, "variant": vlabel, "sym": sym, "sharpe": None, "trades": 0}

    combined = pd.concat(all_daily).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    if len(combined) < 10 or combined.std() == 0:
        return {"strat": strat_name, "variant": vlabel, "sym": sym, "sharpe": 0, "trades": wf.total_trades}

    sharpe = float((combined.mean() / combined.std()) * np.sqrt(252))
    total_ret = float((1+combined).prod()-1)
    cum = (1+combined).cumprod()
    max_dd = float(((cum-cum.cummax())/cum.cummax()).min())
    ds = combined[combined<0]; dss = ds.std() if len(ds)>1 else combined.std()
    sortino = float((combined.mean()/dss)*np.sqrt(252)) if dss>0 else 0
    ws = [r.sharpe_ratio for r in wf.oos_results]
    pw = sum(1 for s in ws if s > 0)

    return {
        "strat": strat_name, "variant": vlabel, "sym": sym,
        "sharpe": round(sharpe,3), "sortino": round(sortino,3),
        "return": round(total_ret,5), "max_dd": round(max_dd,5),
        "trades": wf.total_trades, "win_rate": round(wf.win_rate,4),
        "pf": round(wf.profit_factor,3), "windows_pos": f"{pw}/{len(ws)}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from trading.data.provider import get_minute_bars
    from trading.data.features import prepare_features

    print("=" * 100, flush=True)
    print("EXPERIMENT 15: WAVE 2 STRATEGY SEARCH", flush=True)
    print("4 new families: PriorDayReversal, OpeningDrive, PairsSpread, RangeExpansion", flush=True)
    print("=" * 100, flush=True)

    # Ensure pickled dev data exists
    print(f"\n{time.time()-t0:.0f}s | Loading data...", flush=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sym in UNIVERSE:
        pkl = DATA_CACHE_DIR / f"{sym}_dev.pkl"
        if not pkl.exists():
            df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
            df = prepare_features(df)
            df_dev = df.loc[df["dt"] <= DEV_END].copy()
            df_dev.to_pickle(str(pkl))
        print(f"  {sym}: ready", flush=True)

    # Build jobs for single-instrument strategies
    jobs = []
    for sname, sdef in GRID.items():
        for vlabel, vkwargs in sdef["variants"]:
            for sym in UNIVERSE:
                pkl = str(DATA_CACHE_DIR / f"{sym}_dev.pkl")
                jobs.append((sname, vlabel, sdef["cls"].__name__, vkwargs, sym, pkl, None))

    # Build jobs for pairs strategies
    for sym1, sym2 in PAIRS:
        for vlabel, vkwargs in PAIRS_VARIANTS:
            pkl1 = str(DATA_CACHE_DIR / f"{sym1}_dev.pkl")
            pkl2 = str(DATA_CACHE_DIR / f"{sym2}_dev.pkl")
            jobs.append(("Pairs", f"{sym1}v{sym2}_{vlabel}", "PairsSpread", vkwargs, sym1, pkl1, pkl2))

    total = len(jobs)
    print(f"\n{time.time()-t0:.0f}s | {total} jobs queued.", flush=True)

    # Run
    n_workers = min(os.cpu_count() or 4, 6)
    print(f"  Using {n_workers} workers\n", flush=True)
    results = []; done = 0; t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(_worker, j): j for j in jobs}
        for future in as_completed(future_map):
            done += 1
            try:
                r = future.result(); results.append(r)
                elapsed_p = time.time() - t_start
                rate = done / elapsed_p if elapsed_p > 0 else 1
                remaining = (total - done) / rate if rate > 0 else 0
                sh = r["sharpe"] if r["sharpe"] is not None else -99
                tag = "***" if sh >= 0.5 else "   "
                err = r.get("error", "")
                es = f"  ERR={err[:30]}" if err else ""
                print(f"  {done:>3}/{total} [{elapsed_p:.0f}s ETA {remaining:.0f}s]"
                      f" {tag} {r['strat']:<10} {r['variant']:<25} {r['sym']:<4}"
                      f" S={sh:>7.2f}  T={r['trades']:>4}{es}", flush=True)
            except Exception as e:
                print(f"  {done:>3}/{total} ERROR: {e}", flush=True)
            if done % 20 == 0:
                with open(RESULTS_PATH, "w") as f:
                    json.dump([r for r in results if isinstance(r, dict)], f, indent=2)

    # Save final
    with open(RESULTS_PATH, "w") as f:
        json.dump([r for r in results if isinstance(r, dict)], f, indent=2)

    # Results
    print(f"\n{time.time()-t0:.0f}s | All {total} tests complete.", flush=True)
    print(f"\n{'='*100}", flush=True)
    print("TOP RESULTS (Sharpe > 0)", flush=True)
    print(f"{'='*100}", flush=True)

    valid = [r for r in results if r.get("sharpe") is not None and r["sharpe"] > 0]
    valid.sort(key=lambda x: -x["sharpe"])

    for r in valid[:30]:
        print(f"  {r['strat']:<10} {r['variant']:<25} {r['sym']:<4}"
              f" S={r['sharpe']:>7.2f}  Sort={r.get('sortino',0):>6.2f}"
              f" Ret={r.get('return',0):>+7.2%}  T={r['trades']:>4}"
              f" WR={r.get('win_rate',0):>5.1%}  PF={r.get('pf',0):>5.2f}", flush=True)

    candidates = [r for r in valid if r["sharpe"] >= 0.5]
    print(f"\n  Candidates for locked OOS (Sharpe >= 0.5): {len(candidates)}", flush=True)

    if not candidates:
        print(f"\n  NO candidates survived dev screening.", flush=True)
        print(f"  CONCLUSION: Wave 2 strategies produce no edge.", flush=True)

    print(f"\n{time.time()-t0:.0f}s | EXPERIMENT 15 COMPLETE", flush=True)
