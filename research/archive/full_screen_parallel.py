#!/usr/bin/env python3
"""Full 280-combo screen with multiprocessing parallelism.

All 5 strategies x 8 variants x 7 instruments = 280 walk-forward tests.
Uses ProcessPoolExecutor to run tests in parallel across CPU cores.
Saves results incrementally to JSON so crashes don't lose work.
"""

import sys, os, io, json, time, logging, warnings, pickle, tempfile
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

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

UNIVERSE = ["SPY", "QQQ", "GLD", "XLE", "XLK", "SMH", "TLT"]
RESULTS_PATH = Path("research/screen_results.json")
DATA_CACHE_DIR = Path(tempfile.gettempdir()) / "spy_trader_screen"

t0 = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

from trading.strategies.base import Strategy


class MeanRevExtreme(Strategy):
    name = "mean_rev_extreme"
    def __init__(self, entry_zscore=2.0, exit_zscore=0.5, rsi_oversold=30,
                 rsi_overbought=70, min_atr_pctl=0, stale_bars=90,
                 cooldown_bars=15, last_entry_minute=900):
        self.entry_zscore = entry_zscore; self.exit_zscore = exit_zscore
        self.rsi_oversold = rsi_oversold; self.rsi_overbought = rsi_overbought
        self.min_atr_pctl = min_atr_pctl; self.stale_bars = stale_bars
        self.cooldown_bars = cooldown_bars; self.last_entry_minute = last_entry_minute
    def get_params(self):
        return {"z_in": self.entry_zscore, "z_out": self.exit_zscore,
                "rsi_os": self.rsi_oversold, "rsi_ob": self.rsi_overbought,
                "stale": self.stale_bars}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "vwap_dev" not in df.columns or "rsi" not in df.columns: return df
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); pos = 0; ebar = 0
        cdate = None; skip = False; bse = 999
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; bse = 999
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            vd = row.get("vwap_dev", 0); rsi = row.get("rsi", 50)
            if pd.isna(vd) or pd.isna(rsi): signals[i] = pos; continue
            if pos == 0:
                bse += 1
                if bse < self.cooldown_bars: signals[i] = 0; continue
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                if vd < -self.entry_zscore and rsi < self.rsi_oversold: pos = 1; ebar = i
                elif vd > self.entry_zscore and rsi > self.rsi_overbought: pos = -1; ebar = i
            elif pos == 1:
                if vd >= -self.exit_zscore: pos = 0; bse = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0; bse = 0
            elif pos == -1:
                if vd <= self.exit_zscore: pos = 0; bse = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0; bse = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


class VWAPTrend(Strategy):
    name = "vwap_trend"
    def __init__(self, confirm_bars=10, min_rel_vol=1.0, min_vwap_dist_pct=0.05,
                 stale_bars=120, last_entry_minute=900, min_atr_pctl=0):
        self.confirm_bars = confirm_bars; self.min_rel_vol = min_rel_vol
        self.min_vwap_dist_pct = min_vwap_dist_pct; self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute; self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"conf": self.confirm_bars, "vol": self.min_rel_vol,
                "dist": self.min_vwap_dist_pct, "stale": self.stale_bars}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "intraday_vwap" not in df.columns: return df
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        has_vol = self.min_rel_vol > 0 and "rel_volume" in df.columns
        signals = np.zeros(len(df)); pos = 0; ebar = 0
        ba = 0; bb = 0; cdate = None; skip = False
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; ba = 0; bb = 0
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            vwap = row.get("intraday_vwap", 0)
            if pd.isna(vwap) or vwap <= 0: signals[i] = pos; continue
            p = row["close"]; dp = (p - vwap) / vwap * 100
            if p > vwap: ba += 1; bb = 0
            elif p < vwap: bb += 1; ba = 0
            else: ba = 0; bb = 0
            if pos == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                vok = True
                if has_vol:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_rel_vol: vok = False
                if ba >= self.confirm_bars and dp >= self.min_vwap_dist_pct and vok: pos = 1; ebar = i
                elif bb >= self.confirm_bars and dp <= -self.min_vwap_dist_pct and vok: pos = -1; ebar = i
            elif pos == 1:
                if p < vwap: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                if p > vwap: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


class VolCompression(Strategy):
    name = "vol_compression"
    def __init__(self, bb_period=20, bb_std=2.0, squeeze_lookback=60,
                 squeeze_pctl=20, stale_bars=60, last_entry_minute=900, min_atr_pctl=0):
        self.bb_period = bb_period; self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback; self.squeeze_pctl = squeeze_pctl
        self.stale_bars = stale_bars; self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"bb_p": self.bb_period, "sq": self.squeeze_pctl, "stale": self.stale_bars}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "bb_upper" not in df.columns or "bb_lower" not in df.columns: return df
        bm = df["bb_mid"]; bw = (df["bb_upper"] - df["bb_lower"]) / bm.replace(0, np.nan)
        bwp = bw.rolling(self.squeeze_lookback, min_periods=20).rank(pct=True) * 100
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); pos = 0; ebar = 0
        sq = False; cdate = None; skip = False
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; sq = False
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            bp = bwp.iloc[i] if i < len(bwp) else 50
            if pd.isna(bp): signals[i] = pos; continue
            bu = row.get("bb_upper", 0); bl = row.get("bb_lower", 0)
            if pd.isna(bu) or pd.isna(bl) or bu <= 0: signals[i] = pos; continue
            if bp <= self.squeeze_pctl: sq = True
            if pos == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; sq = False; continue
                if sq and bp > self.squeeze_pctl:
                    if row["close"] > bu: pos = 1; ebar = i; sq = False
                    elif row["close"] < bl: pos = -1; ebar = i; sq = False
            elif pos == 1:
                m = row.get("bb_mid", 0)
                if pd.notna(m) and row["close"] < m: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            elif pos == -1:
                m = row.get("bb_mid", 0)
                if pd.notna(m) and row["close"] > m: pos = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


class MomentumScore(Strategy):
    name = "momentum_score"
    def __init__(self, entry_threshold=3, exit_threshold=1, rsi_bull=55, rsi_bear=45,
                 min_rel_vol=1.0, stale_bars=120, last_entry_minute=900,
                 min_atr_pctl=0, cooldown_bars=10):
        self.entry_threshold = entry_threshold; self.exit_threshold = exit_threshold
        self.rsi_bull = rsi_bull; self.rsi_bear = rsi_bear; self.min_rel_vol = min_rel_vol
        self.stale_bars = stale_bars; self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl; self.cooldown_bars = cooldown_bars
    def get_params(self):
        return {"thr": self.entry_threshold, "exit": self.exit_threshold, "stale": self.stale_bars}
    def _score(self, row):
        s = 0
        e8 = row.get("ema_8", 0); e21 = row.get("ema_21", 0)
        if pd.notna(e8) and pd.notna(e21) and e21 > 0:
            if e8 > e21: s += 1
            elif e8 < e21: s -= 1
        v = row.get("intraday_vwap", 0)
        if pd.notna(v) and v > 0:
            if row["close"] > v: s += 1
            elif row["close"] < v: s -= 1
        r = row.get("rsi", 50)
        if pd.notna(r):
            if r > self.rsi_bull: s += 1
            elif r < self.rsi_bear: s -= 1
        rv = row.get("rel_volume", 1.0)
        if pd.notna(rv) and rv >= self.min_rel_vol:
            if s > 0: s += 1
            elif s < 0: s -= 1
        return s
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); pos = 0; ebar = 0
        cdate = None; skip = False; bse = 999
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; bse = 999
                if has_atr:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            sc = self._score(row)
            if pos == 0:
                bse += 1
                if bse < self.cooldown_bars: signals[i] = 0; continue
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                if sc >= self.entry_threshold: pos = 1; ebar = i
                elif sc <= -self.entry_threshold: pos = -1; ebar = i
            elif pos == 1:
                if sc <= self.exit_threshold: pos = 0; bse = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0; bse = 0
            elif pos == -1:
                if sc >= -self.exit_threshold: pos = 0; bse = 0
                elif self.stale_bars > 0 and (i-ebar) >= self.stale_bars: pos = 0; bse = 0
            signals[i] = pos
        df["signal"] = signals.astype(int); return df


class GapContinuation(Strategy):
    name = "gap_continuation"
    def __init__(self, min_gap_pct=0.3, max_gap_pct=2.5, confirm_minutes=15,
                 target_pct=0.3, stop_pct=0.15, stale_bars=180,
                 last_entry_minute=720, min_atr_pctl=0):
        self.min_gap_pct = min_gap_pct; self.max_gap_pct = max_gap_pct
        self.confirm_minutes = confirm_minutes; self.target_pct = target_pct
        self.stop_pct = stop_pct; self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute; self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"gap": self.min_gap_pct, "tgt": self.target_pct,
                "stp": self.stop_pct, "conf": self.confirm_minutes}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "gap_pct" not in df.columns or "prev_close" not in df.columns: return df
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        ce = 9*60+30+self.confirm_minutes
        signals = np.zeros(len(df)); pos = 0; ep = 0.0; ebar = 0
        tgt = 0.0; stp = 0.0; cdate = None; dg = 0.0; pc = 0.0; skip = False; gh = False
        for i in range(len(df)):
            row = df.iloc[i]; d = row["date"]
            if d != cdate:
                pos = 0; cdate = d; skip = False; gh = False
                dg = row.get("gap_pct", 0); pc = row.get("prev_close", 0)
                if pd.isna(dg) or pd.isna(pc) or pc <= 0: skip = True
                elif abs(dg) < self.min_gap_pct or abs(dg) > self.max_gap_pct: skip = True
                if has_atr and not skip:
                    ap = row.get("atr_percentile", 50)
                    if pd.notna(ap) and ap < self.min_atr_pctl: skip = True
            if skip: signals[i] = 0; continue
            if row["minute_of_day"] > 930: pos = 0; signals[i] = 0; continue
            if row["minute_of_day"] < ce:
                if dg > 0 and row["low"] <= pc: skip = True
                elif dg < 0 and row["high"] >= pc: skip = True
                signals[i] = 0; continue
            if not gh and row["minute_of_day"] >= ce: gh = True
            if pos == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute: signals[i] = 0; continue
                if gh:
                    if dg > 0:
                        pos = 1; ep = row["close"]; ebar = i
                        tgt = ep*(1+self.target_pct/100); stp = ep*(1-self.stop_pct/100); gh = False
                    elif dg < 0:
                        pos = -1; ep = row["close"]; ebar = i
                        tgt = ep*(1-self.target_pct/100); stp = ep*(1+self.stop_pct/100); gh = False
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
    "MeanRev": {
        "cls": MeanRevExtreme,
        "variants": [
            ("z2.0_rsi30/70",   {"entry_zscore": 2.0, "rsi_oversold": 30, "rsi_overbought": 70}),
            ("z2.0_rsi25/75",   {"entry_zscore": 2.0, "rsi_oversold": 25, "rsi_overbought": 75}),
            ("z1.5_rsi30/70",   {"entry_zscore": 1.5, "exit_zscore": 0.3, "rsi_oversold": 30, "rsi_overbought": 70}),
            ("z2.5_rsi30/70",   {"entry_zscore": 2.5, "rsi_oversold": 30, "rsi_overbought": 70}),
            ("z2.0_stale60",    {"entry_zscore": 2.0, "stale_bars": 60}),
            ("z2.0_stale120",   {"entry_zscore": 2.0, "stale_bars": 120}),
            ("z2.0_atr25",      {"entry_zscore": 2.0, "min_atr_pctl": 25}),
            ("z2.0_cd30",       {"entry_zscore": 2.0, "cooldown_bars": 30}),
        ],
    },
    "VWAPTrend": {
        "cls": VWAPTrend,
        "variants": [
            ("default",         {"confirm_bars": 10, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}),
            ("conf5",           {"confirm_bars": 5, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}),
            ("conf15",          {"confirm_bars": 15, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}),
            ("dist0.10",        {"confirm_bars": 10, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.10}),
            ("vol1.2",          {"confirm_bars": 10, "min_rel_vol": 1.2, "min_vwap_dist_pct": 0.05}),
            ("atr25",           {"confirm_bars": 10, "min_rel_vol": 1.0, "min_atr_pctl": 25}),
            ("stale60",         {"confirm_bars": 10, "min_rel_vol": 1.0, "stale_bars": 60}),
            ("noVol",           {"confirm_bars": 10, "min_rel_vol": 0.0, "min_vwap_dist_pct": 0.05}),
        ],
    },
    "VolComp": {
        "cls": VolCompression,
        "variants": [
            ("sq20_stale60",    {"squeeze_pctl": 20, "stale_bars": 60}),
            ("sq20_stale90",    {"squeeze_pctl": 20, "stale_bars": 90}),
            ("sq15_stale60",    {"squeeze_pctl": 15, "stale_bars": 60}),
            ("sq25_stale60",    {"squeeze_pctl": 25, "stale_bars": 60}),
            ("sq20_stale120",   {"squeeze_pctl": 20, "stale_bars": 120}),
            ("sq20_bb30",       {"squeeze_pctl": 20, "stale_bars": 60, "bb_period": 30}),
            ("sq20_bb2.5",      {"squeeze_pctl": 20, "stale_bars": 60, "bb_std": 2.5}),
            ("sq20_atr25",      {"squeeze_pctl": 20, "stale_bars": 60, "min_atr_pctl": 25}),
        ],
    },
    "MomScore": {
        "cls": MomentumScore,
        "variants": [
            ("thr3_exit1",      {"entry_threshold": 3, "exit_threshold": 1}),
            ("thr3_exit0",      {"entry_threshold": 3, "exit_threshold": 0}),
            ("thr4_exit1",      {"entry_threshold": 4, "exit_threshold": 1}),
            ("thr3_stale60",    {"entry_threshold": 3, "exit_threshold": 1, "stale_bars": 60}),
            ("thr3_stale180",   {"entry_threshold": 3, "exit_threshold": 1, "stale_bars": 180}),
            ("thr3_cd20",       {"entry_threshold": 3, "exit_threshold": 1, "cooldown_bars": 20}),
            ("thr3_vol1.2",     {"entry_threshold": 3, "exit_threshold": 1, "min_rel_vol": 1.2}),
            ("thr3_atr25",      {"entry_threshold": 3, "exit_threshold": 1, "min_atr_pctl": 25}),
        ],
    },
    "GapCont": {
        "cls": GapContinuation,
        "variants": [
            ("g0.3_t0.3_s0.15", {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15}),
            ("g0.3_t0.2_s0.10", {"min_gap_pct": 0.3, "target_pct": 0.2, "stop_pct": 0.10}),
            ("g0.3_t0.4_s0.20", {"min_gap_pct": 0.3, "target_pct": 0.4, "stop_pct": 0.20}),
            ("g0.5_t0.3_s0.15", {"min_gap_pct": 0.5, "target_pct": 0.3, "stop_pct": 0.15}),
            ("g0.3_conf30",     {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "confirm_minutes": 30}),
            ("g0.3_stale90",    {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "stale_bars": 90}),
            ("g0.3_atr25",      {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "min_atr_pctl": 25}),
            ("g0.2_t0.3_s0.15", {"min_gap_pct": 0.2, "target_pct": 0.3, "stop_pct": 0.15}),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER FUNCTION (runs in subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def _worker(args):
    """Run one walk-forward test. Loads data from pickle cache."""
    strat_name, variant_label, cls_name, kwargs, sym, pickle_path = args

    # Fix sys.path for subprocess
    import sys, os, warnings, logging
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)

    from trading.backtest.walkforward import walk_forward as wf_func

    # Load data
    df_dev = pd.read_pickle(pickle_path)

    # Reconstruct strategy class
    cls_map = {
        "MeanRevExtreme": MeanRevExtreme,
        "VWAPTrend": VWAPTrend,
        "VolCompression": VolCompression,
        "MomentumScore": MomentumScore,
        "GapContinuation": GapContinuation,
    }
    cls = cls_map[cls_name]
    strat = cls(**kwargs)

    # Generate signals and run walk-forward
    df_dev = strat.generate_signals(df_dev)
    try:
        wf = wf_func(df_dev, strat, sym, train_days=60, test_days=20, step_days=20)
    except Exception as e:
        return {
            "strat": strat_name, "variant": variant_label, "sym": sym,
            "sharpe": None, "sortino": None, "return": None,
            "max_dd": None, "trades": 0, "win_rate": 0, "pf": 0,
            "windows_pos": "0/0", "error": str(e),
        }

    all_daily = []
    for oos_r in wf.oos_results:
        dr = oos_r.daily_returns
        if dr is not None and len(dr) > 0:
            all_daily.append(dr)
    if not all_daily:
        return {
            "strat": strat_name, "variant": variant_label, "sym": sym,
            "sharpe": None, "sortino": None, "return": None,
            "max_dd": None, "trades": 0, "win_rate": 0, "pf": 0,
            "windows_pos": "0/0",
        }

    combined = pd.concat(all_daily).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    if len(combined) < 10 or combined.std() == 0:
        return {
            "strat": strat_name, "variant": variant_label, "sym": sym,
            "sharpe": 0, "sortino": 0, "return": 0,
            "max_dd": 0, "trades": wf.total_trades, "win_rate": wf.win_rate, "pf": 0,
            "windows_pos": "0/0",
        }

    sharpe = float((combined.mean() / combined.std()) * np.sqrt(252))
    total_ret = float((1 + combined).prod() - 1)
    cum = (1 + combined).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    ds = combined[combined < 0]
    dss = ds.std() if len(ds) > 1 else combined.std()
    sortino = float((combined.mean() / dss) * np.sqrt(252)) if dss > 0 else 0

    ws = [r.sharpe_ratio for r in wf.oos_results]
    pw = sum(1 for s in ws if s > 0)

    return {
        "strat": strat_name, "variant": variant_label, "sym": sym,
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "return": round(total_ret, 5), "max_dd": round(max_dd, 5),
        "trades": wf.total_trades, "win_rate": round(wf.win_rate, 4),
        "pf": round(wf.profit_factor, 3),
        "windows_pos": f"{pw}/{len(ws)}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from trading.data.provider import get_minute_bars
    from trading.data.features import prepare_features
    from trading.config import ORB_SHARED_DEFAULTS, SYMBOL_PROFILES
    from trading.strategies.orb import ORBBreakout

    print("=" * 100, flush=True)
    print("FULL 280-COMBO SCREEN (5 strategies x 8 variants x 7 instruments)", flush=True)
    print("Walk-forward OOS on dev period (Jan-Nov 2025)", flush=True)
    print("=" * 100, flush=True)

    # ── Load data and pickle for worker processes ──
    print(f"\n{time.time()-t0:.0f}s | Loading data...", flush=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for sym in UNIVERSE:
        df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
        df = prepare_features(df)
        df_dev = df.loc[df["dt"] <= DEV_END].copy()
        pkl_path = DATA_CACHE_DIR / f"{sym}_dev.pkl"
        df_dev.to_pickle(str(pkl_path))
        nd = df_dev["date"].nunique()
        print(f"  {sym}: {len(df_dev)} dev bars, {nd} days -> {pkl_path}", flush=True)

    # ── Build job list ──
    jobs = []
    for sname, sdef in GRID.items():
        cls_name = sdef["cls"].__name__
        for vlabel, vkwargs in sdef["variants"]:
            for sym in UNIVERSE:
                pkl_path = str(DATA_CACHE_DIR / f"{sym}_dev.pkl")
                jobs.append((sname, vlabel, cls_name, vkwargs, sym, pkl_path))

    total = len(jobs)
    print(f"\n{time.time()-t0:.0f}s | {total} jobs queued. Starting parallel execution...", flush=True)

    # ── Load existing results for resume ──
    existing = {}
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                for r in json.load(f):
                    key = (r["strat"], r["variant"], r["sym"])
                    existing[key] = r
            print(f"  Loaded {len(existing)} cached results from {RESULTS_PATH}", flush=True)
        except Exception:
            pass

    # ── Filter out already-completed jobs ──
    remaining_jobs = []
    for job in jobs:
        key = (job[0], job[1], job[4])
        if key in existing:
            continue
        remaining_jobs.append(job)

    print(f"  {len(existing)} cached, {len(remaining_jobs)} remaining to run", flush=True)

    # ── Run with ProcessPoolExecutor ──
    n_workers = min(os.cpu_count() or 4, 6)  # Cap at 6 to avoid memory issues
    print(f"  Using {n_workers} worker processes\n", flush=True)

    results = list(existing.values())
    done = len(existing)
    start_parallel = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_job = {executor.submit(_worker, job): job for job in remaining_jobs}

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            done += 1
            try:
                r = future.result()
                results.append(r)

                # Print progress
                elapsed_p = time.time() - start_parallel
                rate = (done - len(existing)) / elapsed_p if elapsed_p > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0

                sh = r["sharpe"] if r["sharpe"] is not None else -99
                tag = "***" if sh >= 0.5 else "   "
                err = r.get("error", "")
                err_str = f"  ERR={err[:40]}" if err else ""
                print(f"  {done:>3}/{total} [{elapsed_p:.0f}s ETA {remaining:.0f}s]"
                      f" {tag} {r['strat']:<10} {r['variant']:<20} {r['sym']:<4}"
                      f" S={sh:>7.2f}  T={r['trades']:>4}{err_str}", flush=True)

            except Exception as e:
                print(f"  {done:>3}/{total} ERROR: {job[0]}/{job[1]}/{job[4]}: {e}", flush=True)

            # Save incrementally every 10 results
            if done % 10 == 0:
                serializable = [r for r in results if isinstance(r, dict)]
                with open(RESULTS_PATH, "w") as f:
                    json.dump(serializable, f, indent=2)

    # ── Final save ──
    serializable = [r for r in results if isinstance(r, dict)]
    with open(RESULTS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n{time.time()-t0:.0f}s | All {total} tests complete. Results saved to {RESULTS_PATH}", flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS TABLE
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*100}", flush=True)
    print("FULL RESULTS TABLE (sorted by Sharpe)", flush=True)
    print(f"{'='*100}", flush=True)

    valid = [r for r in results if r.get("sharpe") is not None]
    valid.sort(key=lambda x: -(x["sharpe"] or -99))

    print(f"\n  {'Strategy':<10} {'Variant':<20} {'Sym':<4} {'Sharpe':>7} {'Sortino':>8}"
          f" {'Return':>8} {'MaxDD':>7} {'Trades':>6} {'WR':>5} {'PF':>5} {'WinW':>5}", flush=True)
    print(f"  {'-'*9} {'-'*19} {'-'*3} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*5}", flush=True)

    for r in valid[:50]:  # Top 50
        print(f"  {r['strat']:<10} {r['variant']:<20} {r['sym']:<4}"
              f" {r['sharpe']:>7.2f} {r['sortino']:>8.2f}"
              f" {r['return']:>+8.2%} {r['max_dd']:>7.2%}"
              f" {r['trades']:>6} {r['win_rate']:>5.1%} {r['pf']:>5.2f}"
              f" {r.get('windows_pos',''):>5}", flush=True)

    # ── Summary by strategy ──
    print(f"\n{'='*100}", flush=True)
    print("SUMMARY BY STRATEGY (best Sharpe per strategy-instrument)", flush=True)
    print(f"{'='*100}", flush=True)

    from collections import defaultdict
    best_by_strat_sym = defaultdict(lambda: {"sharpe": -99})
    for r in valid:
        key = (r["strat"], r["sym"])
        if (r["sharpe"] or -99) > best_by_strat_sym[key]["sharpe"]:
            best_by_strat_sym[key] = r

    for sname in GRID:
        print(f"\n  {sname}:", flush=True)
        for sym in UNIVERSE:
            r = best_by_strat_sym.get((sname, sym))
            if r and r["sharpe"] != -99:
                print(f"    {sym}: best={r['variant']:<20} S={r['sharpe']:>6.2f}  T={r['trades']}", flush=True)
            else:
                print(f"    {sym}: no valid results", flush=True)

    # ── Candidates for locked OOS ──
    candidates = [r for r in valid if (r["sharpe"] or 0) >= 0.5]
    print(f"\n{'='*100}", flush=True)
    print(f"CANDIDATES FOR LOCKED OOS (walk-forward Sharpe >= 0.5): {len(candidates)}", flush=True)
    print(f"{'='*100}", flush=True)
    for r in candidates:
        print(f"  {r['strat']}/{r['variant']} {r['sym']}: S={r['sharpe']:.2f}"
              f"  Sort={r['sortino']:.2f}  Ret={r['return']:+.2%}  T={r['trades']}", flush=True)

    # ── Baseline comparison ──
    print(f"\n  Baseline ORB portfolio dev Sharpe: ~2.27 (SPY+QQQ)", flush=True)
    print(f"  Baseline ORB locked OOS Sharpe: ~3.35 (from Experiment 13)", flush=True)

    print(f"\n{time.time()-t0:.0f}s | SCREEN COMPLETE", flush=True)
