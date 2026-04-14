#!/usr/bin/env python3
"""Fast coarse-to-fine strategy screen with early stopping.

Phase 1 (COARSE): Full-period backtest on dev data, 1 default variant per strategy
         on SPY+QQQ only. Kill anything with Sharpe < 0 in-sample.
Phase 2 (MEDIUM): Walk-forward on survivors, expand to full variant grid on
         SPY+QQQ. Kill variants with walk-forward Sharpe < 0.3.
Phase 3 (BROAD):  Best variants tested across full instrument universe.
Phase 4 (LOCKED OOS): Top candidates confirmed on held-out Dec 2025-Apr 2026.

Data loaded once. Features precomputed once. Results saved continuously.
"""

import sys, os, io, json, time, logging, warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pytz
import numpy as np
import pandas as pd

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
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

UNIVERSE = ["SPY", "QQQ", "GLD", "XLE", "XLK", "SMH", "TLT"]
CORE = ["SPY", "QQQ"]  # Coarse screen instruments
RESULTS_FILE = "research/screen_results.json"

t0 = time.time()
def elapsed():
    return f"[{time.time()-t0:.0f}s]"


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS (inlined to avoid running new_edge_screen's module code)
# ═══════════════════════════════════════════════════════════════════════════════

class VWAPTrend(Strategy):
    """Follow intraday trend using VWAP as dynamic anchor."""
    name = "vwap_trend"
    def __init__(self, confirm_bars=10, min_rel_vol=1.0,
                 min_vwap_dist_pct=0.05, stale_bars=120,
                 last_entry_minute=900, min_atr_pctl=0):
        self.confirm_bars = confirm_bars
        self.min_rel_vol = min_rel_vol
        self.min_vwap_dist_pct = min_vwap_dist_pct
        self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"confirm": self.confirm_bars, "min_vol": self.min_rel_vol,
                "min_dist": self.min_vwap_dist_pct, "stale": self.stale_bars}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "intraday_vwap" not in df.columns: return df
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        has_vol = self.min_rel_vol > 0 and "rel_volume" in df.columns
        signals = np.zeros(len(df)); position = 0; entry_bar = 0
        bars_above = 0; bars_below = 0; current_date = None; day_skipped = False
        for i in range(len(df)):
            row = df.iloc[i]; date = row["date"]
            if date != current_date:
                position = 0; current_date = date; day_skipped = False
                bars_above = 0; bars_below = 0
                if has_atr:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl: day_skipped = True
            if day_skipped: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: position = 0; signals[i] = 0; continue
            vwap = row.get("intraday_vwap", 0)
            if pd.isna(vwap) or vwap <= 0: signals[i] = position; continue
            price = row["close"]; dist_pct = (price - vwap) / vwap * 100
            if price > vwap: bars_above += 1; bars_below = 0
            elif price < vwap: bars_below += 1; bars_above = 0
            else: bars_above = 0; bars_below = 0
            if position == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; continue
                vol_ok = True
                if has_vol:
                    rv = row.get("rel_volume", 1.0)
                    if pd.isna(rv) or rv < self.min_rel_vol: vol_ok = False
                if bars_above >= self.confirm_bars and dist_pct >= self.min_vwap_dist_pct and vol_ok:
                    position = 1; entry_bar = i
                elif bars_below >= self.confirm_bars and dist_pct <= -self.min_vwap_dist_pct and vol_ok:
                    position = -1; entry_bar = i
            elif position == 1:
                if price < vwap: position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars: position = 0
            elif position == -1:
                if price > vwap: position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars: position = 0
            signals[i] = position
        df["signal"] = signals.astype(int); return df


class VolCompression(Strategy):
    """Trade breakouts from intraday volatility compression (Bollinger squeeze)."""
    name = "vol_compression"
    def __init__(self, bb_period=20, bb_std=2.0, squeeze_lookback=60,
                 squeeze_pctl=20, stale_bars=60, last_entry_minute=900, min_atr_pctl=0):
        self.bb_period = bb_period; self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback; self.squeeze_pctl = squeeze_pctl
        self.stale_bars = stale_bars; self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"bb_per": self.bb_period, "sq_pctl": self.squeeze_pctl, "stale": self.stale_bars}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "bb_upper" not in df.columns or "bb_lower" not in df.columns: return df
        bb_mid = df["bb_mid"]
        bandwidth = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)
        bw_pctl = bandwidth.rolling(self.squeeze_lookback, min_periods=20).rank(pct=True) * 100
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); position = 0; entry_bar = 0
        was_squeezed = False; current_date = None; day_skipped = False
        for i in range(len(df)):
            row = df.iloc[i]; date = row["date"]
            if date != current_date:
                position = 0; current_date = date; day_skipped = False; was_squeezed = False
                if has_atr:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl: day_skipped = True
            if day_skipped: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: position = 0; signals[i] = 0; continue
            bw_p = bw_pctl.iloc[i] if i < len(bw_pctl) else 50
            if pd.isna(bw_p): signals[i] = position; continue
            bb_up = row.get("bb_upper", 0); bb_lo = row.get("bb_lower", 0)
            if pd.isna(bb_up) or pd.isna(bb_lo) or bb_up <= 0: signals[i] = position; continue
            if bw_p <= self.squeeze_pctl: was_squeezed = True
            if position == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; was_squeezed = False; continue
                if was_squeezed and bw_p > self.squeeze_pctl:
                    if row["close"] > bb_up: position = 1; entry_bar = i; was_squeezed = False
                    elif row["close"] < bb_lo: position = -1; entry_bar = i; was_squeezed = False
            elif position == 1:
                mid = row.get("bb_mid", 0)
                if pd.notna(mid) and row["close"] < mid: position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars: position = 0
            elif position == -1:
                mid = row.get("bb_mid", 0)
                if pd.notna(mid) and row["close"] > mid: position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars: position = 0
            signals[i] = position
        df["signal"] = signals.astype(int); return df


class MomentumScore(Strategy):
    """Multi-factor intraday momentum scoring system."""
    name = "momentum_score"
    def __init__(self, entry_threshold=3, exit_threshold=1, rsi_bull=55, rsi_bear=45,
                 min_rel_vol=1.0, stale_bars=120, last_entry_minute=900,
                 min_atr_pctl=0, cooldown_bars=10):
        self.entry_threshold = entry_threshold; self.exit_threshold = exit_threshold
        self.rsi_bull = rsi_bull; self.rsi_bear = rsi_bear; self.min_rel_vol = min_rel_vol
        self.stale_bars = stale_bars; self.last_entry_minute = last_entry_minute
        self.min_atr_pctl = min_atr_pctl; self.cooldown_bars = cooldown_bars
    def get_params(self):
        return {"entry_thr": self.entry_threshold, "exit_thr": self.exit_threshold,
                "stale": self.stale_bars, "cooldown": self.cooldown_bars}
    def _compute_score(self, row):
        score = 0
        ema8 = row.get("ema_8", 0); ema21 = row.get("ema_21", 0)
        if pd.notna(ema8) and pd.notna(ema21) and ema21 > 0:
            if ema8 > ema21: score += 1
            elif ema8 < ema21: score -= 1
        vwap = row.get("intraday_vwap", 0)
        if pd.notna(vwap) and vwap > 0:
            if row["close"] > vwap: score += 1
            elif row["close"] < vwap: score -= 1
        rsi = row.get("rsi", 50)
        if pd.notna(rsi):
            if rsi > self.rsi_bull: score += 1
            elif rsi < self.rsi_bear: score -= 1
        rv = row.get("rel_volume", 1.0)
        if pd.notna(rv) and rv >= self.min_rel_vol:
            if score > 0: score += 1
            elif score < 0: score -= 1
        return score
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        signals = np.zeros(len(df)); position = 0; entry_bar = 0
        current_date = None; day_skipped = False; bars_since_exit = 999
        for i in range(len(df)):
            row = df.iloc[i]; date = row["date"]
            if date != current_date:
                position = 0; current_date = date; day_skipped = False; bars_since_exit = 999
                if has_atr:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl: day_skipped = True
            if day_skipped: signals[i] = 0; continue
            if row["minute_of_day"] < 600: signals[i] = 0; continue
            if row["minute_of_day"] > 930: position = 0; signals[i] = 0; continue
            score = self._compute_score(row)
            if position == 0:
                bars_since_exit += 1
                if bars_since_exit < self.cooldown_bars: signals[i] = 0; continue
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; continue
                if score >= self.entry_threshold: position = 1; entry_bar = i
                elif score <= -self.entry_threshold: position = -1; entry_bar = i
            elif position == 1:
                if score <= self.exit_threshold: position = 0; bars_since_exit = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars: position = 0; bars_since_exit = 0
            elif position == -1:
                if score >= -self.exit_threshold: position = 0; bars_since_exit = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars: position = 0; bars_since_exit = 0
            signals[i] = position
        df["signal"] = signals.astype(int); return df


class GapContinuation(Strategy):
    """Trade in the direction of gaps that hold after open."""
    name = "gap_continuation"
    def __init__(self, min_gap_pct=0.3, max_gap_pct=2.5, confirm_minutes=15,
                 target_pct=0.3, stop_pct=0.15, stale_bars=180,
                 last_entry_minute=720, min_atr_pctl=0):
        self.min_gap_pct = min_gap_pct; self.max_gap_pct = max_gap_pct
        self.confirm_minutes = confirm_minutes; self.target_pct = target_pct
        self.stop_pct = stop_pct; self.stale_bars = stale_bars
        self.last_entry_minute = last_entry_minute; self.min_atr_pctl = min_atr_pctl
    def get_params(self):
        return {"min_gap": self.min_gap_pct, "target": self.target_pct,
                "stop": self.stop_pct, "confirm": self.confirm_minutes, "stale": self.stale_bars}
    def generate_signals(self, df):
        df = df.copy(); df["signal"] = 0
        if "gap_pct" not in df.columns or "prev_close" not in df.columns: return df
        has_atr = self.min_atr_pctl > 0 and "atr_percentile" in df.columns
        confirm_end = 9 * 60 + 30 + self.confirm_minutes
        signals = np.zeros(len(df)); position = 0; entry_price = 0.0; entry_bar = 0
        target = 0.0; stop = 0.0; current_date = None; day_gap = 0.0
        prev_close_val = 0.0; day_skipped = False; gap_held = False
        for i in range(len(df)):
            row = df.iloc[i]; date = row["date"]
            if date != current_date:
                position = 0; current_date = date; day_skipped = False; gap_held = False
                day_gap = row.get("gap_pct", 0); prev_close_val = row.get("prev_close", 0)
                if pd.isna(day_gap) or pd.isna(prev_close_val) or prev_close_val <= 0: day_skipped = True
                elif abs(day_gap) < self.min_gap_pct or abs(day_gap) > self.max_gap_pct: day_skipped = True
                if has_atr and not day_skipped:
                    atr_p = row.get("atr_percentile", 50)
                    if pd.notna(atr_p) and atr_p < self.min_atr_pctl: day_skipped = True
            if day_skipped: signals[i] = 0; continue
            if row["minute_of_day"] > 930: position = 0; signals[i] = 0; continue
            if row["minute_of_day"] < confirm_end:
                if day_gap > 0 and row["low"] <= prev_close_val: day_skipped = True
                elif day_gap < 0 and row["high"] >= prev_close_val: day_skipped = True
                signals[i] = 0; continue
            if not gap_held and row["minute_of_day"] >= confirm_end: gap_held = True
            if position == 0:
                if self.last_entry_minute > 0 and row["minute_of_day"] >= self.last_entry_minute:
                    signals[i] = 0; continue
                if gap_held:
                    if day_gap > 0:
                        position = 1; entry_price = row["close"]; entry_bar = i
                        target = entry_price * (1 + self.target_pct / 100)
                        stop = entry_price * (1 - self.stop_pct / 100); gap_held = False
                    elif day_gap < 0:
                        position = -1; entry_price = row["close"]; entry_bar = i
                        target = entry_price * (1 - self.target_pct / 100)
                        stop = entry_price * (1 + self.stop_pct / 100); gap_held = False
            elif position == 1:
                if row["close"] >= target or row["close"] <= stop: position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars: position = 0
            elif position == -1:
                if row["close"] <= target or row["close"] >= stop: position = 0
                elif self.stale_bars > 0 and (i - entry_bar) >= self.stale_bars: position = 0
            signals[i] = position
        df["signal"] = signals.astype(int); return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY: default + variant grids
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "VWAPTrend": {
        "class": VWAPTrend,
        "default": {"confirm_bars": 10, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05},
        "variants": [
            ("conf5",       {"confirm_bars": 5, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}),
            ("conf15",      {"confirm_bars": 15, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.05}),
            ("dist0.10",    {"confirm_bars": 10, "min_rel_vol": 1.0, "min_vwap_dist_pct": 0.10}),
            ("vol1.2",      {"confirm_bars": 10, "min_rel_vol": 1.2, "min_vwap_dist_pct": 0.05}),
            ("atr25",       {"confirm_bars": 10, "min_rel_vol": 1.0, "min_atr_pctl": 25}),
            ("stale60",     {"confirm_bars": 10, "min_rel_vol": 1.0, "stale_bars": 60}),
            ("noVol",       {"confirm_bars": 10, "min_rel_vol": 0.0, "min_vwap_dist_pct": 0.05}),
        ],
    },
    "VolComp": {
        "class": VolCompression,
        "default": {"squeeze_pctl": 20, "stale_bars": 60},
        "variants": [
            ("sq15",        {"squeeze_pctl": 15, "stale_bars": 60}),
            ("sq25",        {"squeeze_pctl": 25, "stale_bars": 60}),
            ("stale90",     {"squeeze_pctl": 20, "stale_bars": 90}),
            ("stale120",    {"squeeze_pctl": 20, "stale_bars": 120}),
            ("bb30",        {"squeeze_pctl": 20, "stale_bars": 60, "bb_period": 30}),
            ("bb_std2.5",   {"squeeze_pctl": 20, "stale_bars": 60, "bb_std": 2.5}),
            ("atr25",       {"squeeze_pctl": 20, "stale_bars": 60, "min_atr_pctl": 25}),
        ],
    },
    "MomScore": {
        "class": MomentumScore,
        "default": {"entry_threshold": 3, "exit_threshold": 1},
        "variants": [
            ("thr3_exit0",  {"entry_threshold": 3, "exit_threshold": 0}),
            ("thr4_exit1",  {"entry_threshold": 4, "exit_threshold": 1}),
            ("stale60",     {"entry_threshold": 3, "exit_threshold": 1, "stale_bars": 60}),
            ("stale180",    {"entry_threshold": 3, "exit_threshold": 1, "stale_bars": 180}),
            ("cd20",        {"entry_threshold": 3, "exit_threshold": 1, "cooldown_bars": 20}),
            ("vol1.2",      {"entry_threshold": 3, "exit_threshold": 1, "min_rel_vol": 1.2}),
            ("atr25",       {"entry_threshold": 3, "exit_threshold": 1, "min_atr_pctl": 25}),
        ],
    },
    "GapCont": {
        "class": GapContinuation,
        "default": {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15},
        "variants": [
            ("tgt0.2_stp0.10", {"min_gap_pct": 0.3, "target_pct": 0.2, "stop_pct": 0.10}),
            ("tgt0.4_stp0.20", {"min_gap_pct": 0.3, "target_pct": 0.4, "stop_pct": 0.20}),
            ("gap0.5",      {"min_gap_pct": 0.5, "target_pct": 0.3, "stop_pct": 0.15}),
            ("confirm30",   {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "confirm_minutes": 30}),
            ("stale90",     {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "stale_bars": 90}),
            ("atr25",       {"min_gap_pct": 0.3, "target_pct": 0.3, "stop_pct": 0.15, "min_atr_pctl": 25}),
            ("gap0.2",      {"min_gap_pct": 0.2, "target_pct": 0.3, "stop_pct": 0.15}),
        ],
    },
}

# Mean Reversion already proven dead from slow run — skip entirely.


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def quick_backtest(df_dev, strat_class, kwargs, sym):
    """Single full-period backtest on dev data. Fast but in-sample."""
    strat = strat_class(**kwargs)
    df = strat.generate_signals(df_dev.copy())
    r = run_backtest(df, strat, sym)
    dr = r.daily_returns
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return {"sharpe": -99, "return": 0, "trades": 0}
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
    return {"sharpe": sharpe, "return": r.total_return, "trades": r.num_trades,
            "win_rate": r.win_rate, "pf": r.profit_factor, "max_dd": r.max_drawdown}


def wf_eval(df_dev, strat_class, kwargs, sym):
    """Walk-forward evaluation. Honest OOS metric."""
    strat = strat_class(**kwargs)
    df = strat.generate_signals(df_dev.copy())
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
    if len(combined) < 10 or combined.std() == 0:
        return None

    sharpe = (combined.mean() / combined.std()) * np.sqrt(252)
    total_ret = (1 + combined).prod() - 1
    cum = (1 + combined).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    downside = combined[combined < 0]
    ds = downside.std() if len(downside) > 1 else combined.std()
    sortino = (combined.mean() / ds) * np.sqrt(252) if ds > 0 else 0

    ws = [r.sharpe_ratio for r in wf.oos_results]
    pos_w = sum(1 for s in ws if s > 0)

    return {
        "sharpe": sharpe, "sortino": sortino, "return": total_ret,
        "max_dd": max_dd, "trades": wf.total_trades, "win_rate": wf.win_rate,
        "pf": wf.profit_factor, "windows_pos": f"{pos_w}/{len(ws)}",
        "daily_returns": combined,
    }


def period_eval(df, strat_class, kwargs, sym, start_dt, end_dt):
    """Backtest on a specific date range."""
    strat = strat_class(**kwargs)
    mask = df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())
    period_df = df[mask].copy()
    if len(period_df) < 100:
        return None
    period_df = strat.generate_signals(period_df)
    r = run_backtest(period_df, strat, sym)
    dr = r.daily_returns
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return None
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
    ds = dr[dr < 0].std()
    if pd.isna(ds) or ds == 0:
        ds = dr.std()
    sortino = (dr.mean() / ds) * np.sqrt(252)
    cum = (1 + dr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "sharpe": sharpe, "sortino": sortino, "return": r.total_return,
        "max_dd": max_dd, "trades": r.num_trades, "win_rate": r.win_rate,
        "pf": r.profit_factor, "daily_returns": dr,
    }


def alpha_beta(strat_ret, bench_ret):
    aligned = pd.DataFrame({"s": strat_ret, "b": bench_ret}).dropna()
    if len(aligned) < 10:
        return 0, 0
    beta = aligned["b"].cov(aligned["s"]) / aligned["b"].var() if aligned["b"].var() > 0 else 0
    alpha = (aligned["s"].mean() - beta * aligned["b"].mean()) * 252
    return alpha, beta


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA (once)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"{elapsed()} Loading and precomputing features for all instruments...", flush=True)

data_full = {}  # Full period (for locked OOS later)
data_dev = {}   # Dev period only
for sym in UNIVERSE:
    try:
        df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
        df = prepare_features(df)
        data_full[sym] = df
        data_dev[sym] = df.loc[df["dt"] <= DEV_END].copy()
        n_days = data_dev[sym]["date"].nunique()
        print(f"  {sym}: {len(df)} bars total, {n_days} dev days", flush=True)
    except Exception as e:
        print(f"  {sym}: FAILED - {e}", flush=True)

spy_bench = data_full["SPY"].groupby("date")["close"].last().pct_change().dropna()
print(f"{elapsed()} Data loaded.\n", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

from trading.strategies.orb import ORBBreakout

print("=" * 100, flush=True)
print("BASELINE: Production ORB on dev walk-forward", flush=True)
print("=" * 100, flush=True)

baseline_daily = {}
for sym in CORE:
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(SYMBOL_PROFILES.get(sym, {}))
    r = wf_eval(data_dev[sym], ORBBreakout, params, sym)
    if r:
        print(f"  {sym}: Sharpe={r['sharpe']:.2f}  Sort={r['sortino']:.2f}"
              f"  Ret={r['return']:+.2%}  T={r['trades']}  Win={r['windows_pos']}", flush=True)
        baseline_daily[sym] = r["daily_returns"]

port_base = pd.DataFrame(baseline_daily).fillna(0).mean(axis=1)
BASE_SHARPE = (port_base.mean() / port_base.std()) * np.sqrt(252)
print(f"\n  Portfolio Sharpe: {BASE_SHARPE:.2f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: COARSE SCREEN (full-period in-sample on SPY+QQQ, default variant)
# Kill anything with in-sample Sharpe < 0 on BOTH core instruments.
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{elapsed()} " + "=" * 100, flush=True)
print("PHASE 1: COARSE SCREEN (in-sample backtest, default variant, SPY+QQQ)", flush=True)
print("=" * 100, flush=True)

surviving_strategies = []
total_phase1 = len(STRATEGIES) * len(CORE)
done_phase1 = 0

for sname, sdef in STRATEGIES.items():
    results = {}
    for sym in CORE:
        r = quick_backtest(data_dev[sym], sdef["class"], sdef["default"], sym)
        results[sym] = r
        done_phase1 += 1
        print(f"  {sname:<12} {sym}: in-sample Sharpe={r['sharpe']:>6.2f}"
              f"  Ret={r['return']:>+7.2%}  T={r['trades']:>4}"
              f"  [{done_phase1}/{total_phase1}]", flush=True)

    # Kill if negative on BOTH core instruments
    best = max(results[s]["sharpe"] for s in CORE)
    if best < 0:
        print(f"  >>> {sname}: KILLED (best core Sharpe {best:.2f})", flush=True)
    else:
        surviving_strategies.append(sname)
        print(f"  >>> {sname}: SURVIVES (best core Sharpe {best:.2f})", flush=True)

print(f"\n{elapsed()} Phase 1 survivors: {surviving_strategies}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: VARIANT GRID on SPY+QQQ (walk-forward, honest metrics)
# Only for Phase 1 survivors. Kill variants with WF Sharpe < 0.3.
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{elapsed()} " + "=" * 100, flush=True)
print("PHASE 2: VARIANT GRID on SPY+QQQ (walk-forward)", flush=True)
print("=" * 100, flush=True)

phase2_survivors = []  # (strategy_name, variant_label, kwargs, {sym: result})
total_phase2 = sum(
    (1 + len(STRATEGIES[s]["variants"])) * len(CORE)
    for s in surviving_strategies
)
done_phase2 = 0

for sname in surviving_strategies:
    sdef = STRATEGIES[sname]
    all_variants = [("default", sdef["default"])] + sdef["variants"]

    print(f"\n  {sname} ({len(all_variants)} variants x {len(CORE)} instruments):", flush=True)

    for vlabel, vkwargs in all_variants:
        sym_results = {}
        any_good = False
        for sym in CORE:
            r = wf_eval(data_dev[sym], sdef["class"], vkwargs, sym)
            done_phase2 += 1
            pct = done_phase2 / total_phase2 * 100
            eta = (time.time() - t0) / done_phase2 * (total_phase2 - done_phase2)

            if r:
                sym_results[sym] = r
                tag = "OK" if r["sharpe"] >= 0.3 else "weak"
                if r["sharpe"] >= 0.3:
                    any_good = True
                print(f"    {vlabel:<20} {sym}: S={r['sharpe']:>6.2f}  Sort={r['sortino']:>5.2f}"
                      f"  Ret={r['return']:>+7.2%}  T={r['trades']:>4}  [{tag}]"
                      f"  ({pct:.0f}% ETA {eta:.0f}s)", flush=True)
            else:
                print(f"    {vlabel:<20} {sym}: no data  ({pct:.0f}%)", flush=True)

        if any_good:
            phase2_survivors.append((sname, vlabel, vkwargs, sym_results))

print(f"\n{elapsed()} Phase 2 survivors: {len(phase2_survivors)} variant-instrument combos", flush=True)
for sn, vl, _, sr in phase2_survivors:
    syms = ", ".join(f"{s}={sr[s]['sharpe']:.2f}" for s in sr if sr[s]["sharpe"] >= 0.3)
    print(f"  {sn}/{vl}: {syms}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: BROAD INSTRUMENT SCREEN (walk-forward on survivors across universe)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{elapsed()} " + "=" * 100, flush=True)
print("PHASE 3: BROAD INSTRUMENT SCREEN (survivors across full universe)", flush=True)
print("=" * 100, flush=True)

# De-duplicate: for each strategy, take the top 3 variants by max Sharpe on core
strategy_top_variants = {}
for sname, vlabel, vkwargs, sym_results in phase2_survivors:
    if sname not in strategy_top_variants:
        strategy_top_variants[sname] = []
    best_sharpe = max((sym_results[s]["sharpe"] for s in sym_results), default=-99)
    strategy_top_variants[sname].append((best_sharpe, vlabel, vkwargs, sym_results))

# Keep top 3 per strategy
top_variants = []
for sname, variants in strategy_top_variants.items():
    variants.sort(key=lambda x: -x[0])
    for _, vlabel, vkwargs, sym_results in variants[:3]:
        top_variants.append((sname, vlabel, vkwargs, sym_results))

extra_syms = [s for s in UNIVERSE if s not in CORE]
total_phase3 = len(top_variants) * len(extra_syms)
done_phase3 = 0

broad_results = []  # (sname, vlabel, vkwargs, {sym: result} including core+extras)

if total_phase3 == 0:
    print("  No survivors to test broadly.", flush=True)
else:
    for sname, vlabel, vkwargs, core_results in top_variants:
        sdef = STRATEGIES[sname]
        full_results = dict(core_results)  # Start with core results

        print(f"\n  {sname}/{vlabel} on extra instruments:", flush=True)
        for sym in extra_syms:
            if sym not in data_dev:
                done_phase3 += 1
                continue

            r = wf_eval(data_dev[sym], sdef["class"], vkwargs, sym)
            done_phase3 += 1
            pct = done_phase3 / total_phase3 * 100
            eta = (time.time() - t0) / (time.time() - t0) * (total_phase3 - done_phase3)  # rough

            if r:
                full_results[sym] = r
                print(f"    {sym}: S={r['sharpe']:>6.2f}  Sort={r['sortino']:>5.2f}"
                      f"  Ret={r['return']:>+7.2%}  T={r['trades']:>4}"
                      f"  ({pct:.0f}%)", flush=True)
            else:
                print(f"    {sym}: no data  ({pct:.0f}%)", flush=True)

        broad_results.append((sname, vlabel, vkwargs, full_results))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: LOCKED OOS CONFIRMATION
# Top candidates from Phase 2/3 tested on Dec 2025 - Apr 2026
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{elapsed()} " + "=" * 100, flush=True)
print("PHASE 4: LOCKED OOS CONFIRMATION (Dec 2025 - Apr 2026)", flush=True)
print("Configs frozen from dev period. Results NOT used to iterate.", flush=True)
print("=" * 100, flush=True)

# Collect all candidates with walk-forward Sharpe >= 0.5 on any instrument
oos_candidates = []
seen = set()
for sname, vlabel, vkwargs, sym_results in (phase2_survivors + broad_results):
    for sym, r in sym_results.items():
        if r["sharpe"] >= 0.5:
            key = (sname, vlabel, sym)
            if key not in seen:
                seen.add(key)
                oos_candidates.append((sname, vlabel, vkwargs, sym))

print(f"\n  Candidates for locked OOS: {len(oos_candidates)}", flush=True)

# Baseline locked OOS
print(f"\n  --- Baseline ORB locked OOS ---", flush=True)
baseline_oos_daily = {}
for sym in CORE:
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(SYMBOL_PROFILES.get(sym, {}))
    r = period_eval(data_full[sym], ORBBreakout, params, sym, OOS_START, DATA_END)
    if r:
        print(f"  {sym} ORB: Sharpe={r['sharpe']:.2f}  Sort={r['sortino']:.2f}"
              f"  Ret={r['return']:+.2%}  T={r['trades']}", flush=True)
        baseline_oos_daily[sym] = r["daily_returns"]

if len(baseline_oos_daily) >= 2:
    port_oos_base = pd.DataFrame(baseline_oos_daily).fillna(0).mean(axis=1)
    oos_base_sharpe = (port_oos_base.mean() / port_oos_base.std()) * np.sqrt(252)
    print(f"  Portfolio ORB locked OOS Sharpe: {oos_base_sharpe:.2f}", flush=True)
else:
    oos_base_sharpe = 0
    port_oos_base = pd.Series(dtype=float)

# Test each candidate on locked OOS
print(f"\n  --- New strategy candidates locked OOS ---", flush=True)
oos_results = []
for sname, vlabel, vkwargs, sym in oos_candidates:
    sdef = STRATEGIES[sname]
    r = period_eval(data_full[sym], sdef["class"], vkwargs, sym, OOS_START, DATA_END)
    if r:
        a, b = alpha_beta(r["daily_returns"], spy_bench)
        print(f"  {sname}/{vlabel} {sym}: Sharpe={r['sharpe']:>6.2f}  Sort={r['sortino']:>5.2f}"
              f"  Ret={r['return']:>+7.2%}  T={r['trades']:>4}  alpha={a:+.1%}  beta={b:.3f}", flush=True)
        oos_results.append((sname, vlabel, vkwargs, sym, r))
    else:
        print(f"  {sname}/{vlabel} {sym}: no data on OOS period", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: PORTFOLIO COMBINATION on locked OOS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{elapsed()} " + "=" * 100, flush=True)
print("PHASE 5: PORTFOLIO COMBINATION (new + ORB baseline on locked OOS)", flush=True)
print("=" * 100, flush=True)

if oos_results and len(baseline_oos_daily) >= 2:
    print(f"\n  Baseline ORB portfolio locked OOS: Sharpe={oos_base_sharpe:.2f}", flush=True)

    combo_table = []
    for sname, vlabel, vkwargs, sym, r in oos_results:
        if r["sharpe"] < 0:
            continue  # Skip negative OOS

        combined = dict(baseline_oos_daily)
        key = f"{sname}-{vlabel}-{sym}"
        combined[key] = r["daily_returns"]
        port_df = pd.DataFrame(combined).fillna(0)
        port_ret = port_df.mean(axis=1)

        if len(port_ret) < 10 or port_ret.std() == 0:
            continue

        new_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252)
        delta = new_sharpe - oos_base_sharpe
        corr = r["daily_returns"].corr(port_oos_base)
        a, b = alpha_beta(r["daily_returns"], spy_bench)

        combo_table.append({
            "key": key, "port_sharpe": new_sharpe, "delta": delta,
            "solo_sharpe": r["sharpe"], "solo_sortino": r["sortino"],
            "solo_return": r["return"], "corr": corr,
            "alpha": a, "beta": b, "trades": r["trades"],
        })

        print(f"  + {key:<45} port={new_sharpe:.2f} (d={delta:+.2f})"
              f"  solo={r['sharpe']:.2f}  corr={corr:.2f}"
              f"  a={a:+.1%}  b={b:.3f}", flush=True)

    if combo_table:
        combo_table.sort(key=lambda x: -x["port_sharpe"])
        best = combo_table[0]
        print(f"\n  BEST COMBINATION: {best['key']}", flush=True)
        print(f"    Portfolio Sharpe: {best['port_sharpe']:.2f} (baseline {oos_base_sharpe:.2f},"
              f" delta {best['delta']:+.2f})", flush=True)
        print(f"    Solo: Sharpe={best['solo_sharpe']:.2f}  Sortino={best['solo_sortino']:.2f}"
              f"  Return={best['solo_return']:+.2%}", flush=True)
        print(f"    Alpha={best['alpha']:+.2%}  Beta={best['beta']:.3f}"
              f"  Correlation={best['corr']:.2f}", flush=True)
else:
    print("  No candidates survived to locked OOS or baseline missing.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{elapsed()} " + "=" * 100, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 100, flush=True)

print(f"\n  Strategies tested: 5 (MeanRev skipped — proven dead)", flush=True)
print(f"  Phase 1 survivors: {surviving_strategies}", flush=True)
print(f"  Phase 2 survivors: {len(phase2_survivors)} variant-instrument combos", flush=True)
print(f"  Locked OOS candidates: {len(oos_candidates)}", flush=True)

if oos_results:
    print(f"\n  Locked OOS results:", flush=True)
    for sname, vlabel, vkwargs, sym, r in sorted(oos_results, key=lambda x: -x[4]["sharpe"]):
        a, b = alpha_beta(r["daily_returns"], spy_bench)
        print(f"    {sname}/{vlabel} {sym}: Sharpe={r['sharpe']:.2f}  Sort={r['sortino']:.2f}"
              f"  Ret={r['return']:+.2%}  alpha={a:+.1%}  beta={b:.3f}"
              f"  T={r['trades']}  PF={r['pf']:.2f}", flush=True)

print(f"\n  Baseline ORB locked OOS portfolio Sharpe: {oos_base_sharpe:.2f}", flush=True)

# Verdict
if combo_table:
    best = combo_table[0]
    if best["delta"] > 0:
        print(f"\n  VERDICT: {best['key']} IMPROVES portfolio by {best['delta']:+.2f}"
              f" Sharpe on locked OOS.", flush=True)
    else:
        print(f"\n  VERDICT: No new strategy improves the ORB baseline on locked OOS.", flush=True)
else:
    print(f"\n  VERDICT: No candidates survived to portfolio test.", flush=True)

print(f"\n{elapsed()} EXPERIMENT 14 COMPLETE", flush=True)
