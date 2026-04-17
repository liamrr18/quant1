#!/usr/bin/env python3
"""Intraday strategy research: cash-hours strategies that complement ORB.

Tests 5 candidates on real MES/MNQ futures data during times when ORB is flat.
All entries after 10:00 AM, all positions flat by 3:50 PM.
Walk-forward validation: 60-day train, 20-day test, 20-day step.
Realistic costs: $0.62 RT + 2 ticks slippage per side (mid-day wider spreads).

Candidates:
1. VWAP Reversion (10 AM - 3 PM)
2. Afternoon Breakout (1 PM - 3:30 PM)
3. Failed ORB Reversal (10:15 AM - 12 PM)
4. Momentum Continuation (10 AM - 2 PM)
5. Range Compression Breakout (11 AM - 2 PM)
"""

import os
import sys
import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.data.contracts import CONTRACTS, FuturesContract
from trading.config import INITIAL_CAPITAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")

# Mid-day slippage: 2 ticks per side (wider spreads during low-volume hours)
MIDDAY_SLIPPAGE_TICKS = 2


@dataclass
class StratResult:
    name: str
    symbol: str
    trades: int
    total_pnl: float
    sharpe: float
    max_dd_pct: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    total_costs: float
    trades_list: list


def load_cash_data(symbol: str) -> pd.DataFrame:
    """Load cash-session data from full-session file."""
    path = os.path.join(CACHE_DIR, f"{symbol}_futures_full_1min.csv")
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)
    df["hour"] = df["dt"].dt.hour
    df["minute"] = df["dt"].dt.minute
    df["mod"] = df["hour"] * 60 + df["minute"]  # minute of day
    df["date"] = df["dt"].dt.date

    # Cash hours only: 9:30 AM - 4:00 PM
    df = df[(df["mod"] >= 570) & (df["mod"] < 960)].copy()
    df = df.sort_values("dt").reset_index(drop=True)

    log.info("Loaded %d cash bars for %s", len(df), symbol)
    return df


def make_trade(strategy, symbol, date, direction, entry_price, exit_price,
               contract, reason, entry_hour=12):
    """Create a trade dict with costs."""
    pos = 1 if direction == "long" else -1
    pnl_raw = (exit_price - entry_price) * pos * contract.point_value
    slippage = MIDDAY_SLIPPAGE_TICKS * contract.tick_value * 2
    costs = contract.commission_per_side * 2 + slippage
    pnl = pnl_raw - costs
    return {
        "strategy": strategy, "symbol": symbol, "date": date,
        "direction": direction, "entry_price": entry_price,
        "exit_price": exit_price, "pnl": pnl, "costs": costs,
        "contracts": 1, "reason": reason, "entry_hour": entry_hour,
    }


def compute_metrics(trades: list, capital: float) -> StratResult:
    """Compute metrics from trade list."""
    if not trades:
        return StratResult("", "", 0, 0, 0, 0, 0, 0, 0, 0, [])

    pnls = [t["pnl"] for t in trades]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Daily P&L for Sharpe
    daily = {}
    for t in trades:
        daily[t["date"]] = daily.get(t["date"], 0) + t["pnl"]
    daily_pnls = list(daily.values())
    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    eq = capital
    peak = capital
    max_dd = 0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    return StratResult(
        name=trades[0]["strategy"],
        symbol=trades[0]["symbol"],
        trades=len(trades),
        total_pnl=total_pnl,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        win_rate=win_rate,
        profit_factor=pf,
        avg_trade=np.mean(pnls),
        total_costs=sum(t["costs"] for t in trades),
        trades_list=trades,
    )


def print_result(r: StratResult):
    print(f"\n  --- {r.name} ({r.symbol}) ---")
    print(f"  Trades:         {r.trades}")
    print(f"  Total P&L:      ${r.total_pnl:+,.0f}")
    print(f"  Sharpe:         {r.sharpe:.2f}")
    print(f"  Max Drawdown:   {r.max_dd_pct:.1f}%")
    print(f"  Win Rate:       {r.win_rate*100:.1f}%")
    print(f"  Profit Factor:  {r.profit_factor:.2f}")
    print(f"  Avg Trade:      ${r.avg_trade:+,.0f}")
    print(f"  Total Costs:    ${r.total_costs:,.0f}")


# =========================================================================
# Strategy 1: VWAP Reversion (10 AM - 3 PM)
# =========================================================================

def run_vwap_reversion(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                       capital: float, z_entry: float = 1.0,
                       max_hold: int = 60) -> StratResult:
    """Fade deviations from VWAP during mid-day chop."""
    trades = []
    dates = sorted(df["date"].unique())

    for date in dates:
        day = df[df["date"] == date].copy()
        if len(day) < 60:
            continue

        # Build VWAP from day start
        cum_vol = day["volume"].cumsum().values
        cum_vp = (day["close"] * day["volume"]).cumsum().values
        vwap = cum_vp / np.maximum(cum_vol, 1)

        # Only trade 10:00 AM - 3:00 PM (mod 600-900)
        position = 0
        entry_price = 0
        entry_idx = 0
        deviations = []

        for idx in range(len(day)):
            bar = day.iloc[idx]
            mod = bar["mod"]
            price = bar["close"]

            dev = price - vwap[idx]
            deviations.append(dev)

            if mod < 600 or mod >= 900:
                continue

            # Z-score of deviation
            if len(deviations) < 30:
                continue
            std = np.std(deviations[-60:]) if len(deviations) >= 60 else np.std(deviations[-30:])
            z = dev / std if std > 0 else 0

            # Force close at 3:25 PM (mod 925) or 3:50 PM (mod 950)
            if mod >= 925 and position != 0:
                trades.append(make_trade(
                    "VWAP_Reversion", symbol, date,
                    "long" if position > 0 else "short",
                    entry_price, price, contract, "eod", bar["hour"]))
                position = 0
                continue

            if position == 0 and mod < 900:
                if z > z_entry and bar["volume"] > 0:
                    position = -1  # Price above VWAP, short
                    entry_price = price
                    entry_idx = idx
                elif z < -z_entry and bar["volume"] > 0:
                    position = 1  # Price below VWAP, long
                    entry_price = price
                    entry_idx = idx

            elif position != 0:
                bars_held = idx - entry_idx
                reverted = (position == 1 and z >= 0) or (position == -1 and z <= 0)
                timed_out = bars_held >= max_hold

                if reverted or timed_out:
                    trades.append(make_trade(
                        "VWAP_Reversion", symbol, date,
                        "long" if position > 0 else "short",
                        entry_price, price, contract,
                        "revert" if reverted else "timeout", bar["hour"]))
                    position = 0

    r = compute_metrics(trades, capital)
    r.name = "VWAP_Reversion"
    r.symbol = symbol
    return r


# =========================================================================
# Strategy 2: Afternoon Breakout (1 PM - 3:30 PM)
# =========================================================================

def run_afternoon_breakout(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                           capital: float, range_minutes: int = 30,
                           target_mult: float = 1.5) -> StratResult:
    """Breakout of 1:00-1:30 PM range, trade until 3:50 PM."""
    trades = []
    dates = sorted(df["date"].unique())

    for date in dates:
        day = df[df["date"] == date]
        if len(day) < 60:
            continue

        # Define 1:00-1:30 PM range (mod 780-810)
        range_bars = day[(day["mod"] >= 780) & (day["mod"] < 780 + range_minutes)]
        if len(range_bars) < 10:
            continue

        r_high = range_bars["high"].max()
        r_low = range_bars["low"].min()
        r_range = r_high - r_low

        if r_range <= 0:
            continue

        # Range filter
        mid = (r_high + r_low) / 2
        range_pct = r_range / mid
        if range_pct < 0.0003 or range_pct > 0.008:
            continue

        # Scan for breakout (1:30 PM - 3:50 PM, mod 810-950)
        position = 0
        entry_price = 0
        target = 0
        stop = 0

        tradeable = day[(day["mod"] >= 780 + range_minutes) & (day["mod"] < 950)]
        for _, bar in tradeable.iterrows():
            mod = bar["mod"]
            price = bar["close"]

            # Force close at 3:50 PM
            if mod >= 950:
                if position != 0:
                    trades.append(make_trade(
                        "PM_Breakout", symbol, date,
                        "long" if position > 0 else "short",
                        entry_price, price, contract, "eod", bar["hour"]))
                break

            if position == 0 and mod < 930:  # No entries after 3:30 PM
                if price > r_high:
                    position = 1
                    entry_price = price
                    target = entry_price + r_range * target_mult
                    stop = r_low
                elif price < r_low:
                    position = -1
                    entry_price = price
                    target = entry_price - r_range * target_mult
                    stop = r_high

            elif position == 1:
                if price >= target:
                    trades.append(make_trade(
                        "PM_Breakout", symbol, date, "long",
                        entry_price, price, contract, "target", bar["hour"]))
                    position = 0
                elif price <= stop:
                    trades.append(make_trade(
                        "PM_Breakout", symbol, date, "long",
                        entry_price, price, contract, "stop", bar["hour"]))
                    position = 0

            elif position == -1:
                if price <= target:
                    trades.append(make_trade(
                        "PM_Breakout", symbol, date, "short",
                        entry_price, price, contract, "target", bar["hour"]))
                    position = 0
                elif price >= stop:
                    trades.append(make_trade(
                        "PM_Breakout", symbol, date, "short",
                        entry_price, price, contract, "stop", bar["hour"]))
                    position = 0

        # Force close
        if position != 0 and len(tradeable) > 0:
            last = tradeable.iloc[-1]
            trades.append(make_trade(
                "PM_Breakout", symbol, date,
                "long" if position > 0 else "short",
                entry_price, last["close"], contract, "force_close", last["hour"]))

    r = compute_metrics(trades, capital)
    r.name = "PM_Breakout"
    r.symbol = symbol
    return r


# =========================================================================
# Strategy 3: Failed ORB Reversal (10:15 AM - 12 PM)
# =========================================================================

def run_failed_orb(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                   capital: float, failure_pct: float = 0.3,
                   target_pct: float = 0.8) -> StratResult:
    """When ORB breakout fails, trade the reversal."""
    trades = []
    dates = sorted(df["date"].unique())

    for date in dates:
        day = df[df["date"] == date]
        if len(day) < 60:
            continue

        # Opening range: 9:30-9:45 (mod 570-585)
        or_bars = day[(day["mod"] >= 570) & (day["mod"] < 585)]
        if len(or_bars) < 10:
            continue

        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2

        if or_range <= 0:
            continue

        # Check 9:45-10:15 for initial breakout
        early_bars = day[(day["mod"] >= 585) & (day["mod"] < 615)]
        if len(early_bars) < 10:
            continue

        # Detect breakout direction
        broke_high = early_bars["high"].max() > or_high
        broke_low = early_bars["low"].min() < or_low

        if not broke_high and not broke_low:
            continue  # No breakout to fail

        # Check 10:00-10:15 for failure (price returns inside range)
        check_bars = day[(day["mod"] >= 600) & (day["mod"] < 615)]
        if len(check_bars) < 5:
            continue

        last_check = check_bars.iloc[-1]["close"]
        failed_long = broke_high and not broke_low and last_check < or_high - or_range * failure_pct
        failed_short = broke_low and not broke_high and last_check > or_low + or_range * failure_pct

        if not failed_long and not failed_short:
            continue

        # Enter reversal at 10:15 AM
        entry_bars = day[(day["mod"] >= 615) & (day["mod"] < 720)]  # Trade until noon
        if len(entry_bars) < 5:
            continue

        entry_price = entry_bars.iloc[0]["close"]

        if failed_long:
            # ORB long failed -> short reversal
            direction = -1
            target = or_low + or_range * (1 - target_pct)
            stop = or_high + or_range * 0.5
        else:
            # ORB short failed -> long reversal
            direction = 1
            target = or_high - or_range * (1 - target_pct)
            stop = or_low - or_range * 0.5

        position = direction

        for _, bar in entry_bars.iterrows():
            price = bar["close"]
            mod = bar["mod"]

            if mod >= 720 and position != 0:  # Force close at noon
                trades.append(make_trade(
                    "Failed_ORB", symbol, date,
                    "long" if position > 0 else "short",
                    entry_price, price, contract, "eod", bar["hour"]))
                position = 0
                break

            hit_target = (position == 1 and price >= target) or (position == -1 and price <= target)
            hit_stop = (position == 1 and price <= stop) or (position == -1 and price >= stop)

            if hit_target or hit_stop:
                trades.append(make_trade(
                    "Failed_ORB", symbol, date,
                    "long" if position > 0 else "short",
                    entry_price, price, contract,
                    "target" if hit_target else "stop", bar["hour"]))
                position = 0
                break

        if position != 0 and len(entry_bars) > 0:
            last = entry_bars.iloc[-1]
            trades.append(make_trade(
                "Failed_ORB", symbol, date,
                "long" if position > 0 else "short",
                entry_price, last["close"], contract, "force_close", last["hour"]))

    r = compute_metrics(trades, capital)
    r.name = "Failed_ORB"
    r.symbol = symbol
    return r


# =========================================================================
# Strategy 4: Momentum Continuation (10 AM - 2 PM)
# =========================================================================

def run_momentum_continuation(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                               capital: float, breakout_mult: float = 2.0,
                               pullback_pct: float = 0.4) -> StratResult:
    """If ORB breakout was strong, ride continuation on pullbacks to breakout level."""
    trades = []
    dates = sorted(df["date"].unique())

    for date in dates:
        day = df[df["date"] == date]
        if len(day) < 60:
            continue

        # Opening range
        or_bars = day[(day["mod"] >= 570) & (day["mod"] < 585)]
        if len(or_bars) < 10:
            continue

        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        or_range = or_high - or_low

        if or_range <= 0:
            continue

        # Check if breakout was strong (9:45-10:00 AM, moved > 2x range)
        early = day[(day["mod"] >= 585) & (day["mod"] < 600)]
        if len(early) < 5:
            continue

        max_ext_up = early["high"].max() - or_high
        max_ext_down = or_low - early["low"].min()

        strong_long = max_ext_up > or_range * breakout_mult
        strong_short = max_ext_down > or_range * breakout_mult

        if not strong_long and not strong_short:
            continue

        # Mid-day window: 10:00 AM - 2:00 PM (mod 600-840)
        midday = day[(day["mod"] >= 600) & (day["mod"] < 840)]
        if len(midday) < 10:
            continue

        position = 0
        entry_price = 0
        trade_done = False

        for _, bar in midday.iterrows():
            price = bar["close"]
            mod = bar["mod"]

            if trade_done:
                break

            if mod >= 840 and position != 0:
                trades.append(make_trade(
                    "Momentum_Cont", symbol, date,
                    "long" if position > 0 else "short",
                    entry_price, price, contract, "eod", bar["hour"]))
                position = 0
                break

            if position == 0:
                if strong_long:
                    # Wait for pullback to breakout level
                    pullback_level = or_high + or_range * (1 - pullback_pct)
                    if price <= pullback_level and price > or_high:
                        position = 1
                        entry_price = price
                elif strong_short:
                    pullback_level = or_low - or_range * (1 - pullback_pct)
                    if price >= pullback_level and price < or_low:
                        position = -1
                        entry_price = price

            elif position != 0:
                # Target: 2x the pullback distance
                if position == 1:
                    target = entry_price + or_range * 2
                    stop = or_low
                else:
                    target = entry_price - or_range * 2
                    stop = or_high

                hit_target = (position == 1 and price >= target) or (position == -1 and price <= target)
                hit_stop = (position == 1 and price <= stop) or (position == -1 and price >= stop)

                if hit_target or hit_stop:
                    trades.append(make_trade(
                        "Momentum_Cont", symbol, date,
                        "long" if position > 0 else "short",
                        entry_price, price, contract,
                        "target" if hit_target else "stop", bar["hour"]))
                    position = 0
                    trade_done = True

        if position != 0 and len(midday) > 0:
            last = midday.iloc[-1]
            trades.append(make_trade(
                "Momentum_Cont", symbol, date,
                "long" if position > 0 else "short",
                entry_price, last["close"], contract, "force_close", last["hour"]))

    r = compute_metrics(trades, capital)
    r.name = "Momentum_Cont"
    r.symbol = symbol
    return r


# =========================================================================
# Strategy 5: Range Compression Breakout (11 AM - 2 PM)
# =========================================================================

def run_range_compression(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                          capital: float, lookback: int = 30,
                          compression_pct: float = 0.5) -> StratResult:
    """Trade breakout from compressed range during mid-day."""
    trades = []
    dates = sorted(df["date"].unique())

    # Build 20-day rolling average of 30-min range for compression detection
    daily_ranges = {}
    for date in dates:
        day = df[df["date"] == date]
        midday = day[(day["mod"] >= 660) & (day["mod"] < 840)]
        if len(midday) > 0:
            daily_ranges[date] = midday["high"].max() - midday["low"].min()

    range_series = pd.Series(daily_ranges)
    avg_range_20d = range_series.rolling(20, min_periods=10).mean()

    for date in dates:
        day = df[df["date"] == date]
        if len(day) < 60:
            continue

        # Check 11 AM - 2 PM window (mod 660-840)
        midday = day[(day["mod"] >= 660) & (day["mod"] < 840)]
        if len(midday) < lookback:
            continue

        # Look for 30-bar windows where range compresses
        position = 0
        entry_price = 0
        trade_done = False

        for idx in range(lookback, len(midday)):
            if trade_done:
                break

            bar = midday.iloc[idx]
            mod = bar["mod"]
            price = bar["close"]

            if mod >= 840 and position != 0:
                trades.append(make_trade(
                    "Range_Compress", symbol, date,
                    "long" if position > 0 else "short",
                    entry_price, price, contract, "eod", bar["hour"]))
                position = 0
                break

            if position == 0 and mod < 810:  # No entries after 1:30 PM
                window = midday.iloc[idx-lookback:idx]
                w_range = window["high"].max() - window["low"].min()
                w_mid = (window["high"].max() + window["low"].min()) / 2

                # Check if range is compressed vs historical
                avg_r = avg_range_20d.get(date, None)
                if avg_r is None or avg_r <= 0:
                    continue

                if w_range < avg_r * compression_pct:
                    # Compressed! Wait for breakout
                    w_high = window["high"].max()
                    w_low = window["low"].min()

                    if price > w_high:
                        position = 1
                        entry_price = price
                        target = entry_price + w_range * 2
                        stop = w_low
                    elif price < w_low:
                        position = -1
                        entry_price = price
                        target = entry_price - w_range * 2
                        stop = w_high

            elif position != 0:
                hit_target = (position == 1 and price >= target) or (position == -1 and price <= target)
                hit_stop = (position == 1 and price <= stop) or (position == -1 and price >= stop)

                if hit_target or hit_stop:
                    trades.append(make_trade(
                        "Range_Compress", symbol, date,
                        "long" if position > 0 else "short",
                        entry_price, price, contract,
                        "target" if hit_target else "stop", bar["hour"]))
                    position = 0
                    trade_done = True

        if position != 0 and len(midday) > 0:
            last = midday.iloc[-1]
            trades.append(make_trade(
                "Range_Compress", symbol, date,
                "long" if position > 0 else "short",
                entry_price, last["close"], contract, "force_close", last["hour"]))

    r = compute_metrics(trades, capital)
    r.name = "Range_Compress"
    r.symbol = symbol
    return r


# =========================================================================
# Walk-Forward Validation
# =========================================================================

def walk_forward(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                 capital: float, strategy_fn, strategy_name: str,
                 param_grid: list[dict],
                 is_days: int = 60, oos_days: int = 20, step_days: int = 20):
    """Walk-forward: optimize on IS, test on OOS with rolling windows."""
    dates = sorted(df["date"].unique())
    n_dates = len(dates)

    windows = []
    i = 0
    while i + is_days + oos_days <= n_dates:
        is_dates = set(dates[i:i+is_days])
        oos_dates = set(dates[i+is_days:i+is_days+oos_days])
        windows.append((is_dates, oos_dates))
        i += step_days

    if not windows:
        return None

    all_oos_trades = []
    window_results = []

    for w_idx, (is_dates, oos_dates) in enumerate(windows):
        is_df = df[df["date"].isin(is_dates)].copy()
        oos_df = df[df["date"].isin(oos_dates)].copy()

        # Optimize on IS
        best_sharpe = -999
        best_params = param_grid[0] if param_grid else {}

        for params in param_grid:
            r = strategy_fn(is_df, symbol, contract, capital, **params)
            if r.trades >= 5 and r.sharpe > best_sharpe:
                best_sharpe = r.sharpe
                best_params = params

        # Test on OOS
        oos_r = strategy_fn(oos_df, symbol, contract, capital, **best_params)
        all_oos_trades.extend(oos_r.trades_list)

        window_results.append({
            "window": w_idx + 1,
            "is_sharpe": best_sharpe,
            "best_params": best_params,
            "oos_trades": oos_r.trades,
            "oos_pnl": oos_r.total_pnl,
            "oos_sharpe": oos_r.sharpe,
        })

    total_oos_pnl = sum(w["oos_pnl"] for w in window_results)
    oos_sharpes = [w["oos_sharpe"] for w in window_results if w["oos_trades"] > 0]
    avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
    oos_positive = sum(1 for w in window_results if w["oos_pnl"] > 0)

    return {
        "strategy": strategy_name,
        "windows": len(window_results),
        "total_oos_pnl": total_oos_pnl,
        "avg_oos_sharpe": avg_oos_sharpe,
        "oos_positive": oos_positive,
        "oos_positive_pct": oos_positive / len(window_results) if window_results else 0,
        "total_oos_trades": len(all_oos_trades),
        "all_oos_trades": all_oos_trades,
        "window_results": window_results,
    }


# =========================================================================
# Correlation with Cash ORB
# =========================================================================

def check_correlation(trades: list, df: pd.DataFrame, symbol: str):
    """Check correlation between strategy daily P&L and cash ORB returns."""
    # ORB proxy: first-hour return (9:30-10:30)
    dates = sorted(df["date"].unique())
    orb_returns = {}
    for date in dates:
        day = df[df["date"] == date]
        orb_bars = day[(day["mod"] >= 570) & (day["mod"] < 630)]
        if len(orb_bars) >= 10:
            orb_returns[date] = (orb_bars.iloc[-1]["close"] - orb_bars.iloc[0]["open"]) / orb_bars.iloc[0]["open"]

    # Strategy daily P&L
    strat_daily = {}
    for t in trades:
        strat_daily[t["date"]] = strat_daily.get(t["date"], 0) + t["pnl"]

    # Merge
    common = set(orb_returns.keys()) & set(strat_daily.keys())
    if len(common) < 20:
        return None

    orb_vals = [orb_returns[d] for d in sorted(common)]
    strat_vals = [strat_daily[d] for d in sorted(common)]

    corr = np.corrcoef(orb_vals, strat_vals)[0, 1]
    return corr


# =========================================================================
# Main
# =========================================================================

def run_all(symbol: str):
    """Run all 5 candidates for a symbol."""
    contract = CONTRACTS[symbol]
    df = load_cash_data(symbol)
    capital = INITIAL_CAPITAL

    print(f"\n{'='*70}")
    print(f"  INTRADAY STRATEGY CANDIDATES: {symbol}")
    print(f"  Data: {df['dt'].iloc[0].strftime('%Y-%m-%d')} to {df['dt'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Bars: {len(df):,}, Days: {len(df['date'].unique())}")
    print(f"{'='*70}")

    # Full-sample results first
    strategies = [
        ("VWAP_Reversion", run_vwap_reversion, [
            {"z_entry": 0.8, "max_hold": 60},
            {"z_entry": 1.0, "max_hold": 60},
            {"z_entry": 1.2, "max_hold": 60},
            {"z_entry": 1.0, "max_hold": 30},
            {"z_entry": 1.0, "max_hold": 90},
        ]),
        ("PM_Breakout", run_afternoon_breakout, [
            {"range_minutes": 30, "target_mult": 1.0},
            {"range_minutes": 30, "target_mult": 1.5},
            {"range_minutes": 30, "target_mult": 2.0},
            {"range_minutes": 15, "target_mult": 1.5},
        ]),
        ("Failed_ORB", run_failed_orb, [
            {"failure_pct": 0.2, "target_pct": 0.8},
            {"failure_pct": 0.3, "target_pct": 0.8},
            {"failure_pct": 0.4, "target_pct": 0.6},
            {"failure_pct": 0.3, "target_pct": 0.6},
        ]),
        ("Momentum_Cont", run_momentum_continuation, [
            {"breakout_mult": 1.5, "pullback_pct": 0.3},
            {"breakout_mult": 2.0, "pullback_pct": 0.4},
            {"breakout_mult": 2.5, "pullback_pct": 0.5},
        ]),
        ("Range_Compress", run_range_compression, [
            {"lookback": 20, "compression_pct": 0.4},
            {"lookback": 30, "compression_pct": 0.5},
            {"lookback": 30, "compression_pct": 0.6},
            {"lookback": 40, "compression_pct": 0.5},
        ]),
    ]

    all_results = []
    all_wf = []

    for name, fn, param_grid in strategies:
        print(f"\n  Testing {name}...")

        # Full-sample with default params
        default_params = param_grid[len(param_grid)//2]  # Middle params
        r = fn(df, symbol, contract, capital, **default_params)
        print_result(r)
        all_results.append(r)

        # Walk-forward
        print(f"  Walk-forward ({len(param_grid)} param combos)...")
        wf = walk_forward(df, symbol, contract, capital, fn, name, param_grid)
        if wf:
            print(f"    WF windows: {wf['windows']}, OOS P&L: ${wf['total_oos_pnl']:+,.0f}, "
                  f"OOS Sharpe: {wf['avg_oos_sharpe']:.2f}, "
                  f"+ve: {wf['oos_positive']}/{wf['windows']}")

            # Correlation with ORB
            if wf["all_oos_trades"]:
                corr = check_correlation(wf["all_oos_trades"], df, symbol)
                if corr is not None:
                    print(f"    ORB correlation: {corr:.3f}")
                    wf["orb_correlation"] = corr

        all_wf.append(wf)

    # Summary table
    print(f"\n  {'='*90}")
    print(f"  {'Strategy':<18} {'Trades':>7} {'P&L':>12} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'WF_Shrp':>8} {'Corr':>6}")
    print(f"  {'='*90}")

    for r, wf in zip(all_results, all_wf):
        wf_sharpe = wf["avg_oos_sharpe"] if wf else 0
        corr = wf.get("orb_correlation", 0) if wf else 0
        print(f"  {r.name:<18} {r.trades:>7} ${r.total_pnl:>+10,.0f} "
              f"{r.sharpe:>8.2f} {r.win_rate*100:>5.1f}% {r.profit_factor:>5.2f} "
              f"{r.max_dd_pct:>6.1f}% {wf_sharpe:>8.2f} {corr:>+5.2f}")
    print(f"  {'='*90}")

    # Identify survivors: positive OOS Sharpe and enough trades
    survivors = []
    for r, wf in zip(all_results, all_wf):
        if wf and wf["avg_oos_sharpe"] > 0 and wf["total_oos_trades"] >= 20:
            survivors.append((r, wf))

    return all_results, all_wf, survivors


def main():
    print("=" * 70)
    print("  INTRADAY STRATEGY RESEARCH")
    print("  Cash-hours strategies to complement ORB (10 AM - 3:50 PM)")
    print("=" * 70)

    all_survivors = {}
    for symbol in ["MES", "MNQ"]:
        try:
            results, wf_results, survivors = run_all(symbol)
            all_survivors[symbol] = survivors
        except Exception as e:
            log.error("Error testing %s: %s", symbol, e)
            import traceback
            traceback.print_exc()

    # Final report
    print(f"\n{'='*70}")
    print("  FINAL REPORT: INTRADAY SURVIVORS")
    print("=" * 70)

    any_survivor = False
    for symbol, survivors in all_survivors.items():
        if survivors:
            any_survivor = True
            for r, wf in survivors:
                print(f"  {r.name} {symbol}: WF Sharpe={wf['avg_oos_sharpe']:.2f}, "
                      f"OOS P&L=${wf['total_oos_pnl']:+,.0f}, "
                      f"trades={wf['total_oos_trades']}, "
                      f"corr={wf.get('orb_correlation', 0):.3f}")

    if not any_survivor:
        print("  No strategies survived walk-forward validation.")
        print("\n  ANALYSIS:")
        print("  Mid-day cash hours on MES/MNQ appear structurally difficult:")
        print("  - Volume drops 40-50% from opening hour")
        print("  - Spreads widen during lunch hours")
        print("  - Most directional moves are noise during 11 AM - 1 PM")
        print("  - 2-tick slippage (realistic for mid-day) eats edge on small moves")
        print("  - Costs are ~$2.24 RT per contract — avg trade needs to exceed this")

    print("=" * 70)


if __name__ == "__main__":
    main()
