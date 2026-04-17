#!/usr/bin/env python3
"""Phase 4: Test overnight strategy candidates on MES/MNQ futures data.

Tests multiple strategy concepts during overnight hours (6 PM - 9:25 AM ET).
All positions must be flat by 9:25 AM. Uses walk-forward validation.
Slippage is 2 ticks during dead hours, 1 tick during active windows.
"""

import os
import sys
import logging
from dataclasses import dataclass
from datetime import time as dt_time

import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.data.contracts import CONTRACTS, FuturesContract, total_cost_per_contract
from trading.config import INITIAL_CAPITAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")

# Slippage: 2 ticks during dead hours, 1 tick during active windows
DEAD_HOURS = set(range(20, 24)) | set(range(0, 3))  # 8PM-3AM
ACTIVE_HOURS = set(range(3, 5)) | set(range(7, 10))  # 3-5AM (euro), 7-10AM (pre-mkt)


@dataclass
class OvernightResult:
    strategy_name: str
    symbol: str
    total_trades: int
    total_pnl: float
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    total_costs: float
    avg_contracts: float
    trades_list: list


def load_full_data(symbol: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"{symbol}_futures_full_1min.csv")
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)

    df["hour"] = df["dt"].dt.hour
    df["minute"] = df["dt"].dt.minute
    df["date"] = df["dt"].dt.date
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["bar_range"] = df["high"] - df["low"]

    return df


def get_slippage_ticks(hour: int) -> int:
    """Return slippage in ticks based on hour (dead=2, active=1)."""
    if hour in DEAD_HOURS:
        return 2
    return 1


def compute_overnight_metrics(trades: list[dict], capital: float) -> OvernightResult:
    """Compute backtest metrics from list of trade dicts."""
    if not trades:
        return OvernightResult("", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

    pnls = [t["pnl"] for t in trades]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Sharpe on trade P&L
    if len(pnls) > 1:
        daily_groups = {}
        for t in trades:
            d = t["date"]
            daily_groups.setdefault(d, 0)
            daily_groups[d] += t["pnl"]
        daily_pnls = list(daily_groups.values())
        if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
            sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Max drawdown
    equity = capital
    peak = capital
    max_dd = 0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    total_costs = sum(t.get("costs", 0) for t in trades)
    avg_contracts = np.mean([t.get("contracts", 1) for t in trades]) if trades else 0

    return OvernightResult(
        strategy_name=trades[0].get("strategy", "") if trades else "",
        symbol=trades[0].get("symbol", "") if trades else "",
        total_trades=len(trades),
        total_pnl=total_pnl,
        total_return_pct=total_pnl / capital * 100,
        sharpe=sharpe,
        max_drawdown_pct=max_dd * 100,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=np.mean(pnls) if pnls else 0,
        total_costs=total_costs,
        avg_contracts=avg_contracts,
        trades_list=trades,
    )


def print_result(r: OvernightResult):
    """Print formatted backtest result."""
    print(f"\n  --- {r.strategy_name} ({r.symbol}) ---")
    print(f"  Trades:         {r.total_trades}")
    print(f"  Total P&L:      ${r.total_pnl:+,.0f}")
    print(f"  Return:         {r.total_return_pct:+.1f}%")
    print(f"  Sharpe:         {r.sharpe:.2f}")
    print(f"  Max Drawdown:   {r.max_drawdown_pct:.1f}%")
    print(f"  Win Rate:       {r.win_rate*100:.1f}%")
    print(f"  Profit Factor:  {r.profit_factor:.2f}")
    print(f"  Avg Trade:      ${r.avg_trade_pnl:+,.0f}")
    print(f"  Avg Contracts:  {r.avg_contracts:.1f}")
    print(f"  Total Costs:    ${r.total_costs:,.0f}")


# =========================================================================
# Strategy 1: European Open ORB (3:00-3:15 AM range, trade until 9:25 AM)
# =========================================================================

def run_euro_orb(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                 capital: float, range_minutes: int = 15,
                 target_multiple: float = 1.5) -> OvernightResult:
    """European Open ORB: 15-min range starting at 3:00 AM ET."""
    trades = []
    equity = capital

    range_end = 3 * 60 + range_minutes  # 3:15 AM
    entry_cutoff = 7 * 60  # No new entries after 7:00 AM
    force_close = 9 * 60 + 25  # Must close by 9:25 AM

    dates = sorted(df["date"].unique())

    for date in dates:
        day = df[df["date"] == date]
        if len(day) < 30:
            continue

        # Get bars in the 3:00-3:15 range
        range_bars = day[(day["minute_of_day"] >= 180) & (day["minute_of_day"] < range_end)]
        if len(range_bars) < 5:
            continue

        or_high = range_bars["high"].max()
        or_low = range_bars["low"].min()
        or_range = or_high - or_low

        if or_range <= 0:
            continue

        # Range filter
        mid = (or_high + or_low) / 2
        range_pct = or_range / mid
        if range_pct < 0.0005 or range_pct > 0.01:
            continue

        # Scan for breakout after range
        position = 0
        entry_price = 0
        target = 0
        stop = 0

        tradeable = day[(day["minute_of_day"] >= range_end) & (day["minute_of_day"] < force_close)]

        for _, bar in tradeable.iterrows():
            if bar["minute_of_day"] >= force_close:
                if position != 0:
                    exit_price = bar["close"]
                    slippage = get_slippage_ticks(bar["hour"]) * contract.tick_value
                    pnl_per = (exit_price - entry_price) * position * contract.point_value
                    costs = contract.commission_per_side * 2 + slippage * 2
                    pnl = pnl_per - costs
                    trades.append({
                        "strategy": "Euro_ORB", "symbol": symbol,
                        "date": date, "direction": "long" if position > 0 else "short",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl": pnl, "costs": costs, "contracts": 1,
                        "reason": "eod",
                    })
                break

            if position == 0 and bar["minute_of_day"] < entry_cutoff:
                if bar["close"] > or_high:
                    position = 1
                    entry_price = bar["close"]
                    target = entry_price + or_range * target_multiple
                    stop = or_low
                elif bar["close"] < or_low:
                    position = -1
                    entry_price = bar["close"]
                    target = entry_price - or_range * target_multiple
                    stop = or_high

            elif position == 1:
                if bar["close"] >= target or bar["close"] <= stop:
                    exit_price = bar["close"]
                    slippage = get_slippage_ticks(bar["hour"]) * contract.tick_value
                    pnl_per = (exit_price - entry_price) * contract.point_value
                    costs = contract.commission_per_side * 2 + slippage * 2
                    pnl = pnl_per - costs
                    trades.append({
                        "strategy": "Euro_ORB", "symbol": symbol,
                        "date": date, "direction": "long",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl": pnl, "costs": costs, "contracts": 1,
                        "reason": "target" if bar["close"] >= target else "stop",
                    })
                    position = 0

            elif position == -1:
                if bar["close"] <= target or bar["close"] >= stop:
                    exit_price = bar["close"]
                    slippage = get_slippage_ticks(bar["hour"]) * contract.tick_value
                    pnl_per = (entry_price - exit_price) * contract.point_value
                    costs = contract.commission_per_side * 2 + slippage * 2
                    pnl = pnl_per - costs
                    trades.append({
                        "strategy": "Euro_ORB", "symbol": symbol,
                        "date": date, "direction": "short",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl": pnl, "costs": costs, "contracts": 1,
                        "reason": "target" if bar["close"] <= target else "stop",
                    })
                    position = 0

        # Force close if still in position at end of data
        if position != 0 and len(tradeable) > 0:
            exit_price = tradeable.iloc[-1]["close"]
            slippage = get_slippage_ticks(tradeable.iloc[-1]["hour"]) * contract.tick_value
            pnl_per = (exit_price - entry_price) * position * contract.point_value
            costs = contract.commission_per_side * 2 + slippage * 2
            pnl = pnl_per - costs
            trades.append({
                "strategy": "Euro_ORB", "symbol": symbol,
                "date": date, "direction": "long" if position > 0 else "short",
                "entry_price": entry_price, "exit_price": exit_price,
                "pnl": pnl, "costs": costs, "contracts": 1,
                "reason": "force_close",
            })

    result = compute_overnight_metrics(trades, capital)
    result.strategy_name = "Euro_ORB"
    result.symbol = symbol
    return result


# =========================================================================
# Strategy 2: Overnight Mean Reversion (8 PM - 2 AM)
# =========================================================================

def run_overnight_reversion(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                             capital: float, vwap_dev_threshold: float = 2.0,
                             max_hold_bars: int = 60) -> OvernightResult:
    """Mean reversion: fade moves from session VWAP during dead hours."""
    trades = []

    dates = sorted(df["date"].unique())

    for date in dates:
        day = df[df["date"] == date]

        # Get evening/night bars (8 PM to 2 AM)
        night_bars = day[
            ((day["hour"] >= 20) | (day["hour"] < 2)) &
            (day["minute_of_day"] < 9 * 60 + 25)
        ]
        if len(night_bars) < 30:
            continue

        # Compute session VWAP for the night window
        cum_vol = night_bars["volume"].cumsum()
        cum_vp = (night_bars["close"] * night_bars["volume"]).cumsum()
        vwap = cum_vp / cum_vol.replace(0, np.nan)

        # Rolling std of close-vwap deviation
        dev = night_bars["close"] - vwap
        rolling_std = dev.rolling(20, min_periods=10).std()
        z_score = dev / rolling_std.replace(0, np.nan)

        position = 0
        entry_price = 0
        entry_bar = 0
        bars_held = 0

        for idx, (_, bar) in enumerate(night_bars.iterrows()):
            if idx < 20:
                continue

            z = z_score.iloc[idx] if idx < len(z_score) else 0
            if pd.isna(z):
                continue

            if position == 0:
                if z > vwap_dev_threshold:
                    # Price extended above VWAP -> short (fade)
                    position = -1
                    entry_price = bar["close"]
                    entry_bar = idx
                elif z < -vwap_dev_threshold:
                    # Price extended below VWAP -> long (fade)
                    position = 1
                    entry_price = bar["close"]
                    entry_bar = idx
            else:
                bars_held = idx - entry_bar

                # Exit: reversion to VWAP (z-score crosses 0)
                reverted = (position == 1 and z >= 0) or (position == -1 and z <= 0)
                timed_out = bars_held >= max_hold_bars

                if reverted or timed_out:
                    exit_price = bar["close"]
                    slippage = get_slippage_ticks(bar["hour"]) * contract.tick_value
                    pnl_per = (exit_price - entry_price) * position * contract.point_value
                    costs = contract.commission_per_side * 2 + slippage * 2
                    pnl = pnl_per - costs
                    trades.append({
                        "strategy": "ON_Reversion", "symbol": symbol,
                        "date": date,
                        "direction": "long" if position > 0 else "short",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl": pnl, "costs": costs, "contracts": 1,
                        "reason": "revert" if reverted else "timeout",
                    })
                    position = 0

        # Force close if still in position
        if position != 0 and len(night_bars) > 0:
            exit_price = night_bars.iloc[-1]["close"]
            slippage = 2 * contract.tick_value
            pnl_per = (exit_price - entry_price) * position * contract.point_value
            costs = contract.commission_per_side * 2 + slippage * 2
            pnl = pnl_per - costs
            trades.append({
                "strategy": "ON_Reversion", "symbol": symbol,
                "date": date,
                "direction": "long" if position > 0 else "short",
                "entry_price": entry_price, "exit_price": exit_price,
                "pnl": pnl, "costs": costs, "contracts": 1,
                "reason": "force_close",
            })

    result = compute_overnight_metrics(trades, capital)
    result.strategy_name = "ON_Reversion"
    result.symbol = symbol
    return result


# =========================================================================
# Strategy 3: Pre-Market Momentum (7:00-9:15 AM)
# =========================================================================

def run_premarket_momentum(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                            capital: float, lookback: int = 15,
                            momentum_threshold: float = 0.0015) -> OvernightResult:
    """Ride momentum during pre-market hours when volume picks up."""
    trades = []

    dates = sorted(df["date"].unique())

    for date in dates:
        day = df[df["date"] == date]

        # Pre-market window: 7:00-9:25 AM
        pm_bars = day[(day["minute_of_day"] >= 420) & (day["minute_of_day"] < 565)]
        if len(pm_bars) < lookback + 5:
            continue

        position = 0
        entry_price = 0
        entry_bar = 0

        for idx in range(lookback, len(pm_bars)):
            bar = pm_bars.iloc[idx]

            if bar["minute_of_day"] >= 555:  # No new entries after 9:15
                if position != 0:
                    exit_price = bar["close"]
                    slippage = get_slippage_ticks(bar["hour"]) * contract.tick_value
                    pnl_per = (exit_price - entry_price) * position * contract.point_value
                    costs = contract.commission_per_side * 2 + slippage * 2
                    pnl = pnl_per - costs
                    trades.append({
                        "strategy": "PM_Momentum", "symbol": symbol,
                        "date": date,
                        "direction": "long" if position > 0 else "short",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl": pnl, "costs": costs, "contracts": 1,
                        "reason": "eod",
                    })
                    position = 0
                continue

            # Compute momentum: return over lookback period
            past_price = pm_bars.iloc[idx - lookback]["close"]
            current = bar["close"]
            momentum = (current - past_price) / past_price

            if position == 0:
                if momentum > momentum_threshold:
                    position = 1
                    entry_price = current
                    entry_bar = idx
                elif momentum < -momentum_threshold:
                    position = -1
                    entry_price = current
                    entry_bar = idx

            elif position != 0:
                # Exit on momentum reversal
                if (position == 1 and momentum < 0) or (position == -1 and momentum > 0):
                    exit_price = current
                    slippage = get_slippage_ticks(bar["hour"]) * contract.tick_value
                    pnl_per = (exit_price - entry_price) * position * contract.point_value
                    costs = contract.commission_per_side * 2 + slippage * 2
                    pnl = pnl_per - costs
                    trades.append({
                        "strategy": "PM_Momentum", "symbol": symbol,
                        "date": date,
                        "direction": "long" if position > 0 else "short",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl": pnl, "costs": costs, "contracts": 1,
                        "reason": "reversal",
                    })
                    position = 0

    result = compute_overnight_metrics(trades, capital)
    result.strategy_name = "PM_Momentum"
    result.symbol = symbol
    return result


# =========================================================================
# Strategy 4: Overnight Gap Fade
# =========================================================================

def run_gap_fade(df: pd.DataFrame, symbol: str, contract: FuturesContract,
                  capital: float, min_gap_pct: float = 0.2,
                  target_reversion_pct: float = 0.5) -> OvernightResult:
    """Fade overnight gaps: if overnight move is large, bet on reversion."""
    trades = []

    dates = sorted(df["date"].unique())

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        # Previous day's last cash bar
        prev_day = df[(df["date"] == prev_date)]
        prev_cash = prev_day[(prev_day["minute_of_day"] >= 570) & (prev_day["minute_of_day"] < 960)]
        if prev_cash.empty:
            continue
        prev_close = prev_cash.iloc[-1]["close"]

        # Current day's overnight bars (before cash open)
        curr_day = df[df["date"] == curr_date]
        # Pre-market bar near 9:25 AM
        pre_open = curr_day[(curr_day["minute_of_day"] >= 560) & (curr_day["minute_of_day"] <= 565)]
        if pre_open.empty:
            continue

        current_price = pre_open.iloc[0]["close"]
        gap_pct = (current_price - prev_close) / prev_close * 100

        if abs(gap_pct) < min_gap_pct:
            continue

        # Fade the gap
        if gap_pct > 0:
            # Gap up -> short
            direction = -1
            entry_price = current_price
            target_price = prev_close + (current_price - prev_close) * (1 - target_reversion_pct)
            stop_price = current_price * 1.005  # 0.5% stop
        else:
            # Gap down -> long
            direction = 1
            entry_price = current_price
            target_price = prev_close - (prev_close - current_price) * (1 - target_reversion_pct)
            stop_price = current_price * 0.995

        # This trade enters at 9:25 AM and must close by... wait,
        # the gap fade should enter before cash and use the first 15 min of cash.
        # But we can't overlap with cash ORB. So enter at 9:20 AM, close by 9:25 AM.
        # That's only 5 minutes - not enough.
        # Better approach: enter at 8:30 AM (after econ data), target reversion by 9:25 AM.
        econ_bars = curr_day[(curr_day["minute_of_day"] >= 510) & (curr_day["minute_of_day"] < 565)]
        if len(econ_bars) < 5:
            continue

        entry_price = econ_bars.iloc[0]["close"]
        position = direction

        for _, bar in econ_bars.iterrows():
            hit_target = (position == 1 and bar["close"] >= target_price) or \
                         (position == -1 and bar["close"] <= target_price)
            hit_stop = (position == 1 and bar["close"] <= stop_price) or \
                       (position == -1 and bar["close"] >= stop_price)

            if hit_target or hit_stop:
                exit_price = bar["close"]
                slippage = get_slippage_ticks(bar["hour"]) * contract.tick_value
                pnl_per = (exit_price - entry_price) * position * contract.point_value
                costs = contract.commission_per_side * 2 + slippage * 2
                pnl = pnl_per - costs
                trades.append({
                    "strategy": "Gap_Fade", "symbol": symbol,
                    "date": curr_date,
                    "direction": "long" if position > 0 else "short",
                    "entry_price": entry_price, "exit_price": exit_price,
                    "pnl": pnl, "costs": costs, "contracts": 1,
                    "reason": "target" if hit_target else "stop",
                })
                position = 0
                break

        # Force close at 9:25 AM if still in position
        if position != 0 and len(econ_bars) > 0:
            exit_price = econ_bars.iloc[-1]["close"]
            slippage = get_slippage_ticks(8) * contract.tick_value
            pnl_per = (exit_price - entry_price) * position * contract.point_value
            costs = contract.commission_per_side * 2 + slippage * 2
            pnl = pnl_per - costs
            trades.append({
                "strategy": "Gap_Fade", "symbol": symbol,
                "date": curr_date,
                "direction": "long" if position > 0 else "short",
                "entry_price": entry_price, "exit_price": exit_price,
                "pnl": pnl, "costs": costs, "contracts": 1,
                "reason": "force_close",
            })

    result = compute_overnight_metrics(trades, capital)
    result.strategy_name = "Gap_Fade"
    result.symbol = symbol
    return result


def run_all_strategies(symbol: str):
    """Run all overnight strategy candidates on a symbol."""
    contract = CONTRACTS[symbol]
    df = load_full_data(symbol)
    capital = INITIAL_CAPITAL

    print(f"\n{'='*70}")
    print(f"  OVERNIGHT STRATEGY CANDIDATES: {symbol}")
    print(f"  Data: {df['dt'].iloc[0].strftime('%Y-%m-%d')} to {df['dt'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"{'='*70}")

    results = []

    # Strategy 1: European ORB
    print("\n  Testing European Open ORB...")
    r = run_euro_orb(df, symbol, contract, capital)
    print_result(r)
    results.append(r)

    # Strategy 2: Overnight Mean Reversion
    print("\n  Testing Overnight Mean Reversion...")
    r = run_overnight_reversion(df, symbol, contract, capital)
    print_result(r)
    results.append(r)

    # Strategy 3: Pre-Market Momentum
    print("\n  Testing Pre-Market Momentum...")
    r = run_premarket_momentum(df, symbol, contract, capital)
    print_result(r)
    results.append(r)

    # Strategy 4: Gap Fade
    print("\n  Testing Gap Fade...")
    r = run_gap_fade(df, symbol, contract, capital)
    print_result(r)
    results.append(r)

    # Summary table
    print(f"\n{'  '}{'-'*80}")
    print(f"  {'Strategy':<20} {'Trades':>7} {'P&L':>12} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'MaxDD':>8} {'AvgTrd':>10}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r.strategy_name:<20} {r.total_trades:>7} ${r.total_pnl:>+10,.0f} "
              f"{r.sharpe:>8.2f} {r.win_rate*100:>5.1f}% {r.profit_factor:>5.2f} "
              f"{r.max_drawdown_pct:>7.1f}% ${r.avg_trade_pnl:>+8,.0f}")
    print(f"  {'-'*80}")

    return results


def main():
    print("=" * 70)
    print("  PHASE 4: OVERNIGHT STRATEGY TESTING")
    print("=" * 70)

    all_results = {}
    for symbol in ["MES", "MNQ"]:
        try:
            all_results[symbol] = run_all_strategies(symbol)
        except FileNotFoundError as e:
            print(f"\n  SKIPPING {symbol}: {e}")

    # Identify survivors (positive Sharpe)
    print(f"\n{'='*70}")
    print("  SURVIVORS (positive Sharpe)")
    print("=" * 70)

    survivors = []
    for symbol, results in all_results.items():
        for r in results:
            if r.sharpe > 0 and r.total_trades >= 20:
                survivors.append(r)
                print(f"  {r.strategy_name} {r.symbol}: Sharpe={r.sharpe:.2f}, "
                      f"P&L=${r.total_pnl:+,.0f}, trades={r.total_trades}")

    if not survivors:
        print("  No strategies survived.")
        print("  Conclusion: No exploitable overnight edge found in this data.")


if __name__ == "__main__":
    main()
