#!/usr/bin/env python3
"""Phase 2: Re-validate cash-hours ORB on real MES/MNQ futures data.

Compares backtest results using:
  1. SPY/QQQ proxy data (original backtest)
  2. Real MES/MNQ futures data from IB

Same strategy parameters, same walk-forward framework, same costs.
This is a confidence check: does the edge hold on real futures bars?
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.config import INITIAL_CAPITAL, ORB_SHARED_DEFAULTS, SYMBOL_PROFILES
from trading.strategies.orb import ORBBreakout
from trading.data.features import prepare_features
from trading.data.contracts import CONTRACTS, MES, MNQ
from trading.backtest.engine import run_backtest, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
EQUITY_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
    "spy-trader", ".claude", "worktrees", "flamboyant-lewin", "data", "cache"
)


def load_proxy_data(symbol: str) -> pd.DataFrame:
    """Load SPY/QQQ proxy data (same as original backtest)."""
    proxy_map = {"MES": "SPY", "MNQ": "QQQ"}
    proxy = proxy_map[symbol]

    # Find the cached file
    import glob
    pattern = os.path.join(EQUITY_CACHE_DIR, f"{proxy}_1min_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No proxy data found: {pattern}")

    path = sorted(files)[-1]  # Most recent
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)

    # Filter to cash hours only
    times = df["dt"].dt.strftime("%H:%M")
    df = df[(times >= "09:30") & (times < "16:00")]
    df = df.sort_values("dt").reset_index(drop=True)

    log.info("Loaded %d proxy bars for %s (%s) from %s", len(df), symbol, proxy, path)
    return df


def load_futures_data(symbol: str) -> pd.DataFrame:
    """Load real MES/MNQ cash-hours data from IB download."""
    path = os.path.join(CACHE_DIR, f"{symbol}_futures_cash_1min.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No futures data found: {path}")

    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)

    # Ensure cash hours only
    times = df["dt"].dt.strftime("%H:%M")
    df = df[(times >= "09:30") & (times < "16:00")]
    df = df.sort_values("dt").reset_index(drop=True)

    log.info("Loaded %d futures bars for %s from %s", len(df), symbol, path)
    return df


def make_strategy(symbol: str) -> ORBBreakout:
    """Create ORB strategy with per-symbol profile."""
    params = dict(ORB_SHARED_DEFAULTS)
    if symbol in SYMBOL_PROFILES:
        params.update(SYMBOL_PROFILES[symbol])
    return ORBBreakout(**params)


def run_comparison(symbol: str):
    """Run backtest on both proxy and real data, compare results."""
    contract = CONTRACTS[symbol]
    strat = make_strategy(symbol)

    print(f"\n{'='*70}")
    print(f"  CASH ORB VALIDATION: {symbol} ({contract.name})")
    print(f"{'='*70}")

    # Load both datasets
    proxy_df = load_proxy_data(symbol)
    futures_df = load_futures_data(symbol)

    # Find overlapping date range
    proxy_dates = set(proxy_df["dt"].dt.date.unique())
    futures_dates = set(futures_df["dt"].dt.date.unique())
    common_dates = sorted(proxy_dates & futures_dates)

    if not common_dates:
        print("  ERROR: No overlapping dates between proxy and futures data")
        return

    start_date = common_dates[0]
    end_date = common_dates[-1]
    print(f"  Overlapping period: {start_date} to {end_date} ({len(common_dates)} days)")

    # Filter both to common date range
    proxy_df = proxy_df[proxy_df["dt"].dt.date.isin(common_dates)].reset_index(drop=True)
    futures_df = futures_df[futures_df["dt"].dt.date.isin(common_dates)].reset_index(drop=True)

    print(f"  Proxy bars:   {len(proxy_df):,}")
    print(f"  Futures bars: {len(futures_df):,}")

    # Prepare features and generate signals
    proxy_df = prepare_features(proxy_df)
    futures_df = prepare_features(futures_df)
    proxy_df = strat.generate_signals(proxy_df)
    futures_df = strat.generate_signals(futures_df)

    # Run backtests
    print(f"\n  Running proxy backtest ({symbol} via {'SPY' if symbol == 'MES' else 'QQQ'})...")
    proxy_result = run_backtest(proxy_df, strat, symbol, INITIAL_CAPITAL, contract=contract)

    print(f"  Running futures backtest ({symbol} real IB data)...")
    futures_result = run_backtest(futures_df, strat, symbol, INITIAL_CAPITAL, contract=contract)

    # Compare results
    print(f"\n{'  '}{'-'*56}")
    print(f"  {'Metric':<25} {'Proxy':>15} {'Futures':>15}")
    print(f"  {'-'*56}")

    comparisons = [
        ("Trades", proxy_result.num_trades, futures_result.num_trades),
        ("Total P&L", f"${proxy_result.total_pnl:+,.0f}", f"${futures_result.total_pnl:+,.0f}"),
        ("Total Return", f"{proxy_result.total_return*100:+.1f}%", f"{futures_result.total_return*100:+.1f}%"),
        ("Sharpe Ratio", f"{proxy_result.sharpe_ratio:.2f}", f"{futures_result.sharpe_ratio:.2f}"),
        ("Max Drawdown", f"{proxy_result.max_drawdown*100:.1f}%", f"{futures_result.max_drawdown*100:.1f}%"),
        ("Win Rate", f"{proxy_result.win_rate*100:.1f}%", f"{futures_result.win_rate*100:.1f}%"),
        ("Profit Factor", f"{proxy_result.profit_factor:.2f}", f"{futures_result.profit_factor:.2f}"),
        ("Avg Trade", f"${proxy_result.avg_trade_dollars:+,.0f}", f"${futures_result.avg_trade_dollars:+,.0f}"),
        ("Avg Contracts", f"{proxy_result.avg_contracts:.1f}", f"{futures_result.avg_contracts:.1f}"),
        ("Total Costs", f"${proxy_result.total_costs:,.0f}", f"${futures_result.total_costs:,.0f}"),
    ]

    for label, proxy_val, futures_val in comparisons:
        print(f"  {label:<25} {str(proxy_val):>15} {str(futures_val):>15}")

    print(f"  {'-'*56}")

    # Divergence check
    if proxy_result.total_pnl != 0:
        pnl_diff_pct = abs(futures_result.total_pnl - proxy_result.total_pnl) / abs(proxy_result.total_pnl) * 100
    else:
        pnl_diff_pct = 100 if futures_result.total_pnl != 0 else 0

    trade_diff = abs(futures_result.num_trades - proxy_result.num_trades)

    print(f"\n  P&L divergence: {pnl_diff_pct:.1f}%")
    print(f"  Trade count diff: {trade_diff}")

    if pnl_diff_pct <= 20:
        print(f"  VERDICT: Proxy was VALID (within 20% threshold)")
    else:
        print(f"  VERDICT: Proxy DIVERGED significantly (>{pnl_diff_pct:.0f}%)")
        print(f"  Investigation needed: check price differences, gaps, volume")

    return proxy_result, futures_result


def main():
    print("\n" + "=" * 70)
    print("  PHASE 2: CASH-HOURS ORB RE-VALIDATION")
    print("  Real MES/MNQ futures data vs SPY/QQQ proxy data")
    print("=" * 70)

    results = {}
    for symbol in ["MES", "MNQ"]:
        try:
            results[symbol] = run_comparison(symbol)
        except FileNotFoundError as e:
            print(f"\n  SKIPPING {symbol}: {e}")
            print(f"  Run download_ib_data.py first to get futures data.")

    if results:
        print(f"\n{'='*70}")
        print(f"  CONCLUSION")
        print(f"{'='*70}")
        for sym, (proxy_r, futures_r) in results.items():
            pnl_diff = abs(futures_r.total_pnl - proxy_r.total_pnl) / max(abs(proxy_r.total_pnl), 1) * 100
            status = "VALID" if pnl_diff <= 20 else "DIVERGED"
            print(f"  {sym}: proxy ${proxy_r.total_pnl:+,.0f} vs futures ${futures_r.total_pnl:+,.0f} "
                  f"({pnl_diff:.1f}% diff) -> {status}")


if __name__ == "__main__":
    main()
