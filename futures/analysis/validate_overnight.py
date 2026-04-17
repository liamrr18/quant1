#!/usr/bin/env python3
"""Phase 5: Validate the ON_Reversion MNQ survivor from Phase 4.

Tests:
1. Walk-forward OOS: 6-month IS / 3-month OOS rolling windows
2. Parameter sensitivity: vary z-score threshold and max hold bars
3. Correlation with cash ORB: are returns correlated?
4. Combined portfolio: what does adding ON_Reversion do to overall metrics?
"""

import os
import sys
import logging
from itertools import product

import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.data.contracts import CONTRACTS
from trading.config import INITIAL_CAPITAL
from analysis.overnight_strategies import (
    load_full_data, run_overnight_reversion, compute_overnight_metrics,
    get_slippage_ticks, OvernightResult
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")


# =========================================================================
# Test 1: Walk-Forward Out-of-Sample
# =========================================================================

def walk_forward_oos(df: pd.DataFrame, symbol: str, contract, capital: float,
                     is_months: int = 6, oos_months: int = 3):
    """Walk-forward: optimize on IS window, test on OOS window."""
    print(f"\n  --- Walk-Forward OOS ({symbol}) ---")
    print(f"  IS={is_months}mo, OOS={oos_months}mo")

    dates = sorted(df["date"].unique())
    first = pd.Timestamp(dates[0])
    last = pd.Timestamp(dates[-1])

    # Generate windows
    windows = []
    is_start = first
    while True:
        is_end = is_start + pd.DateOffset(months=is_months)
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        if oos_end > last:
            break
        windows.append((is_start, is_end, oos_start, oos_end))
        is_start = is_start + pd.DateOffset(months=oos_months)  # Roll forward

    if not windows:
        print("  Not enough data for walk-forward windows")
        return None

    print(f"  Windows: {len(windows)}")

    # Parameter grid for IS optimization
    z_thresholds = [1.5, 2.0, 2.5, 3.0]
    max_hold_bars_opts = [30, 60, 90]

    all_oos_trades = []
    window_results = []

    for w_idx, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        is_dates = set(d for d in dates if is_s.date() <= d <= is_e.date())
        oos_dates = set(d for d in dates if oos_s.date() <= d <= oos_e.date())

        is_df = df[df["date"].isin(is_dates)].copy()
        oos_df = df[df["date"].isin(oos_dates)].copy()

        if len(is_df) < 100 or len(oos_df) < 50:
            continue

        # Optimize on IS
        best_sharpe = -999
        best_params = (2.0, 60)

        for z_thresh, max_hold in product(z_thresholds, max_hold_bars_opts):
            r = run_overnight_reversion(is_df, symbol, contract, capital,
                                        vwap_dev_threshold=z_thresh,
                                        max_hold_bars=max_hold)
            if r.total_trades >= 10 and r.sharpe > best_sharpe:
                best_sharpe = r.sharpe
                best_params = (z_thresh, max_hold)

        # Run best params on OOS
        oos_result = run_overnight_reversion(oos_df, symbol, contract, capital,
                                             vwap_dev_threshold=best_params[0],
                                             max_hold_bars=best_params[1])

        window_results.append({
            "window": w_idx + 1,
            "is_period": f"{is_s.strftime('%Y-%m')} to {is_e.strftime('%Y-%m')}",
            "oos_period": f"{oos_s.strftime('%Y-%m')} to {oos_e.strftime('%Y-%m')}",
            "is_sharpe": best_sharpe,
            "best_z": best_params[0],
            "best_hold": best_params[1],
            "oos_trades": oos_result.total_trades,
            "oos_pnl": oos_result.total_pnl,
            "oos_sharpe": oos_result.sharpe,
            "oos_wr": oos_result.win_rate,
            "oos_pf": oos_result.profit_factor,
        })

        all_oos_trades.extend(oos_result.trades_list)

        print(f"  W{w_idx+1}: IS={best_sharpe:+.2f} (z={best_params[0]}, hold={best_params[1]}) "
              f"-> OOS: {oos_result.total_trades} trades, ${oos_result.total_pnl:+,.0f}, "
              f"Sharpe={oos_result.sharpe:.2f}")

    if not window_results:
        print("  No valid walk-forward windows")
        return None

    # Aggregate OOS results
    total_oos_pnl = sum(w["oos_pnl"] for w in window_results)
    avg_oos_sharpe = np.mean([w["oos_sharpe"] for w in window_results])
    oos_positive = sum(1 for w in window_results if w["oos_pnl"] > 0)

    print(f"\n  Walk-Forward Summary:")
    print(f"  Total OOS P&L:    ${total_oos_pnl:+,.0f}")
    print(f"  Avg OOS Sharpe:   {avg_oos_sharpe:.2f}")
    print(f"  OOS windows +ve:  {oos_positive}/{len(window_results)}")
    print(f"  Total OOS trades: {len(all_oos_trades)}")

    passed = avg_oos_sharpe > 0 and oos_positive >= len(window_results) * 0.5
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    return {
        "windows": window_results,
        "total_oos_pnl": total_oos_pnl,
        "avg_oos_sharpe": avg_oos_sharpe,
        "oos_positive_pct": oos_positive / len(window_results),
        "passed": passed,
        "all_oos_trades": all_oos_trades,
    }


# =========================================================================
# Test 2: Parameter Sensitivity
# =========================================================================

def parameter_sensitivity(df: pd.DataFrame, symbol: str, contract, capital: float):
    """Test how sensitive results are to parameter changes."""
    print(f"\n  --- Parameter Sensitivity ({symbol}) ---")

    z_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    max_hold_bars_opts = [20, 40, 60, 80, 100, 120]

    results = []
    print(f"  {'z_thresh':>8} {'max_hold':>8} {'trades':>7} {'P&L':>12} {'Sharpe':>8} {'WR':>6} {'PF':>6}")
    print(f"  {'-'*60}")

    for z_thresh in z_thresholds:
        for max_hold in max_hold_bars_opts:
            r = run_overnight_reversion(df, symbol, contract, capital,
                                        vwap_dev_threshold=z_thresh,
                                        max_hold_bars=max_hold)
            results.append({
                "z_thresh": z_thresh,
                "max_hold": max_hold,
                "trades": r.total_trades,
                "pnl": r.total_pnl,
                "sharpe": r.sharpe,
                "wr": r.win_rate,
                "pf": r.profit_factor,
            })

    # Print a subset (every other row)
    for r in results:
        if r["max_hold"] in [20, 60, 120]:
            print(f"  {r['z_thresh']:>8.1f} {r['max_hold']:>8} {r['trades']:>7} "
                  f"${r['pnl']:>+10,.0f} {r['sharpe']:>8.2f} "
                  f"{r['wr']*100:>5.1f}% {r['pf']:>5.2f}")

    # Check stability: how many combos are profitable?
    profitable = sum(1 for r in results if r["pnl"] > 0)
    positive_sharpe = sum(1 for r in results if r["sharpe"] > 0)
    total = len(results)

    print(f"\n  Stability Summary:")
    print(f"  Profitable combos: {profitable}/{total} ({profitable/total*100:.0f}%)")
    print(f"  Positive Sharpe:   {positive_sharpe}/{total} ({positive_sharpe/total*100:.0f}%)")

    # Is the signal robust or a narrow peak?
    sharpe_vals = [r["sharpe"] for r in results]
    best_sharpe = max(sharpe_vals)
    median_sharpe = np.median(sharpe_vals)

    print(f"  Best Sharpe:       {best_sharpe:.2f}")
    print(f"  Median Sharpe:     {median_sharpe:.2f}")

    robust = profitable / total >= 0.3 and median_sharpe > -0.5
    print(f"  VERDICT: {'ROBUST' if robust else 'FRAGILE'} parameter space")

    return {"results": results, "profitable_pct": profitable / total,
            "median_sharpe": median_sharpe, "robust": robust}


# =========================================================================
# Test 3: Correlation with Cash ORB
# =========================================================================

def correlation_with_cash_orb(on_trades: list, symbol: str):
    """Check if overnight reversion returns are correlated with cash ORB."""
    print(f"\n  --- Correlation with Cash ORB ({symbol}) ---")

    # Load cash ORB equity curve if available
    # We'll compute daily ON P&L and compare with cash-session returns
    cash_path = os.path.join(CACHE_DIR, f"{symbol}_futures_cash_1min.csv")
    if not os.path.exists(cash_path):
        print("  No cash data available for correlation")
        return None

    cash_df = pd.read_csv(cash_path)
    cash_df["dt"] = pd.to_datetime(cash_df["dt"], utc=True).dt.tz_convert(ET)
    cash_df["date"] = cash_df["dt"].dt.date

    # Cash daily returns
    cash_daily = cash_df.groupby("date").agg(
        open=("open", "first"),
        close=("close", "last")
    )
    cash_daily["cash_return"] = (cash_daily["close"] - cash_daily["open"]) / cash_daily["open"]

    # ON daily P&L
    on_daily = {}
    for t in on_trades:
        d = t["date"]
        on_daily[d] = on_daily.get(d, 0) + t["pnl"]

    on_series = pd.Series(on_daily, name="on_pnl")

    # Merge
    merged = cash_daily.join(on_series, how="inner")
    merged = merged.dropna(subset=["cash_return", "on_pnl"])

    if len(merged) < 20:
        print(f"  Only {len(merged)} overlapping days — insufficient")
        return None

    corr = merged["cash_return"].corr(merged["on_pnl"])
    print(f"  Overlapping days:    {len(merged)}")
    print(f"  Correlation:         {corr:.4f}")

    # Are they diversifying?
    if abs(corr) < 0.2:
        print(f"  VERDICT: LOW correlation — good diversifier")
    elif corr > 0.2:
        print(f"  VERDICT: POSITIVE correlation — not diversifying")
    else:
        print(f"  VERDICT: NEGATIVE correlation — anti-correlated (hedging)")

    return {"correlation": corr, "n_days": len(merged)}


# =========================================================================
# Test 4: Combined Portfolio
# =========================================================================

def combined_portfolio_analysis(on_trades: list, symbol: str, capital: float):
    """What does the combined cash ORB + overnight reversion portfolio look like?"""
    print(f"\n  --- Combined Portfolio Analysis ({symbol}) ---")

    cash_path = os.path.join(CACHE_DIR, f"{symbol}_futures_cash_1min.csv")
    if not os.path.exists(cash_path):
        print("  No cash data available")
        return None

    cash_df = pd.read_csv(cash_path)
    cash_df["dt"] = pd.to_datetime(cash_df["dt"], utc=True).dt.tz_convert(ET)
    cash_df["date"] = cash_df["dt"].dt.date

    # Cash daily returns (as proxy for cash ORB P&L)
    cash_daily = cash_df.groupby("date").agg(
        open=("open", "first"),
        close=("close", "last")
    )
    cash_daily["cash_pnl"] = (cash_daily["close"] - cash_daily["open"]) * CONTRACTS[symbol].point_value

    # ON daily P&L
    on_daily = {}
    for t in on_trades:
        d = t["date"]
        on_daily[d] = on_daily.get(d, 0) + t["pnl"]
    on_series = pd.Series(on_daily, name="on_pnl")

    # Merge
    merged = cash_daily.join(on_series, how="outer").fillna(0)

    # Combined
    merged["combined_pnl"] = merged["cash_pnl"] + merged["on_pnl"]

    # Metrics
    for col, label in [("cash_pnl", "Cash Only"), ("on_pnl", "ON Reversion Only"), ("combined_pnl", "Combined")]:
        series = merged[col]
        total = series.sum()
        daily_mean = series.mean()
        daily_std = series.std()
        sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0

        # Max drawdown
        cumsum = series.cumsum()
        peak = cumsum.cummax()
        dd = peak - cumsum
        max_dd = dd.max()

        print(f"\n  {label}:")
        print(f"    Total P&L:      ${total:+,.0f}")
        print(f"    Daily Sharpe:   {sharpe:.2f}")
        print(f"    Max Drawdown:   ${max_dd:,.0f}")
        print(f"    Days traded:    {(series != 0).sum()}")

    return merged


def main():
    print("=" * 70)
    print("  PHASE 5: VALIDATE ON_REVERSION MNQ SURVIVOR")
    print("=" * 70)

    symbol = "MNQ"
    contract = CONTRACTS[symbol]
    capital = INITIAL_CAPITAL

    df = load_full_data(symbol)
    print(f"  Data: {df['dt'].iloc[0].strftime('%Y-%m-%d')} to {df['dt'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Bars: {len(df):,}")

    # Test 1: Walk-forward OOS
    wf_result = walk_forward_oos(df, symbol, contract, capital)

    # Test 2: Parameter sensitivity
    sensitivity = parameter_sensitivity(df, symbol, contract, capital)

    # Get trades for correlation/portfolio tests
    full_result = run_overnight_reversion(df, symbol, contract, capital)

    # Test 3: Correlation with cash
    corr_result = correlation_with_cash_orb(full_result.trades_list, symbol)

    # Test 4: Combined portfolio
    portfolio = combined_portfolio_analysis(full_result.trades_list, symbol, capital)

    # Final verdict
    print(f"\n{'='*70}")
    print("  PHASE 5 FINAL VERDICT")
    print("=" * 70)

    checks = {
        "Walk-Forward OOS": wf_result["passed"] if wf_result else False,
        "Parameter Robustness": sensitivity["robust"] if sensitivity else False,
        "Low Correlation": abs(corr_result["correlation"]) < 0.3 if corr_result else False,
    }

    all_pass = all(checks.values())

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check:<25} {status}")

    if all_pass:
        print(f"\n  CONCLUSION: ON_Reversion MNQ PASSES all validation checks.")
        print(f"  Proceed to Phase 6: implementation and paper trading.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n  CONCLUSION: ON_Reversion MNQ FAILS validation.")
        print(f"  Failed checks: {', '.join(failed)}")
        print(f"  Do NOT proceed to live implementation.")

    return all_pass


if __name__ == "__main__":
    main()
