#!/usr/bin/env python3
"""Validate VWAP Reversion survivor with locked OOS and parameter sensitivity.

Final gate before implementation.
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.data.contracts import CONTRACTS
from trading.config import INITIAL_CAPITAL
from analysis.intraday_strategies import (
    load_cash_data, run_vwap_reversion, compute_metrics, check_correlation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


def locked_oos_test(df, symbol, contract, capital):
    """Final locked OOS: train on first 80%, test on last 20%."""
    print(f"\n  --- Locked OOS Test ({symbol}) ---")

    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.8)
    train_dates = set(dates[:split])
    test_dates = set(dates[split:])

    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    print(f"  Train: {len(train_dates)} days ({dates[0]} to {dates[split-1]})")
    print(f"  Test:  {len(test_dates)} days ({dates[split]} to {dates[-1]})")

    # Optimize on train
    param_grid = [
        {"z_entry": 0.6, "max_hold": 60},
        {"z_entry": 0.8, "max_hold": 60},
        {"z_entry": 1.0, "max_hold": 30},
        {"z_entry": 1.0, "max_hold": 60},
        {"z_entry": 1.0, "max_hold": 90},
        {"z_entry": 1.2, "max_hold": 60},
        {"z_entry": 1.5, "max_hold": 60},
    ]

    best_sharpe = -999
    best_params = {"z_entry": 1.0, "max_hold": 60}

    print(f"\n  Training (testing {len(param_grid)} param combos)...")
    for params in param_grid:
        r = run_vwap_reversion(train_df, symbol, contract, capital, **params)
        print(f"    z={params['z_entry']}, hold={params['max_hold']}: "
              f"{r.trades} trades, ${r.total_pnl:+,.0f}, Sharpe={r.sharpe:.2f}")
        if r.trades >= 20 and r.sharpe > best_sharpe:
            best_sharpe = r.sharpe
            best_params = params

    print(f"\n  Best train params: z={best_params['z_entry']}, hold={best_params['max_hold']} "
          f"(Sharpe={best_sharpe:.2f})")

    # Test on locked OOS
    print(f"\n  Locked OOS results:")
    oos_r = run_vwap_reversion(test_df, symbol, contract, capital, **best_params)

    print(f"  Trades:       {oos_r.trades}")
    print(f"  Total P&L:    ${oos_r.total_pnl:+,.0f}")
    print(f"  Sharpe:       {oos_r.sharpe:.2f}")
    print(f"  Max Drawdown: {oos_r.max_dd_pct:.1f}%")
    print(f"  Win Rate:     {oos_r.win_rate*100:.1f}%")
    print(f"  Profit Factor:{oos_r.profit_factor:.2f}")
    print(f"  Avg Trade:    ${oos_r.avg_trade:+,.0f}")

    # Performance by hour
    if oos_r.trades_list:
        print(f"\n  Performance by hour:")
        by_hour = {}
        for t in oos_r.trades_list:
            h = t.get("entry_hour", 12)
            by_hour.setdefault(h, []).append(t["pnl"])
        for h in sorted(by_hour.keys()):
            pnls = by_hour[h]
            print(f"    {h:02d}:00  trades={len(pnls):>4}  P&L=${sum(pnls):>+8,.0f}  "
                  f"avg=${np.mean(pnls):>+6,.0f}  wr={sum(1 for x in pnls if x > 0)/len(pnls)*100:.0f}%"
                  if len(pnls) > 0 else f"    {h:02d}:00  no trades")

    # ORB correlation
    corr = check_correlation(oos_r.trades_list, test_df, symbol)
    if corr is not None:
        print(f"\n  ORB correlation: {corr:.4f}")

    passed = oos_r.sharpe > 0 and oos_r.total_pnl > 0 and oos_r.trades >= 20
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "best_params": best_params,
        "oos_result": oos_r,
        "oos_correlation": corr,
    }


def parameter_sensitivity(df, symbol, contract, capital):
    """How stable is VWAP reversion across parameter space?"""
    print(f"\n  --- Parameter Sensitivity ({symbol}) ---")

    z_vals = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    hold_vals = [20, 40, 60, 80, 100]

    results = []
    print(f"  {'z_entry':>7} {'hold':>5} {'trades':>7} {'P&L':>12} {'Sharpe':>8} {'WR':>6} {'PF':>6}")
    print(f"  {'-'*55}")

    for z in z_vals:
        for hold in hold_vals:
            r = run_vwap_reversion(df, symbol, contract, capital, z_entry=z, max_hold=hold)
            results.append({"z": z, "hold": hold, "trades": r.trades,
                           "pnl": r.total_pnl, "sharpe": r.sharpe,
                           "wr": r.win_rate, "pf": r.profit_factor})
            if hold in [20, 60, 100]:
                print(f"  {z:>7.1f} {hold:>5} {r.trades:>7} ${r.total_pnl:>+10,.0f} "
                      f"{r.sharpe:>8.2f} {r.win_rate*100:>5.1f}% {r.profit_factor:>5.2f}")

    profitable = sum(1 for r in results if r["pnl"] > 0)
    total = len(results)
    median_sharpe = np.median([r["sharpe"] for r in results])

    print(f"\n  Profitable: {profitable}/{total} ({profitable/total*100:.0f}%)")
    print(f"  Median Sharpe: {median_sharpe:.2f}")

    robust = profitable / total >= 0.5
    print(f"  VERDICT: {'ROBUST' if robust else 'FRAGILE'}")
    return robust


def main():
    print("=" * 70)
    print("  VWAP REVERSION FINAL VALIDATION")
    print("=" * 70)

    all_pass = True

    for symbol in ["MES", "MNQ"]:
        contract = CONTRACTS[symbol]
        df = load_cash_data(symbol)
        capital = INITIAL_CAPITAL

        print(f"\n{'='*70}")
        print(f"  {symbol}")
        print(f"{'='*70}")

        # Locked OOS
        oos = locked_oos_test(df, symbol, contract, capital)
        if not oos["passed"]:
            all_pass = False

        # Param sensitivity
        robust = parameter_sensitivity(df, symbol, contract, capital)
        if not robust:
            all_pass = False

    print(f"\n{'='*70}")
    print(f"  FINAL VERDICT: {'ALL PASS - IMPLEMENT' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'='*70}")

    return all_pass


if __name__ == "__main__":
    main()
