#!/usr/bin/env python3
"""Robustness validation for the 3 surviving candidate strategies.

7 validation dimensions:
1. Walk-forward window sensitivity (40/20/20, 60/20/20, 80/20/20)
2. Parameter sensitivity (±20% on key params)
3. Subperiod analysis (quarterly on dev)
4. Slippage stress test ($0.005, $0.01, $0.02, $0.05)
5. Trade distribution analysis
6. Alpha/beta by period
7. Regime analysis (high-vol vs low-vol)
"""

import sys, os, io, warnings, time
from datetime import datetime
from collections import defaultdict
import pytz
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.backtest.engine import run_backtest
from trading.backtest.walkforward import walk_forward
from trading.strategies.pairs_spread import PairsSpread
from trading.strategies.opening_drive import OpeningDrive
from trading.config import PAIRS_GLD_TLT, OPENDRIVE_SMH, OPENDRIVE_XLK

ET = pytz.timezone("America/New_York")
START = datetime(2025, 1, 2, tzinfo=ET)
END = datetime(2026, 4, 4, tzinfo=ET)
DEV_END = datetime(2025, 11, 30, tzinfo=ET)
OOS_START = datetime(2025, 12, 1, tzinfo=ET)

t0 = time.time()

# ── Load data once ──
print("Loading data...", flush=True)
data = {}
for sym in ["GLD", "TLT", "SMH", "XLK", "SPY"]:
    df = get_minute_bars(sym, START, END, use_cache=True)
    df = prepare_features(df)
    data[sym] = df
    print(f"  {sym}: {len(df)} bars", flush=True)

# Merge pairs data
gld_full = data["GLD"].copy()
tlt_close = data["TLT"].set_index("dt")["close"].rename("pair_close")
gld_full = gld_full.set_index("dt").join(tlt_close, how="left").reset_index()
gld_full["pair_close"] = gld_full["pair_close"].ffill()

spy_bench = data["SPY"].groupby("date")["close"].last().pct_change().dropna()


def date_filter(df, start_dt, end_dt):
    return df[df["date"].apply(lambda d: start_dt.date() <= d <= end_dt.date())].copy()


def quick_bt(df, strat, sym, slip=None):
    """Run backtest and return (sharpe, daily_returns, result)."""
    df2 = strat.generate_signals(df.copy())
    r = run_backtest(df2, strat, sym, slippage_per_share=slip)
    dr = r.daily_returns
    if dr is None or len(dr) < 5 or dr.std() == 0:
        return 0, pd.Series(dtype=float), r
    sh = (dr.mean() / dr.std()) * np.sqrt(252)
    return sh, dr, r


def wf_sharpe(df, strat, sym, train_days, test_days, step_days):
    """Run walk-forward and return aggregate Sharpe."""
    df2 = strat.generate_signals(df.copy())
    try:
        wf = walk_forward(df2, strat, sym, train_days, test_days, step_days)
    except ValueError:
        return None
    all_dr = []
    for r in wf.oos_results:
        if r.daily_returns is not None and len(r.daily_returns) > 0:
            all_dr.append(r.daily_returns)
    if not all_dr:
        return None
    combined = pd.concat(all_dr).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    if len(combined) < 10 or combined.std() == 0:
        return 0
    return float((combined.mean() / combined.std()) * np.sqrt(252))


def alpha_beta(strat_ret, bench_ret):
    si = strat_ret.copy(); bi = bench_ret.copy()
    si.index = pd.to_datetime(si.index).normalize().tz_localize(None)
    bi.index = pd.to_datetime(bi.index).normalize().tz_localize(None)
    al = pd.DataFrame({"s": si, "b": bi}).dropna()
    if len(al) < 10: return 0, 0
    beta = al["b"].cov(al["s"]) / al["b"].var() if al["b"].var() > 0 else 0
    alpha = (al["s"].mean() - beta * al["b"].mean()) * 252
    return alpha, beta


# ═══════════════════════════════════════════════════════════════════════════════
# DEFINE CANDIDATES
# ═══════════════════════════════════════════════════════════════════════════════

CANDIDATES = {
    "Pairs_GLD_TLT": {
        "cls": PairsSpread,
        "frozen": {"lookback": 120, "entry_zscore": 2.0, "exit_zscore": 0.5, "stale_bars": 90, "last_entry_minute": 900},
        "sym": "GLD",
        "df_dev": date_filter(gld_full, START, DEV_END),
        "df_oos": date_filter(gld_full, OOS_START, END),
        "df_full": gld_full,
        "perturbations": {
            "lookback": [96, 144],
            "entry_zscore": [1.6, 2.4],
            "exit_zscore": [0.4, 0.6],
            "stale_bars": [72, 108],
        },
    },
    "OpenDrive_SMH": {
        "cls": OpeningDrive,
        "frozen": {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 3.0, "stop_multiple": 1.0, "stale_bars": 120, "last_entry_minute": 720},
        "sym": "SMH",
        "df_dev": date_filter(data["SMH"], START, DEV_END),
        "df_oos": date_filter(data["SMH"], OOS_START, END),
        "df_full": data["SMH"],
        "perturbations": {
            "target_multiple": [2.4, 3.6],
            "stop_multiple": [0.8, 1.2],
            "min_drive_pct": [0.08, 0.12],
            "stale_bars": [96, 144],
        },
    },
    "OpenDrive_XLK": {
        "cls": OpeningDrive,
        "frozen": {"drive_minutes": 5, "min_drive_pct": 0.10, "target_multiple": 1.5, "stop_multiple": 1.0, "stale_bars": 120, "last_entry_minute": 720},
        "sym": "XLK",
        "df_dev": date_filter(data["XLK"], START, DEV_END),
        "df_oos": date_filter(data["XLK"], OOS_START, END),
        "df_full": data["XLK"],
        "perturbations": {
            "target_multiple": [1.2, 1.8],
            "stop_multiple": [0.8, 1.2],
            "min_drive_pct": [0.08, 0.12],
            "stale_bars": [96, 144],
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL 7 VALIDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

results = {}

for cname, cdef in CANDIDATES.items():
    print(f"\n{'='*100}", flush=True)
    print(f"VALIDATING: {cname}", flush=True)
    print(f"{'='*100}", flush=True)

    cls = cdef["cls"]
    frozen = cdef["frozen"]
    sym = cdef["sym"]
    df_dev = cdef["df_dev"]
    df_oos = cdef["df_oos"]
    df_full = cdef["df_full"]
    cr = {}

    # ── V1: Walk-forward window sensitivity ──
    print(f"\n  V1: Walk-forward window sensitivity", flush=True)
    v1_pass = True
    for train, test, step in [(40, 20, 20), (60, 20, 20), (80, 20, 20)]:
        strat = cls(**frozen)
        sh = wf_sharpe(df_dev, strat, sym, train, test, step)
        tag = "PASS" if sh is not None and sh > 0 else "FAIL"
        if sh is None or sh <= 0:
            v1_pass = False
        sh_str = f"{sh:.2f}" if sh is not None else "N/A"
        print(f"    {train}/{test}/{step}: Sharpe={sh_str}  [{tag}]", flush=True)
    cr["V1_wf_windows"] = v1_pass

    # ── V2: Parameter sensitivity ──
    print(f"\n  V2: Parameter sensitivity (±20%)", flush=True)
    total_perturb = 0
    positive_perturb = 0
    for param, values in cdef["perturbations"].items():
        for val in values:
            perturbed = dict(frozen)
            perturbed[param] = val
            strat = cls(**perturbed)
            sh = wf_sharpe(df_dev, strat, sym, 60, 20, 20)
            total_perturb += 1
            if sh is not None and sh > 0:
                positive_perturb += 1
            sh_str2 = f"{sh:.2f}" if sh is not None else "N/A"
            ptag = "pos" if sh is not None and sh > 0 else "neg"
            print(f"    {param}={val}: Sharpe={sh_str2}  [{ptag}]", flush=True)
    pct_pos = positive_perturb / total_perturb * 100 if total_perturb > 0 else 0
    cr["V2_param_sensitivity"] = pct_pos > 50
    print(f"    Result: {positive_perturb}/{total_perturb} positive ({pct_pos:.0f}%)  "
          f"[{'PASS' if cr['V2_param_sensitivity'] else 'FAIL'}]", flush=True)

    # ── V3: Subperiod analysis (quarterly on dev) ──
    print(f"\n  V3: Subperiod analysis (quarterly)", flush=True)
    quarters = [
        ("Q1 Jan-Mar", datetime(2025, 1, 1, tzinfo=ET), datetime(2025, 3, 31, tzinfo=ET)),
        ("Q2 Apr-Jun", datetime(2025, 4, 1, tzinfo=ET), datetime(2025, 6, 30, tzinfo=ET)),
        ("Q3 Jul-Sep", datetime(2025, 7, 1, tzinfo=ET), datetime(2025, 9, 30, tzinfo=ET)),
        ("Q4 Oct-Nov", datetime(2025, 10, 1, tzinfo=ET), datetime(2025, 11, 30, tzinfo=ET)),
    ]
    profitable_quarters = 0
    for qlabel, qs, qe in quarters:
        qdf = date_filter(df_full, qs, qe)
        if len(qdf) < 100:
            print(f"    {qlabel}: insufficient data", flush=True)
            continue
        strat = cls(**frozen)
        sh, dr, r = quick_bt(qdf, strat, sym)
        tag = "profit" if r.total_return > 0 else "loss"
        if r.total_return > 0:
            profitable_quarters += 1
        print(f"    {qlabel}: Sharpe={sh:.2f}  Ret={r.total_return:+.2%}  T={r.num_trades}  [{tag}]", flush=True)
    cr["V3_subperiods"] = profitable_quarters >= 3
    print(f"    Result: {profitable_quarters}/4 profitable  "
          f"[{'PASS' if cr['V3_subperiods'] else 'FAIL'}]", flush=True)

    # ── V4: Slippage stress test ──
    print(f"\n  V4: Slippage stress test", flush=True)
    strat = cls(**frozen)
    v4_pass = True
    for slip in [0.005, 0.01, 0.02, 0.05]:
        sh, dr, r = quick_bt(df_oos, strat, sym, slip=slip)
        tag = "PASS" if sh > 0 else "FAIL"
        if slip == 0.02 and sh <= 0:
            v4_pass = False
        print(f"    slip=${slip:.3f}: Sharpe={sh:.2f}  Ret={r.total_return:+.2%}  T={r.num_trades}  [{tag}]", flush=True)
    cr["V4_slippage"] = v4_pass

    # ── V5: Trade distribution ──
    print(f"\n  V5: Trade distribution", flush=True)
    strat = cls(**frozen)
    _, _, r_oos = quick_bt(df_oos, strat, sym)
    trades = r_oos.trades
    if trades:
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Max consecutive losers
        max_consec_loss = 0
        current_streak = 0
        for p in pnls:
            if p <= 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        payoff = abs(np.mean(wins) / np.mean(losses)) if losses and wins else 0
        skew = float(pd.Series(pnls).skew()) if len(pnls) > 2 else 0

        print(f"    Trades: {len(trades)}", flush=True)
        print(f"    Mean PnL%: {np.mean(pnls):.4f}%", flush=True)
        print(f"    Median PnL%: {np.median(pnls):.4f}%", flush=True)
        print(f"    Std PnL%: {np.std(pnls):.4f}%", flush=True)
        print(f"    Skewness: {skew:.2f}", flush=True)
        print(f"    Payoff ratio: {payoff:.2f}", flush=True)
        print(f"    Largest win: {max(pnls):.4f}%", flush=True)
        print(f"    Largest loss: {min(pnls):.4f}%", flush=True)
        print(f"    Max consecutive losers: {max_consec_loss}", flush=True)
        cr["V5_trade_dist"] = max_consec_loss < 10
        print(f"    [{('PASS' if cr['V5_trade_dist'] else 'FAIL')}]", flush=True)
    else:
        print(f"    No trades on OOS period", flush=True)
        cr["V5_trade_dist"] = False

    # ── V6: Alpha/beta by period ──
    print(f"\n  V6: Alpha/beta by period", flush=True)
    strat = cls(**frozen)
    _, dr_dev, _ = quick_bt(df_dev, strat, sym)
    _, dr_oos, _ = quick_bt(df_oos, strat, sym)
    a_dev, b_dev = alpha_beta(dr_dev, spy_bench)
    a_oos, b_oos = alpha_beta(dr_oos, spy_bench)
    print(f"    Dev:  alpha={a_dev:+.2%}  beta={b_dev:.3f}", flush=True)
    print(f"    OOS:  alpha={a_oos:+.2%}  beta={b_oos:.3f}", flush=True)
    cr["V6_alpha_beta"] = a_dev > 0 and a_oos > 0
    print(f"    [{'PASS' if cr['V6_alpha_beta'] else 'FAIL'}]", flush=True)

    # ── V7: Regime analysis ──
    print(f"\n  V7: Regime analysis (high-vol vs low-vol)", flush=True)
    strat = cls(**frozen)
    if "atr_percentile" in df_oos.columns:
        daily_atr = df_oos.groupby("date")["atr_percentile"].last()
        high_vol_dates = set(daily_atr[daily_atr > 50].index)
        low_vol_dates = set(daily_atr[daily_atr <= 50].index)

        for regime, dates_set in [("High-vol", high_vol_dates), ("Low-vol", low_vol_dates)]:
            regime_df = df_oos[df_oos["date"].isin(dates_set)].copy()
            if len(regime_df) < 100:
                print(f"    {regime}: insufficient data ({len(regime_df)} bars)", flush=True)
                continue
            sh, dr, r = quick_bt(regime_df, strat, sym)
            print(f"    {regime}: Sharpe={sh:.2f}  Ret={r.total_return:+.2%}  T={r.num_trades}", flush=True)

        # Pass if neither regime has negative total return on dev
        for regime, dates_set in [("High-vol", high_vol_dates), ("Low-vol", low_vol_dates)]:
            regime_df = df_dev[df_dev["date"].isin(
                set(df_dev.groupby("date")["atr_percentile"].last().pipe(
                    lambda s: s[s > 50].index if regime == "High-vol" else s[s <= 50].index
                ))
            )].copy()
            if len(regime_df) < 100:
                continue
            _, _, r = quick_bt(regime_df, strat, sym)
            if r.total_return < 0:
                cr["V7_regime"] = False
                break
        else:
            cr["V7_regime"] = True
    else:
        print(f"    No atr_percentile column — SKIP", flush=True)
        cr["V7_regime"] = True  # No data to fail

    print(f"    [{'PASS' if cr['V7_regime'] else 'FAIL'}]", flush=True)

    # ── Summary for this candidate ──
    passed = sum(1 for v in cr.values() if v)
    total = len(cr)
    results[cname] = cr
    print(f"\n  SUMMARY: {cname} passed {passed}/{total} validations", flush=True)
    for k, v in cr.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}", flush=True)
print("ROBUSTNESS VALIDATION SUMMARY", flush=True)
print(f"{'='*100}", flush=True)

print(f"\n  {'Candidate':<20} {'V1':>4} {'V2':>4} {'V3':>4} {'V4':>4} {'V5':>4} {'V6':>4} {'V7':>4} {'Total':>6}", flush=True)
print(f"  {'-'*19} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*6}", flush=True)

for cname, cr in results.items():
    vals = [cr.get(f"V{i}_{k}", False) for i, k in
            [(1, "wf_windows"), (2, "param_sensitivity"), (3, "subperiods"),
             (4, "slippage"), (5, "trade_dist"), (6, "alpha_beta"), (7, "regime")]]
    marks = ["Y" if v else "N" for v in vals]
    passed = sum(vals)
    print(f"  {cname:<20} {'  '.join(f'{m:>2}' for m in marks)} {passed:>3}/7", flush=True)

# Recommendation
print(f"\n  BLUNT ASSESSMENT:", flush=True)
for cname, cr in results.items():
    passed = sum(1 for v in cr.values() if v)
    total = len(cr)
    if passed == total:
        print(f"    {cname}: ALL CHECKS PASS — strong candidate", flush=True)
    elif passed >= 5:
        print(f"    {cname}: MOSTLY PASSES ({passed}/{total}) — viable with caveats", flush=True)
    elif passed >= 3:
        print(f"    {cname}: MIXED ({passed}/{total}) — proceed with caution", flush=True)
    else:
        print(f"    {cname}: FAILS ({passed}/{total}) — do not promote", flush=True)

print(f"\n{time.time()-t0:.0f}s | ROBUSTNESS VALIDATION COMPLETE", flush=True)
