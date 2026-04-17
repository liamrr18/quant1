#!/usr/bin/env python3
"""Screen new instruments for ORB and VWAP Reversion edges.

Tests MGC, MCL, M2K, EURUSD, GBPUSD with walk-forward validation.
Compares correlation with existing MES strategies.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.orb import ORBBreakout
from trading.strategies.vwap_reversion import VWAPReversion
from trading.config import ORB_SHARED_DEFAULTS, VWAP_REVERSION_DEFAULTS

ET = pytz.timezone("America/New_York")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")


def load_data(filename):
    path = os.path.join(CACHE, filename)
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)
    df["date"] = df["dt"].dt.date
    df["hour"] = df["dt"].dt.hour
    df["minute"] = df["dt"].dt.minute
    return df


def filter_cash_hours(df):
    """Filter to 9:30 - 16:00 ET."""
    mod = df["hour"] * 60 + df["minute"]
    return df[(mod >= 570) & (mod < 960)].copy()


def walk_forward(df, strategy_fn, train_days=60, test_days=20, step_days=20):
    """Walk-forward OOS test. Returns daily returns series."""
    dates = sorted(df["date"].unique())
    all_oos_returns = []

    i = 0
    while i + train_days + test_days <= len(dates):
        test_start = dates[i + train_days]
        test_end = dates[min(i + train_days + test_days - 1, len(dates) - 1)]
        test_dates = set(dates[i + train_days: i + train_days + test_days])

        test_df = df[df["date"].isin(test_dates)].copy()
        if len(test_df) < 50:
            i += step_days
            continue

        strat = strategy_fn()
        sig_df = strat.generate_signals(test_df)
        sig_df["position"] = sig_df["signal"].shift(1).fillna(0)
        sig_df["bar_ret"] = sig_df["close"].pct_change().fillna(0)
        sig_df["strat_ret"] = sig_df["position"] * sig_df["bar_ret"]
        daily = sig_df.groupby("date")["strat_ret"].sum()
        all_oos_returns.append(daily)

        i += step_days

    if all_oos_returns:
        return pd.concat(all_oos_returns)
    return pd.Series(dtype=float)


def compute_metrics(daily_returns):
    if len(daily_returns) < 10 or daily_returns.std() == 0:
        return {"sharpe": 0, "pnl_pct": 0, "trades_approx": 0, "win_rate": 0, "days": len(daily_returns)}
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    pnl_pct = daily_returns.sum()
    active = daily_returns[daily_returns != 0]
    trades = len(active)
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    return {
        "sharpe": sharpe,
        "pnl_pct": pnl_pct,
        "trades_approx": trades,
        "win_rate": win_rate,
        "days": len(daily_returns),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 90)
print("  NEW INSTRUMENT SCREENING — ORB + VWAP Walk-Forward")
print("=" * 90)

# Load MES as reference for correlation
print("\nLoading data...", flush=True)
mes_df = load_data("MES_futures_cash_1min.csv")
mes_df = filter_cash_hours(mes_df)
print(f"  MES reference: {len(mes_df)} bars, {mes_df['date'].nunique()} days")

# Generate MES ORB and VWAP returns for correlation
mes_orb_strat = ORBBreakout(**ORB_SHARED_DEFAULTS)
mes_orb_df = mes_orb_strat.generate_signals(mes_df.copy())
mes_orb_df["position"] = mes_orb_df["signal"].shift(1).fillna(0)
mes_orb_df["bar_ret"] = mes_orb_df["close"].pct_change().fillna(0)
mes_orb_df["strat_ret"] = mes_orb_df["position"] * mes_orb_df["bar_ret"]
mes_orb_daily = mes_orb_df.groupby("date")["strat_ret"].sum()

mes_vwap_strat = VWAPReversion(**VWAP_REVERSION_DEFAULTS)
mes_vwap_df = mes_vwap_strat.generate_signals(mes_df.copy())
mes_vwap_df["position"] = mes_vwap_df["signal"].shift(1).fillna(0)
mes_vwap_df["bar_ret"] = mes_vwap_df["close"].pct_change().fillna(0)
mes_vwap_df["strat_ret"] = mes_vwap_df["position"] * mes_vwap_df["bar_ret"]
mes_vwap_daily = mes_vwap_df.groupby("date")["strat_ret"].sum()
print(f"  MES ORB ref: Sharpe {mes_orb_daily.mean()/mes_orb_daily.std()*np.sqrt(252):.2f}")
print(f"  MES VWAP ref: Sharpe {mes_vwap_daily.mean()/mes_vwap_daily.std()*np.sqrt(252):.2f}")

instruments = {
    "MGC": ("MGC_futures_cash_1min.csv", "Micro Gold", True),
    "MCL": ("MCL_futures_cash_1min.csv", "Micro Crude", True),
    "M2K": ("M2K_futures_cash_1min.csv", "Micro Russell", True),
    "MBT": ("MBT_futures_cash_1min.csv", "Micro Bitcoin", True),
    "EURUSD": ("EURUSD_forex_1min.csv", "EUR/USD Forex", False),
    "GBPUSD": ("GBPUSD_forex_1min.csv", "GBP/USD Forex", False),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN EACH INSTRUMENT
# ═══════════════════════════════════════════════════════════════════════════════

results = []

for sym, (filename, name, is_futures) in instruments.items():
    print(f"\n--- {sym} ({name}) ---", flush=True)
    try:
        df = load_data(filename)
        n_days = df["date"].nunique()
        n_bars = len(df)
        print(f"  {n_bars} bars, {n_days} days")

        if is_futures:
            df = filter_cash_hours(df)
            print(f"  After cash filter: {len(df)} bars")
        else:
            # Forex: filter to US session 8 AM - 5 PM for comparability
            mod = df["hour"] * 60 + df["minute"]
            df = df[(mod >= 480) & (mod < 1020)].copy()
            print(f"  After US session filter: {len(df)} bars")

        if n_days < 30:
            print(f"  SKIP: only {n_days} days (need 30+)")
            continue

        # Daily return correlation with MES
        daily_close = df.groupby("date")["close"].last().pct_change().dropna()
        mes_daily_close = mes_df.groupby("date")["close"].last().pct_change().dropna()
        common = pd.DataFrame({"new": daily_close, "mes": mes_daily_close}).dropna()
        price_corr = common["new"].corr(common["mes"]) if len(common) > 10 else 0
        avg_vol = df.groupby("date")["volume"].sum().mean()
        avg_range = ((df["high"] - df["low"]) / df["close"]).mean() * 100

        print(f"  Avg daily volume: {avg_vol:,.0f}")
        print(f"  Avg bar range: {avg_range:.3f}%")
        print(f"  Price correlation with MES: {price_corr:.3f}")

        # ORB test
        print(f"  Testing ORB...", flush=True)
        orb_returns = walk_forward(df, lambda: ORBBreakout(**ORB_SHARED_DEFAULTS))
        orb_m = compute_metrics(orb_returns)

        # Correlation with MES ORB
        orb_corr = 0
        if len(orb_returns) > 10:
            combined = pd.DataFrame({"new": orb_returns, "mes": mes_orb_daily}).dropna()
            orb_corr = combined["new"].corr(combined["mes"]) if len(combined) > 10 else 0

        print(f"    ORB: Sharpe={orb_m['sharpe']:.2f}, trades={orb_m['trades_approx']}, "
              f"WR={orb_m['win_rate']:.1%}, corr_MES={orb_corr:.2f}")

        # VWAP test
        print(f"  Testing VWAP...", flush=True)
        vwap_returns = walk_forward(df, lambda: VWAPReversion(**VWAP_REVERSION_DEFAULTS))
        vwap_m = compute_metrics(vwap_returns)

        vwap_corr = 0
        if len(vwap_returns) > 10:
            combined = pd.DataFrame({"new": vwap_returns, "mes": mes_vwap_daily}).dropna()
            vwap_corr = combined["new"].corr(combined["mes"]) if len(combined) > 10 else 0

        print(f"    VWAP: Sharpe={vwap_m['sharpe']:.2f}, trades={vwap_m['trades_approx']}, "
              f"WR={vwap_m['win_rate']:.1%}, corr_MES={vwap_corr:.2f}")

        results.append({
            "symbol": sym, "name": name, "days": n_days, "bars": n_bars,
            "price_corr_mes": price_corr, "avg_volume": avg_vol, "avg_range_pct": avg_range,
            "orb_sharpe": orb_m["sharpe"], "orb_trades": orb_m["trades_approx"],
            "orb_wr": orb_m["win_rate"], "orb_corr_mes": orb_corr,
            "vwap_sharpe": vwap_m["sharpe"], "vwap_trades": vwap_m["trades_approx"],
            "vwap_wr": vwap_m["win_rate"], "vwap_corr_mes": vwap_corr,
        })

    except Exception as e:
        print(f"  ERROR: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 90}")
print("  SCREENING RESULTS SUMMARY")
print(f"{'=' * 90}")

print(f"\n{'Instrument':<12} {'Days':>5} {'PriceCorr':>9} {'ORB_Sh':>7} {'ORB_Tr':>7} {'ORB_WR':>7} {'ORB_Corr':>8} "
      f"{'VWAP_Sh':>8} {'VWAP_Tr':>8} {'VWAP_WR':>8} {'VWAP_Corr':>9} {'Verdict':>10}")
print("-" * 110)

for r in results:
    # Verdict: PASS if Sharpe > 0 and corr < 0.3
    orb_pass = r["orb_sharpe"] > 0 and abs(r["orb_corr_mes"]) < 0.3
    vwap_pass = r["vwap_sharpe"] > 0 and abs(r["vwap_corr_mes"]) < 0.3
    if orb_pass and vwap_pass:
        verdict = "BOTH"
    elif orb_pass:
        verdict = "ORB only"
    elif vwap_pass:
        verdict = "VWAP only"
    else:
        verdict = "FAIL"

    print(f"{r['symbol']:<12} {r['days']:>5} {r['price_corr_mes']:>+9.3f} "
          f"{r['orb_sharpe']:>7.2f} {r['orb_trades']:>7} {r['orb_wr']:>7.1%} {r['orb_corr_mes']:>+8.2f} "
          f"{r['vwap_sharpe']:>8.2f} {r['vwap_trades']:>8} {r['vwap_wr']:>8.1%} {r['vwap_corr_mes']:>+9.2f} "
          f"{verdict:>10}")

print(f"\n{'=' * 90}")
print("  SCREENING COMPLETE")
print(f"{'=' * 90}")
