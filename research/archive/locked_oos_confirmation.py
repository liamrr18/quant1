#!/usr/bin/env python3
"""ONE-TIME locked OOS confirmation.

Pre-specified configs from dev period analysis. NOT iterating on these results.
Tests three portfolio configurations on Dec 2025 - Apr 2026 (locked holdout).
"""

import sys, os, logging, warnings
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.orb import ORBBreakout
from trading.backtest.engine import run_backtest
from trading.config import ORB_SHARED_DEFAULTS

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)
LOCKED_START = datetime(2025, 12, 1, tzinfo=ET)

SYMBOLS = ["SPY", "QQQ", "IWM"]

# ── Pre-specified configs (frozen from dev period analysis) ──

# Config A: Updated per-symbol (changes validated on dev period)
CONFIG_A = {
    "SPY": {"min_gap_pct": 0.3, "min_atr_percentile": 25, "min_breakout_volume": 1.2,
            "last_entry_minute": 900, "stale_exit_bars": 90},  # NEW: stale=90
    "QQQ": {"stale_exit_bars": 90, "min_atr_percentile": 25, "min_breakout_volume": 1.2,
            "last_entry_minute": 900},  # unchanged
    "IWM": {"min_atr_percentile": 0, "min_breakout_volume": 0,
            "last_entry_minute": 900},  # NEW: last_entry=900
}

# Config B: SPY+QQQ only (drop IWM)
CONFIG_B = {
    "SPY": CONFIG_A["SPY"],
    "QQQ": CONFIG_A["QQQ"],
}

# Config C: Original profiles (baseline)
CONFIG_C = {
    "SPY": {"min_gap_pct": 0.3, "min_atr_percentile": 25, "min_breakout_volume": 1.2,
            "last_entry_minute": 900},
    "QQQ": {"stale_exit_bars": 90, "min_atr_percentile": 25, "min_breakout_volume": 1.2,
            "last_entry_minute": 900},
    "IWM": {"min_atr_percentile": 0, "min_breakout_volume": 0, "last_entry_minute": 0},
}

# ── Load data ──
print("Loading data...")
raw_data = {}
for sym in SYMBOLS:
    df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
    df = prepare_features(df)
    raw_data[sym] = df


def make_strat(profile):
    params = dict(ORB_SHARED_DEFAULTS)
    params.update(profile)
    return ORBBreakout(**params)


def split_data(df, start_dt, end_dt):
    mask = (df["dt"] >= start_dt) & (df["dt"] <= end_dt)
    return df.loc[mask].copy()


def get_benchmark_daily(start_dt, end_dt):
    df = raw_data["SPY"]
    sub = split_data(df, start_dt, end_dt)
    daily = sub.set_index("dt")["close"].resample("D").last().dropna()
    return daily.pct_change().dropna()


def full_metrics(result, bench_daily):
    """Full metrics suite."""
    dr = result.daily_returns
    if dr is None or len(dr) < 5:
        return None

    n_days = len(dr)
    total_ret = result.total_return
    annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
    down = dr[dr < 0]
    down_std = down.std() if len(down) > 2 else dr.std()
    sortino = (dr.mean() / down_std) * np.sqrt(252) if down_std > 0 else 0
    cum = (1 + dr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

    aligned = pd.DataFrame({"s": dr, "b": bench_daily}).dropna()
    if len(aligned) > 10:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(aligned["b"], aligned["s"])
        beta = slope
        alpha_ann = intercept * 252
        resid = aligned["s"] - (beta * aligned["b"] + intercept)
        te = resid.std() * np.sqrt(252)
        ir = alpha_ann / te if te > 0 else 0
    else:
        beta = alpha_ann = ir = 0

    bench_ret = (1 + bench_daily).prod() - 1

    trades = result.trades
    long_t = [t for t in trades if t.direction == "long"]
    short_t = [t for t in trades if t.direction == "short"]

    return {
        "n_days": n_days, "total_return": total_ret, "annual_return": annual_ret,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "max_drawdown": max_dd,
        "alpha": alpha_ann, "beta": beta, "ir": ir,
        "profit_factor": result.profit_factor, "win_rate": result.win_rate,
        "avg_trade_pct": result.avg_trade_pct, "num_trades": result.num_trades,
        "exposure": result.exposure_pct, "bench_return": bench_ret,
        "active_passive": total_ret - bench_ret,
        "long_trades": len(long_t), "short_trades": len(short_t),
        "long_pnl": sum(t.pnl for t in long_t), "short_pnl": sum(t.pnl for t in short_t),
        "daily_returns": dr,
    }


def print_sym_metrics(sym, m):
    if m is None:
        print(f"    {sym}: no data")
        return
    print(f"    {sym} ({m['n_days']}d, {m['num_trades']}T):")
    print(f"      Return:   total={m['total_return']:+.2%}  annual={m['annual_return']:+.2%}"
          f"  bench={m['bench_return']:+.2%}  active-passive={m['active_passive']:+.2%}")
    print(f"      Risk:     Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}"
          f"  Calmar={m['calmar']:.2f}  MaxDD={m['max_drawdown']:.2%}")
    print(f"      Alpha:    alpha={m['alpha']:+.2%}  beta={m['beta']:.3f}  IR={m['ir']:.2f}")
    print(f"      Trades:   PF={m['profit_factor']:.2f}  WR={m['win_rate']:.1%}"
          f"  avg={m['avg_trade_pct']:.3f}%  exposure={m['exposure']:.1f}%")
    print(f"      L/S:      long={m['long_trades']} (${m['long_pnl']:.0f})"
          f"  short={m['short_trades']} (${m['short_pnl']:.0f})")


def run_config(config_name, sym_profiles, start_dt, end_dt):
    """Run a full config on a period."""
    print(f"\n  {config_name}:")
    bench = get_benchmark_daily(start_dt, end_dt)
    sym_metrics = {}
    sym_daily = {}

    for sym, profile in sym_profiles.items():
        strat = make_strat(profile)
        df = split_data(raw_data[sym], start_dt, end_dt)
        df = strat.generate_signals(df)
        result = run_backtest(df, strat, sym)
        m = full_metrics(result, bench)
        if m:
            sym_metrics[sym] = m
            sym_daily[sym] = m["daily_returns"]
            print_sym_metrics(sym, m)

    # Portfolio
    if len(sym_daily) >= 2:
        port_df = pd.DataFrame(sym_daily).fillna(0)
        port_ret = port_df.mean(axis=1)
        n = len(port_ret)
        total_ret = (1 + port_ret).prod() - 1
        annual_ret = (1 + total_ret) ** (252 / max(n, 1)) - 1
        sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() > 0 else 0
        down = port_ret[port_ret < 0]
        down_std = down.std() if len(down) > 2 else port_ret.std()
        sortino = (port_ret.mean() / down_std) * np.sqrt(252) if down_std > 0 else 0
        cum = (1 + port_ret).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

        aligned = pd.DataFrame({"s": port_ret, "b": bench}).dropna()
        if len(aligned) > 10:
            slope, intercept, r_val, _, _ = scipy_stats.linregress(aligned["b"], aligned["s"])
            beta = slope
            alpha_ann = intercept * 252
            resid = aligned["s"] - (beta * aligned["b"] + intercept)
            te = resid.std() * np.sqrt(252)
            ir = alpha_ann / te if te > 0 else 0
        else:
            beta = alpha_ann = ir = 0

        bench_ret = (1 + bench).prod() - 1
        total_trades = sum(m["num_trades"] for m in sym_metrics.values())
        avg_exp = np.mean([m["exposure"] for m in sym_metrics.values()])

        print(f"\n    PORTFOLIO ({n}d, {total_trades}T):")
        print(f"      Return:   total={total_ret:+.2%}  annual={annual_ret:+.2%}"
              f"  bench={bench_ret:+.2%}  active-passive={total_ret - bench_ret:+.2%}")
        print(f"      Risk:     Sharpe={sharpe:.2f}  Sortino={sortino:.2f}"
              f"  Calmar={calmar:.2f}  MaxDD={max_dd:.2%}")
        print(f"      Alpha:    alpha={alpha_ann:+.2%}  beta={beta:.3f}  IR={ir:.2f}")
        print(f"      Trades:   total={total_trades}  avg_exposure={avg_exp:.1f}%")

        return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar,
                "return": total_ret, "annual_return": annual_ret,
                "max_dd": max_dd, "alpha": alpha_ann, "beta": beta, "ir": ir,
                "trades": total_trades, "exposure": avg_exp}
    return None


# ═══════════════════════════════════════════════════════════
print("=" * 100)
print("LOCKED OOS CONFIRMATION (Dec 1, 2025 - Apr 4, 2026)")
print("This is a one-time confirmation. Results are NOT used to iterate.")
print("=" * 100)

# Also run on full period for comparison
configs = {
    "A: Updated profiles (SPY+stale, IWM+late-cut)": CONFIG_A,
    "B: SPY+QQQ only (updated)": CONFIG_B,
    "C: Original profiles (baseline)": CONFIG_C,
}

# Run each on locked OOS
print("\n" + "-" * 100)
print("LOCKED OOS PERIOD (Dec 1, 2025 - Apr 4, 2026)")
print("-" * 100)
oos_results = {}
for name, cfg in configs.items():
    oos_results[name] = run_config(name, cfg, LOCKED_START, DATA_END)

# Also run on full period for comparison
print("\n" + "-" * 100)
print("FULL PERIOD (Jan 2, 2025 - Apr 4, 2026) - for reference")
print("-" * 100)
full_results = {}
for name, cfg in configs.items():
    full_results[name] = run_config(name, cfg, DATA_START, DATA_END)

# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("COMPARISON TABLE")
print("=" * 100)

print(f"\n  {'Config':<48} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Alpha':>8}"
      f" {'Beta':>6} {'MaxDD':>7} {'Trades':>7}")
print(f"  {'-'*48} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*7}")

for period, results, label in [("Locked OOS", oos_results, "OOS"),
                                ("Full period", full_results, "FULL")]:
    for name, r in results.items():
        if r:
            short_name = f"[{label}] {name[:40]}"
            print(f"  {short_name:<48} {r['sharpe']:>7.2f} {r['sortino']:>8.2f}"
                  f" {r['return']:>+8.2%} {r['alpha']:>+8.2%} {r['beta']:>6.3f}"
                  f" {r['max_dd']:>7.2%} {r['trades']:>7}")

print("\n" + "=" * 100)
print("CONFIRMATION COMPLETE")
print("=" * 100)
