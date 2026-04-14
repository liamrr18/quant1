#!/usr/bin/env python3
"""Deep diagnostic: alpha/beta decomposition, proper splits, full metrics.

Splits:
  Train:      Jan 2 - Jul 31, 2025   (~147 trading days)
  Validation: Aug 1 - Nov 30, 2025   (~85 trading days)
  Locked OOS: Dec 1, 2025 - Apr 4, 2026 (~87 trading days)

Reports per split and per symbol:
  Sharpe, Sortino, Calmar, max drawdown, total return, annual return,
  alpha, beta, information ratio, profit factor, win rate, avg trade,
  trade count, exposure, benchmark return, active-passive spread.
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
from trading.strategies.vwap_reversion import VWAPReversion
from trading.strategies.rsi_reversion import RSIReversion
from trading.backtest.engine import run_backtest
from trading.backtest.walkforward import walk_forward
from trading.config import SYMBOL_PROFILES, ORB_SHARED_DEFAULTS

ET = pytz.timezone("America/New_York")
DATA_START = datetime(2025, 1, 2, tzinfo=ET)
DATA_END = datetime(2026, 4, 4, tzinfo=ET)

# Period boundaries
TRAIN_END = datetime(2025, 7, 31, tzinfo=ET)
VAL_END = datetime(2025, 11, 30, tzinfo=ET)
# Locked OOS: Dec 1 2025 - Apr 4 2026

SYMBOLS = ["SPY", "QQQ", "IWM"]

# ── Load all data ──
print("Loading data...")
raw_data = {}
for sym in SYMBOLS:
    df = get_minute_bars(sym, DATA_START, DATA_END, use_cache=True)
    df = prepare_features(df)
    raw_data[sym] = df
    print(f"  {sym}: {len(df)} bars, {df['date'].nunique()} trading days")


def make_strat(sym):
    params = dict(ORB_SHARED_DEFAULTS)
    if sym in SYMBOL_PROFILES:
        params.update(SYMBOL_PROFILES[sym])
    return ORBBreakout(**params)


def split_data(df, start_dt, end_dt):
    """Extract data between dates (inclusive)."""
    mask = (df["dt"] >= start_dt) & (df["dt"] <= end_dt)
    return df.loc[mask].copy()


def get_benchmark_daily_returns(sym, start_dt, end_dt):
    """Buy-and-hold daily returns for a symbol over a period."""
    df = raw_data[sym]
    sub = split_data(df, start_dt, end_dt)
    daily = sub.set_index("dt")["close"].resample("D").last().dropna()
    return daily.pct_change().dropna()


def compute_full_metrics(result, benchmark_daily, period_name):
    """Compute the full metrics suite from a BacktestResult and benchmark."""
    dr = result.daily_returns
    if dr is None or len(dr) < 5:
        return None

    # Basic
    n_days = len(dr)
    total_ret = result.total_return
    annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

    # Sharpe
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0.0

    # Sortino
    downside = dr[dr < 0]
    down_std = downside.std() if len(downside) > 2 else dr.std()
    sortino = (dr.mean() / down_std) * np.sqrt(252) if down_std > 0 else 0.0

    # Max drawdown
    cum = (1 + dr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0.0

    # Alpha, Beta, IR via regression
    # Align strategy and benchmark returns by date
    aligned = pd.DataFrame({"strat": dr, "bench": benchmark_daily}).dropna()
    if len(aligned) > 10:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            aligned["bench"], aligned["strat"]
        )
        beta = slope
        alpha_daily = intercept
        alpha_annual = alpha_daily * 252
        residuals = aligned["strat"] - (beta * aligned["bench"] + alpha_daily)
        tracking_error = residuals.std() * np.sqrt(252)
        info_ratio = alpha_annual / tracking_error if tracking_error > 0 else 0.0
        r_squared = r_value ** 2
    else:
        beta = alpha_annual = info_ratio = r_squared = 0.0
        tracking_error = 0.0

    # Benchmark stats
    bench_total = (1 + benchmark_daily).prod() - 1 if len(benchmark_daily) > 0 else 0.0
    bench_annual = (1 + bench_total) ** (252 / max(len(benchmark_daily), 1)) - 1
    bench_sharpe = ((benchmark_daily.mean() / benchmark_daily.std()) * np.sqrt(252)
                    if len(benchmark_daily) > 2 and benchmark_daily.std() > 0 else 0.0)

    # Trade direction analysis
    trades = result.trades
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]
    long_pnl = sum(t.pnl for t in long_trades)
    short_pnl = sum(t.pnl for t in short_trades)

    return {
        "period": period_name,
        "n_days": n_days,
        "total_return": total_ret,
        "annual_return": annual_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "alpha_annual": alpha_annual,
        "beta": beta,
        "info_ratio": info_ratio,
        "r_squared": r_squared,
        "tracking_error": tracking_error,
        "profit_factor": result.profit_factor,
        "win_rate": result.win_rate,
        "avg_trade_pct": result.avg_trade_pct,
        "num_trades": result.num_trades,
        "exposure_pct": result.exposure_pct,
        "bench_total": bench_total,
        "bench_annual": bench_annual,
        "bench_sharpe": bench_sharpe,
        "active_minus_passive": total_ret - bench_total,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
    }


def print_metrics(m, sym):
    """Print a single metrics dict."""
    if m is None:
        print(f"    {sym}: insufficient data")
        return
    print(f"    {sym} [{m['period']}] ({m['n_days']}d, {m['num_trades']}T):")
    print(f"      Return:   total={m['total_return']:+.2%}  annual={m['annual_return']:+.2%}"
          f"  bench={m['bench_total']:+.2%}  active-passive={m['active_minus_passive']:+.2%}")
    print(f"      Risk:     Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}"
          f"  Calmar={m['calmar']:.2f}  MaxDD={m['max_drawdown']:.2%}")
    print(f"      Alpha:    alpha={m['alpha_annual']:+.2%}  beta={m['beta']:.3f}"
          f"  IR={m['info_ratio']:.2f}  R2={m['r_squared']:.3f}")
    print(f"      Trades:   PF={m['profit_factor']:.2f}  WR={m['win_rate']:.1%}"
          f"  avg={m['avg_trade_pct']:.3f}%  exposure={m['exposure_pct']:.1f}%")
    print(f"      L/S:      long={m['long_trades']} (${m['long_pnl']:.0f})"
          f"  short={m['short_trades']} (${m['short_pnl']:.0f})")


def run_period(period_name, start_dt, end_dt):
    """Run all symbols on a period, return dict of metrics."""
    print(f"\n  --- {period_name} ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}) ---")
    results = {}
    all_strat_returns = {}
    all_bench_returns = {}

    for sym in SYMBOLS:
        strat = make_strat(sym)
        df = split_data(raw_data[sym], start_dt, end_dt)
        if len(df) < 100:
            print(f"    {sym}: only {len(df)} bars, skipping")
            continue

        df = strat.generate_signals(df)
        result = run_backtest(df, strat, sym)

        bench_dr = get_benchmark_daily_returns("SPY", start_dt, end_dt)
        m = compute_full_metrics(result, bench_dr, period_name)
        results[sym] = m
        print_metrics(m, sym)

        if m is not None and result.daily_returns is not None:
            all_strat_returns[sym] = result.daily_returns
            all_bench_returns[sym] = bench_dr

    # Portfolio metrics (equal weight)
    if len(all_strat_returns) >= 2:
        port_df = pd.DataFrame(all_strat_returns).fillna(0)
        port_ret = port_df.mean(axis=1)

        bench_dr = get_benchmark_daily_returns("SPY", start_dt, end_dt)

        # Compute portfolio metrics manually
        n_days = len(port_ret)
        total_ret = (1 + port_ret).prod() - 1
        annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
        sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() > 0 else 0
        downside = port_ret[port_ret < 0]
        down_std = downside.std() if len(downside) > 2 else port_ret.std()
        sortino = (port_ret.mean() / down_std) * np.sqrt(252) if down_std > 0 else 0
        cum = (1 + port_ret).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

        aligned = pd.DataFrame({"strat": port_ret, "bench": bench_dr}).dropna()
        if len(aligned) > 10:
            slope, intercept, r_val, _, _ = scipy_stats.linregress(aligned["bench"], aligned["strat"])
            beta = slope
            alpha_annual = intercept * 252
            residuals = aligned["strat"] - (beta * aligned["bench"] + intercept)
            te = residuals.std() * np.sqrt(252)
            ir = alpha_annual / te if te > 0 else 0
        else:
            beta = alpha_annual = ir = 0

        bench_total = (1 + bench_dr).prod() - 1
        bench_sharpe = (bench_dr.mean() / bench_dr.std()) * np.sqrt(252) if bench_dr.std() > 0 else 0

        total_trades = sum(m["num_trades"] for m in results.values() if m)
        avg_exposure = np.mean([m["exposure_pct"] for m in results.values() if m])

        print(f"\n    PORTFOLIO [{period_name}] ({n_days}d, {total_trades}T):")
        print(f"      Return:   total={total_ret:+.2%}  annual={annual_ret:+.2%}"
              f"  bench={bench_total:+.2%}  active-passive={total_ret - bench_total:+.2%}")
        print(f"      Risk:     Sharpe={sharpe:.2f}  Sortino={sortino:.2f}"
              f"  Calmar={calmar:.2f}  MaxDD={max_dd:.2%}")
        print(f"      Alpha:    alpha={alpha_annual:+.2%}  beta={beta:.3f}  IR={ir:.2f}")
        print(f"      Trades:   total={total_trades}  avg_exposure={avg_exposure:.1f}%")

        results["PORTFOLIO"] = {
            "period": period_name, "n_days": n_days, "total_return": total_ret,
            "annual_return": annual_ret, "sharpe": sharpe, "sortino": sortino,
            "calmar": calmar, "max_drawdown": max_dd, "alpha_annual": alpha_annual,
            "beta": beta, "info_ratio": ir, "num_trades": total_trades,
            "exposure_pct": avg_exposure, "bench_total": bench_total,
            "bench_sharpe": bench_sharpe, "active_minus_passive": total_ret - bench_total,
        }

    return results


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("SECTION 1: PER-PERIOD ANALYSIS (Train / Validation / Locked OOS)")
print("=" * 100)

periods = [
    ("TRAIN", DATA_START, TRAIN_END),
    ("VALIDATION", datetime(2025, 8, 1, tzinfo=ET), VAL_END),
    ("LOCKED_OOS", datetime(2025, 12, 1, tzinfo=ET), DATA_END),
]

all_results = {}
for name, start, end in periods:
    all_results[name] = run_period(name, start, end)

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 2: FULL-PERIOD ANALYSIS")
print("=" * 100)
all_results["FULL"] = run_period("FULL", DATA_START, DATA_END)

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 3: WALK-FORWARD OOS (development period only, 60/20/20)")
print("=" * 100)
print("\n  Walk-forward across TRAIN+VAL period (Jan 2025 - Nov 2025)")

dev_end = VAL_END
wf_sym_returns = {}
wf_sym_sharpes = {}

for sym in SYMBOLS:
    strat = make_strat(sym)
    df = split_data(raw_data[sym], DATA_START, dev_end)
    df = strat.generate_signals(df)
    wf = walk_forward(df, strat, sym, train_days=60, test_days=20, step_days=20)

    # Extract concatenated OOS daily returns
    all_daily = []
    for oos_r in wf.oos_results:
        dr = oos_r.daily_returns
        if dr is not None and len(dr) > 0:
            all_daily.append(dr)
    if all_daily:
        combined = pd.concat(all_daily).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        wf_sym_returns[sym] = combined

    window_sharpes = [r.sharpe_ratio for r in wf.oos_results]
    pos_windows = sum(1 for s in window_sharpes if s > 0)
    n_windows = len(window_sharpes)

    # Compute Sharpe from concatenated returns (NOT average of per-window)
    if sym in wf_sym_returns and len(wf_sym_returns[sym]) > 10:
        cr = wf_sym_returns[sym]
        concat_sharpe = (cr.mean() / cr.std()) * np.sqrt(252)
    else:
        concat_sharpe = 0

    wf_sym_sharpes[sym] = concat_sharpe
    print(f"  {sym}: concat_OOS_Sharpe={concat_sharpe:.2f}  avg_window_Sharpe={np.mean(window_sharpes):.2f}"
          f"  windows={pos_windows}/{n_windows} positive"
          f"  trades={wf.total_trades}  return={wf.total_return*100:+.2f}%")

# WF portfolio
if len(wf_sym_returns) >= 2:
    wf_port = pd.DataFrame(wf_sym_returns).fillna(0)
    wf_port_ret = wf_port.mean(axis=1)
    wf_port_sharpe = (wf_port_ret.mean() / wf_port_ret.std()) * np.sqrt(252) if wf_port_ret.std() > 0 else 0
    wf_total = (1 + wf_port_ret).prod() - 1
    cum = (1 + wf_port_ret).cumprod()
    wf_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    print(f"\n  PORTFOLIO WF-OOS: Sharpe={wf_port_sharpe:.2f}  Return={wf_total:+.2%}  MaxDD={wf_dd:.2%}")

    # Portfolio alpha/beta on WF-OOS
    bench_wf = get_benchmark_daily_returns("SPY", DATA_START, dev_end)
    aligned = pd.DataFrame({"strat": wf_port_ret, "bench": bench_wf}).dropna()
    if len(aligned) > 10:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(aligned["bench"], aligned["strat"])
        print(f"  PORTFOLIO WF-OOS alpha/beta: alpha={intercept*252:+.2%}  beta={slope:.3f}  R2={r_val**2:.3f}")

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 4: TRADE-LEVEL ANALYSIS (full period)")
print("=" * 100)

for sym in SYMBOLS:
    strat = make_strat(sym)
    df = raw_data[sym].copy()
    df = strat.generate_signals(df)
    result = run_backtest(df, strat, sym)
    trades = result.trades

    if not trades:
        continue

    long_t = [t for t in trades if t.direction == "long"]
    short_t = [t for t in trades if t.direction == "short"]

    long_wins = sum(1 for t in long_t if t.pnl > 0)
    short_wins = sum(1 for t in short_t if t.pnl > 0)

    long_avg = np.mean([t.pnl_pct for t in long_t]) * 100 if long_t else 0
    short_avg = np.mean([t.pnl_pct for t in short_t]) * 100 if short_t else 0

    long_total = sum(t.pnl for t in long_t)
    short_total = sum(t.pnl for t in short_t)

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in exit_reasons:
            exit_reasons[r] = {"count": 0, "pnl": 0}
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["pnl"] += t.pnl

    # Duration analysis
    durations = []
    for t in trades:
        if t.entry_time and t.exit_time:
            dur = (t.exit_time - t.entry_time).total_seconds() / 60
            durations.append(dur)

    print(f"\n  {sym} ({len(trades)} trades):")
    print(f"    Long:  {len(long_t)} trades, WR={long_wins/max(len(long_t),1):.1%},"
          f" avg={long_avg:+.3f}%, total=${long_total:+.0f}")
    print(f"    Short: {len(short_t)} trades, WR={short_wins/max(len(short_t),1):.1%},"
          f" avg={short_avg:+.3f}%, total=${short_total:+.0f}")
    print(f"    Ratio: {len(long_t)/max(len(trades),1):.0%} long / {len(short_t)/max(len(trades),1):.0%} short")

    if durations:
        print(f"    Duration: median={np.median(durations):.0f}min, mean={np.mean(durations):.0f}min,"
              f" max={max(durations):.0f}min")

    print(f"    Exit reasons:")
    for reason, info in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
        print(f"      {reason:10s}: {info['count']:4d} trades, ${info['pnl']:+8.0f}")

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 5: STRATEGY COMPARISON (ORB vs VWAP vs RSI)")
print("=" * 100)

strategies_to_test = {
    "ORB (per-symbol)": lambda sym: make_strat(sym),
    "VWAP (1.5/0.3)": lambda sym: VWAPReversion(entry_std=1.5, exit_std=0.3),
    "VWAP (2.0/0.5)": lambda sym: VWAPReversion(entry_std=2.0, exit_std=0.5),
    "RSI (14/25/75)": lambda sym: RSIReversion(rsi_period=14, oversold=25, overbought=75),
    "RSI (7/20/80)": lambda sym: RSIReversion(rsi_period=7, oversold=20, overbought=80),
}

print(f"\n  {'Strategy':<22} {'Symbol':<6} {'Sharpe':>7} {'Return':>8} {'Alpha':>8}"
      f" {'Beta':>6} {'Trades':>7} {'WR':>6} {'PF':>6}")
print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")

strat_results = {}
for strat_name, strat_fn in strategies_to_test.items():
    sym_daily = {}
    for sym in SYMBOLS:
        strat = strat_fn(sym)
        df = raw_data[sym].copy()
        df = strat.generate_signals(df)
        result = run_backtest(df, strat, sym)
        bench_dr = get_benchmark_daily_returns("SPY", DATA_START, DATA_END)
        m = compute_full_metrics(result, bench_dr, "FULL")
        if m:
            print(f"  {strat_name:<22} {sym:<6} {m['sharpe']:>7.2f} {m['total_return']:>+8.2%}"
                  f" {m['alpha_annual']:>+8.2%} {m['beta']:>6.3f} {m['num_trades']:>7}"
                  f" {m['win_rate']:>6.1%} {m['profit_factor']:>6.2f}")
            if result.daily_returns is not None:
                sym_daily[sym] = result.daily_returns

    # Portfolio for this strategy
    if len(sym_daily) >= 2:
        port_df = pd.DataFrame(sym_daily).fillna(0)
        port_ret = port_df.mean(axis=1)
        port_sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() > 0 else 0
        port_total = (1 + port_ret).prod() - 1

        aligned = pd.DataFrame({"strat": port_ret, "bench": bench_dr}).dropna()
        if len(aligned) > 10:
            slope, intercept, _, _, _ = scipy_stats.linregress(aligned["bench"], aligned["strat"])
            port_alpha = intercept * 252
            port_beta = slope
        else:
            port_alpha = port_beta = 0

        total_trades = sum(m["num_trades"] for m in [compute_full_metrics(
            run_backtest(strat_fn(s).generate_signals(raw_data[s].copy()), strat_fn(s), s),
            bench_dr, "FULL") for s in SYMBOLS] if m)

        print(f"  {strat_name:<22} {'PORT':<6} {port_sharpe:>7.2f} {port_total:>+8.2%}"
              f" {port_alpha:>+8.2%} {port_beta:>6.3f}")
        strat_results[strat_name] = {"sharpe": port_sharpe, "returns": port_ret}

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 6: MULTI-STRATEGY PORTFOLIO TEST")
print("=" * 100)

# Test combining ORB + VWAP as uncorrelated alpha sources
print("\n  Testing ORB + VWAP combined portfolio:")

orb_daily = {}
vwap_daily = {}
for sym in SYMBOLS:
    # ORB
    strat_orb = make_strat(sym)
    df_orb = raw_data[sym].copy()
    df_orb = strat_orb.generate_signals(df_orb)
    r_orb = run_backtest(df_orb, strat_orb, sym)
    if r_orb.daily_returns is not None:
        orb_daily[f"{sym}_ORB"] = r_orb.daily_returns

    # VWAP
    strat_vwap = VWAPReversion(entry_std=1.5, exit_std=0.3)
    df_vwap = raw_data[sym].copy()
    df_vwap = strat_vwap.generate_signals(df_vwap)
    r_vwap = run_backtest(df_vwap, strat_vwap, sym)
    if r_vwap.daily_returns is not None:
        vwap_daily[f"{sym}_VWAP"] = r_vwap.daily_returns

# Correlation between ORB and VWAP returns per symbol
print("\n  Cross-strategy return correlations:")
for sym in SYMBOLS:
    orb_key = f"{sym}_ORB"
    vwap_key = f"{sym}_VWAP"
    if orb_key in orb_daily and vwap_key in vwap_daily:
        aligned = pd.DataFrame({"ORB": orb_daily[orb_key], "VWAP": vwap_daily[vwap_key]}).dropna()
        if len(aligned) > 10:
            corr = aligned.corr().iloc[0, 1]
            print(f"    {sym}: ORB-VWAP correlation = {corr:.3f}")

# Combined portfolio
all_streams = {**orb_daily, **vwap_daily}
if len(all_streams) >= 2:
    combined_df = pd.DataFrame(all_streams).fillna(0)
    combined_ret = combined_df.mean(axis=1)
    combined_sharpe = (combined_ret.mean() / combined_ret.std()) * np.sqrt(252) if combined_ret.std() > 0 else 0
    combined_total = (1 + combined_ret).prod() - 1
    cum = (1 + combined_ret).cumprod()
    combined_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    bench_dr = get_benchmark_daily_returns("SPY", DATA_START, DATA_END)
    aligned = pd.DataFrame({"strat": combined_ret, "bench": bench_dr}).dropna()
    if len(aligned) > 10:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(aligned["bench"], aligned["strat"])
        combined_alpha = intercept * 252
        combined_beta = slope
    else:
        combined_alpha = combined_beta = 0

    n_orb = sum(1 for k in orb_daily)
    n_vwap = sum(1 for k in vwap_daily)
    print(f"\n  ORB-only portfolio ({n_orb} streams):")
    orb_only = pd.DataFrame(orb_daily).fillna(0).mean(axis=1)
    orb_sharpe = (orb_only.mean() / orb_only.std()) * np.sqrt(252) if orb_only.std() > 0 else 0
    orb_total = (1 + orb_only).prod() - 1
    print(f"    Sharpe={orb_sharpe:.2f}  Return={orb_total:+.2%}")

    print(f"\n  ORB+VWAP combined ({n_orb + n_vwap} streams):")
    print(f"    Sharpe={combined_sharpe:.2f}  Return={combined_total:+.2%}  MaxDD={combined_dd:.2%}")
    print(f"    Alpha={combined_alpha:+.2%}  Beta={combined_beta:.3f}")
    print(f"    Diversification lift: {combined_sharpe - orb_sharpe:+.2f} Sharpe")

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 7: SUMMARY TABLE")
print("=" * 100)

print(f"\n  {'Metric':<30} {'Train':>10} {'Val':>10} {'Locked OOS':>12} {'Full':>10}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

metrics_to_show = [
    ("Portfolio Sharpe", "sharpe"),
    ("Portfolio Sortino", "sortino"),
    ("Portfolio Calmar", "calmar"),
    ("Portfolio Return", "total_return"),
    ("Portfolio Annual Return", "annual_return"),
    ("Max Drawdown", "max_drawdown"),
    ("Alpha (annual)", "alpha_annual"),
    ("Beta", "beta"),
    ("Information Ratio", "info_ratio"),
    ("Trades", "num_trades"),
    ("Avg Exposure %", "exposure_pct"),
    ("Benchmark Return", "bench_total"),
    ("Active - Passive", "active_minus_passive"),
]

for label, key in metrics_to_show:
    vals = []
    for period in ["TRAIN", "VALIDATION", "LOCKED_OOS", "FULL"]:
        r = all_results.get(period, {}).get("PORTFOLIO")
        if r and key in r:
            v = r[key]
            if key in ("total_return", "annual_return", "max_drawdown", "alpha_annual",
                        "bench_total", "active_minus_passive"):
                vals.append(f"{v:+.2%}")
            elif key in ("num_trades",):
                vals.append(f"{v:d}")
            elif key in ("exposure_pct",):
                vals.append(f"{v:.1f}%")
            else:
                vals.append(f"{v:.2f}")
        else:
            vals.append("--")
    print(f"  {label:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>12} {vals[3]:>10}")

print("\n" + "=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)
