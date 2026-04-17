#!/usr/bin/env python3
"""Futures ORB backtest runner.

Runs the same ORB strategy on MES/MNQ using SPY/QQQ data as proxy.
Supports walk-forward validation, HTML reporting, and equity comparison.

Usage:
    python run_backtest.py                          # Full backtest, both symbols
    python run_backtest.py --symbols MES             # MES only
    python run_backtest.py --walkforward             # Walk-forward validation
    python run_backtest.py --report                  # Generate HTML report
    python run_backtest.py --oos                     # Locked OOS period only
    python run_backtest.py --sensitivity             # Parameter sensitivity
    python run_backtest.py --compare                 # Equity vs futures comparison
    python run_backtest.py --projections             # Dollar PnL projections
"""

import argparse
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from trading.config import (
    INITIAL_CAPITAL, ORB_SHARED_DEFAULTS, SYMBOL_PROFILES,
    FUTURES_SYMBOLS,
)
from trading.strategies.orb import ORBBreakout
from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.data.contracts import CONTRACTS, total_cost_per_contract
from trading.backtest.engine import run_backtest, BacktestResult
from trading.backtest.walkforward import walk_forward, format_results

ET = pytz.timezone("America/New_York")
log = logging.getLogger(__name__)


def make_orb_for_symbol(futures_symbol: str) -> ORBBreakout:
    """Create ORB strategy with per-symbol profile (same params as equity)."""
    params = dict(ORB_SHARED_DEFAULTS)
    if futures_symbol in SYMBOL_PROFILES:
        params.update(SYMBOL_PROFILES[futures_symbol])
    return ORBBreakout(**params)


def load_data(futures_symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load and prepare data for a futures symbol via its ETF proxy."""
    start_dt = ET.localize(datetime.strptime(start, "%Y-%m-%d"))
    end_dt = ET.localize(datetime.strptime(end, "%Y-%m-%d"))

    df = get_minute_bars(futures_symbol, start_dt, end_dt)
    df = prepare_features(df)
    return df


def run_single_backtest(futures_symbol: str, start: str, end: str,
                        capital: float = INITIAL_CAPITAL,
                        report: bool = False) -> BacktestResult:
    """Run a single backtest for one futures symbol."""
    contract = CONTRACTS[futures_symbol]
    strategy = make_orb_for_symbol(futures_symbol)

    print(f"\n{'='*60}")
    print(f"Backtesting {futures_symbol} ({contract.name})")
    print(f"  Proxy: {contract.proxy_symbol}")
    print(f"  Multiplier: ${contract.multiplier}/point")
    print(f"  Margin: ${contract.margin_intraday:,.0f}/contract")
    print(f"  Round-trip cost: ${total_cost_per_contract(contract):.2f}/contract")
    print(f"  Period: {start} to {end}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Strategy: {strategy.get_params()}")
    print(f"{'='*60}")

    df = load_data(futures_symbol, start, end)
    df = strategy.generate_signals(df)
    result = run_backtest(df, strategy, futures_symbol,
                          initial_capital=capital, contract=contract)

    print_result(result)
    return result


def print_result(r: BacktestResult):
    """Print backtest results."""
    print(f"\n--- {r.symbol} Results ---")
    print(f"  Trades:         {r.num_trades}")
    print(f"  Total Return:   {r.total_return*100:+.2f}%")
    print(f"  Total P&L:      ${r.total_pnl:+,.0f}")
    print(f"  Sharpe Ratio:   {r.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:   {r.max_drawdown*100:.2f}%")
    print(f"  Win Rate:       {r.win_rate*100:.1f}%")
    print(f"  Profit Factor:  {r.profit_factor:.2f}")
    print(f"  Avg Trade:      ${r.avg_trade_dollars:+,.0f} ({r.avg_trade_pct:+.3f}%)")
    print(f"  Avg Contracts:  {r.avg_contracts:.1f}")
    print(f"  Avg Hold (min): {r.avg_bars_held:.0f}")
    print(f"  Exposure:       {r.exposure_pct:.1f}%")
    print(f"  Total Costs:    ${r.total_costs:,.0f}")


def run_equity_comparison(start: str, end: str, capital: float):
    """Run equity backtest for comparison (using the equity engine logic)."""
    from trading.config import INITIAL_CAPITAL

    results = {}
    # Equity engine parameters (same as spy-trader)
    EQUITY_SLIPPAGE = 0.01
    EQUITY_COMMISSION = 0.0
    EQUITY_POSITION_PCT = 0.30

    for proxy_sym, futures_sym in [("SPY", "MES"), ("QQQ", "MNQ")]:
        strategy = make_orb_for_symbol(futures_sym)
        start_dt = ET.localize(datetime.strptime(start, "%Y-%m-%d"))
        end_dt = ET.localize(datetime.strptime(end, "%Y-%m-%d"))
        df = get_minute_bars(futures_sym, start_dt, end_dt)
        df = prepare_features(df)
        df = strategy.generate_signals(df)

        # Run as if equity (shares, not contracts)
        n = len(df)
        eq_capital = capital
        position = 0
        entry_price = 0.0
        entry_time = None
        equity = np.full(n, capital)
        trades = []

        for i in range(1, n):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            prev_signal = int(prev_row["signal"])
            exec_price = float(row["open"])
            date = row["date"]
            eod = row["minute_of_day"] >= 15 * 60 + 50
            new_day = i == 1 or date != df.iloc[i-1]["date"]

            if position != 0:
                exit_reason = None
                if eod or new_day:
                    exit_reason = "eod"
                elif prev_signal == 0:
                    exit_reason = "signal"
                elif np.sign(prev_signal) != np.sign(position):
                    exit_reason = "signal"

                if exit_reason:
                    shares = abs(position)
                    fill = exec_price - EQUITY_SLIPPAGE * np.sign(position)
                    if position > 0:
                        pnl = (fill - entry_price) * shares
                    else:
                        pnl = (entry_price - fill) * shares
                    eq_capital += pnl
                    trades.append({"pnl": pnl, "pnl_pct": pnl / (entry_price * shares) if entry_price > 0 else 0})
                    position = 0

            if position == 0 and not eod and prev_signal != 0 and not new_day:
                shares = int(eq_capital * EQUITY_POSITION_PCT / exec_price)
                if shares > 0:
                    if prev_signal > 0:
                        entry_price = exec_price + EQUITY_SLIPPAGE
                    else:
                        entry_price = exec_price - EQUITY_SLIPPAGE
                    position = shares if prev_signal > 0 else -shares

            if position != 0:
                mark = float(row["close"])
                unrealized = (mark - entry_price) * position
                equity[i] = eq_capital + abs(position) * entry_price + unrealized
            else:
                equity[i] = eq_capital

        equity_series = pd.Series(equity, index=df["dt"])
        daily_eq = equity_series.resample("D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
        total_ret = (eq_capital / capital) - 1
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0
        cummax = equity_series.cummax()
        dd = ((equity_series - cummax) / cummax).min()
        total_pnl = eq_capital - capital

        results[proxy_sym] = {
            "total_return": total_ret,
            "total_pnl": total_pnl,
            "sharpe": sharpe,
            "max_dd": dd,
            "trades": len(trades),
            "win_rate": len([t for t in trades if t["pnl"] > 0]) / len(trades) * 100 if trades else 0,
            "equity_curve": equity_series,
            "daily_returns": daily_ret,
        }

    return results


def run_parameter_sensitivity(futures_symbol: str, start: str, end: str,
                              capital: float):
    """Test ORB edge survival under nearby parameter changes."""
    contract = CONTRACTS[futures_symbol]
    base_params = dict(ORB_SHARED_DEFAULTS)
    if futures_symbol in SYMBOL_PROFILES:
        base_params.update(SYMBOL_PROFILES[futures_symbol])

    df = load_data(futures_symbol, start, end)

    variations = {
        "target_multiple": [1.0, 1.25, 1.5, 1.75, 2.0],
        "range_minutes": [10, 15, 20, 30],
        "min_range_pct": [0.0005, 0.001, 0.0015, 0.002],
        "max_range_pct": [0.006, 0.008, 0.010, 0.012],
    }

    print(f"\n{'='*70}")
    print(f"Parameter Sensitivity: {futures_symbol}")
    print(f"{'='*70}")

    for param_name, values in variations.items():
        print(f"\n--- {param_name} ---")
        print(f"  {'Value':>10} {'Trades':>7} {'P&L':>10} {'Sharpe':>8} {'WR':>6} {'PF':>6}")

        for val in values:
            params = dict(base_params)
            params[param_name] = val
            strategy = ORBBreakout(**params)
            test_df = strategy.generate_signals(df.copy())
            result = run_backtest(test_df, strategy, futures_symbol,
                                  initial_capital=capital, contract=contract)

            marker = " <-- base" if val == base_params.get(param_name) else ""
            print(f"  {val:>10} {result.num_trades:>7} ${result.total_pnl:>+9,.0f} "
                  f"{result.sharpe_ratio:>8.2f} {result.win_rate*100:>5.1f}% "
                  f"{result.profit_factor:>5.2f}{marker}")


def print_projections(results: dict[str, BacktestResult], period_days: int):
    """Print dollar P&L projections at various account sizes."""
    print(f"\n{'='*70}")
    print("DOLLAR P&L PROJECTIONS")
    print(f"{'='*70}")

    account_sizes = [25_000, 50_000, 100_000, 250_000]

    for sym, r in results.items():
        contract = r.contract
        daily_return = r.total_return / max(period_days / 252, 0.01)
        annual_return = (1 + r.total_return) ** (252 / max(period_days, 1)) - 1

        print(f"\n--- {sym} ({contract.name}) ---")
        print(f"  Backtest return: {r.total_return*100:+.2f}% over {period_days} days")
        print(f"  Annualized:      {annual_return*100:+.2f}%")
        print(f"  {'Account':>10} {'6-Month':>12} {'12-Month':>12} {'Max DD$':>12}")

        for size in account_sizes:
            scale = size / INITIAL_CAPITAL
            pnl_6m = r.total_pnl * scale * (126 / max(period_days, 1))
            pnl_12m = r.total_pnl * scale * (252 / max(period_days, 1))
            max_dd_dollars = abs(r.max_drawdown) * size

            print(f"  ${size:>9,} ${pnl_6m:>+11,.0f} ${pnl_12m:>+11,.0f} "
                  f"${max_dd_dollars:>11,.0f}")


def print_break_even(results: dict[str, BacktestResult], period_days: int):
    """Show contracts needed to clear income targets."""
    print(f"\n{'='*70}")
    print("BREAK-EVEN ANALYSIS")
    print(f"{'='*70}")
    targets = [50_000, 100_000, 200_000]

    for sym, r in results.items():
        if r.num_trades == 0:
            continue
        contract = r.contract
        avg_pnl_per_trade = r.avg_trade_dollars
        trades_per_year = r.num_trades * (252 / max(period_days, 1))

        print(f"\n--- {sym} ---")
        print(f"  Avg P&L/trade: ${avg_pnl_per_trade:+,.0f}")
        print(f"  Est. trades/year: {trades_per_year:.0f}")
        print(f"  Avg contracts/trade: {r.avg_contracts:.1f}")

        if avg_pnl_per_trade <= 0:
            print(f"  WARNING: avg trade is negative, cannot break even")
            continue

        # P&L per contract per trade
        pnl_per_contract = avg_pnl_per_trade / max(r.avg_contracts, 1)

        for target in targets:
            needed_per_trade = target / max(trades_per_year, 1)
            contracts_needed = needed_per_trade / max(pnl_per_contract, 0.01)
            margin_needed = contracts_needed * contract.margin_intraday * 2
            print(f"  ${target/1000:.0f}k/yr: ~{contracts_needed:.0f} contracts/trade "
                  f"(~${margin_needed:,.0f} margin needed)")


def main():
    parser = argparse.ArgumentParser(description="Futures ORB Backtest")
    parser.add_argument("--symbols", nargs="+", default=FUTURES_SYMBOLS,
                        help="Futures symbols to test (default: MES MNQ)")
    parser.add_argument("--start", default="2024-01-02",
                        help="Start date (default: 2024-01-02)")
    parser.add_argument("--end", default="2026-04-14",
                        help="End date (default: 2026-04-14)")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help=f"Starting capital (default: ${INITIAL_CAPITAL:,.0f})")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML report")
    parser.add_argument("--oos", action="store_true",
                        help="Run locked OOS period only (Dec 2025 - Apr 2026)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run parameter sensitivity analysis")
    parser.add_argument("--compare", action="store_true",
                        help="Compare equity vs futures results")
    parser.add_argument("--projections", action="store_true",
                        help="Show dollar P&L projections")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.oos:
        args.start = "2025-12-01"
        args.end = "2026-04-14"
        print("\n*** LOCKED OUT-OF-SAMPLE PERIOD: Dec 2025 - Apr 2026 ***\n")

    # Run backtests
    results = {}
    for sym in args.symbols:
        if args.walkforward:
            df = load_data(sym, args.start, args.end)
            contract = CONTRACTS[sym]
            strategy = make_orb_for_symbol(sym)
            wf = walk_forward(df, strategy, sym, contract=contract,
                              initial_capital=args.capital)
            print(format_results(wf))
        else:
            result = run_single_backtest(sym, args.start, args.end,
                                         capital=args.capital)
            results[sym] = result

    # Portfolio summary
    if len(results) > 1:
        total_pnl = sum(r.total_pnl for r in results.values())
        total_trades = sum(r.num_trades for r in results.values())
        total_costs = sum(r.total_costs for r in results.values())
        avg_sharpe = np.mean([r.sharpe_ratio for r in results.values()])

        print(f"\n{'='*60}")
        print("PORTFOLIO SUMMARY (MES + MNQ)")
        print(f"{'='*60}")
        print(f"  Total Trades:    {total_trades}")
        print(f"  Total P&L:       ${total_pnl:+,.0f}")
        print(f"  Total Costs:     ${total_costs:,.0f}")
        print(f"  Avg Sharpe:      {avg_sharpe:.2f}")
        print(f"  P&L after costs: ${total_pnl:+,.0f}")

    # Equity comparison
    if args.compare and results:
        print(f"\n{'='*70}")
        print("EQUITY vs FUTURES COMPARISON")
        print(f"{'='*70}")

        equity_results = run_equity_comparison(args.start, args.end, args.capital)

        print(f"\n  {'Metric':<20} {'SPY (Equity)':>15} {'MES (Futures)':>15} "
              f"{'QQQ (Equity)':>15} {'MNQ (Futures)':>15}")
        print(f"  {'-'*80}")

        for eq_sym, fut_sym in [("SPY", "MES"), ("QQQ", "MNQ")]:
            if fut_sym not in results or eq_sym not in equity_results:
                continue
            er = equity_results[eq_sym]
            fr = results[fut_sym]

            if eq_sym == "SPY":
                print(f"  {'Return':<20} {er['total_return']*100:>14.2f}% {fr.total_return*100:>14.2f}%", end="")
            else:
                print(f" {er['total_return']*100:>14.2f}% {fr.total_return*100:>14.2f}%")

        for eq_sym, fut_sym in [("SPY", "MES"), ("QQQ", "MNQ")]:
            if fut_sym not in results or eq_sym not in equity_results:
                continue
            er = equity_results[eq_sym]
            fr = results[fut_sym]

        print()
        for label, field_eq, field_fut in [
            ("Total P&L", "total_pnl", "total_pnl"),
            ("Sharpe", "sharpe", "sharpe_ratio"),
            ("Max Drawdown", "max_dd", "max_drawdown"),
            ("Trades", "trades", "num_trades"),
            ("Win Rate", "win_rate", "win_rate"),
        ]:
            line = f"  {label:<20}"
            for eq_sym, fut_sym in [("SPY", "MES"), ("QQQ", "MNQ")]:
                if fut_sym not in results or eq_sym not in equity_results:
                    continue
                er = equity_results[eq_sym]
                fr = results[fut_sym]

                eq_val = er[field_eq]
                fut_val = getattr(fr, field_fut)

                if "pnl" in label.lower():
                    line += f" ${eq_val:>+13,.0f} ${fut_val:>+13,.0f}"
                elif "drawdown" in label.lower():
                    line += f" {eq_val*100:>14.2f}% {fut_val*100:>14.2f}%"
                elif "rate" in label.lower():
                    eq_pct = eq_val if eq_val > 1 else eq_val * 100
                    fut_pct = fut_val if fut_val > 1 else fut_val * 100
                    line += f" {eq_pct:>14.1f}% {fut_pct:>14.1f}%"
                elif "sharpe" in label.lower():
                    line += f" {eq_val:>15.2f} {fut_val:>15.2f}"
                else:
                    line += f" {eq_val:>15} {fut_val:>15}"
            print(line)

    # Sensitivity analysis
    if args.sensitivity:
        for sym in args.symbols:
            run_parameter_sensitivity(sym, args.start, args.end, args.capital)

    # Projections
    if args.projections and results:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        period_days = (end_dt - start_dt).days
        print_projections(results, period_days)
        print_break_even(results, period_days)

    # Risk analysis
    if results:
        print(f"\n{'='*70}")
        print("RISK ANALYSIS: WORST-CASE SCENARIOS")
        print(f"{'='*70}")

        for sym, r in results.items():
            if not r.trades:
                continue

            trade_pnls = [t.pnl for t in r.trades]
            daily_eq = r.equity_curve.resample("D").last().dropna()
            daily_pnl = daily_eq.diff().dropna()

            print(f"\n--- {sym} ---")
            print(f"  Worst single trade:   ${min(trade_pnls):+,.0f}")
            print(f"  Best single trade:    ${max(trade_pnls):+,.0f}")
            print(f"  Worst day:            ${daily_pnl.min():+,.0f}")
            print(f"  Best day:             ${daily_pnl.max():+,.0f}")

            # Worst week
            weekly_pnl = daily_eq.resample("W").last().dropna().diff().dropna()
            if len(weekly_pnl) > 0:
                print(f"  Worst week:           ${weekly_pnl.min():+,.0f}")
                print(f"  Best week:            ${weekly_pnl.max():+,.0f}")

            # Max consecutive losses
            max_consec = 0
            current_consec = 0
            for t in r.trades:
                if t.pnl <= 0:
                    current_consec += 1
                    max_consec = max(max_consec, current_consec)
                else:
                    current_consec = 0
            print(f"  Max consecutive losses: {max_consec}")

            # Max drawdown in dollars
            dd_dollars = (r.equity_curve - r.equity_curve.cummax()).min()
            print(f"  Max drawdown ($):     ${dd_dollars:+,.0f}")

    # Generate report
    if args.report and results:
        from trading.reporting.dashboard import generate_report
        report_results = list(results.values())
        comparison_data = None

        if args.compare:
            equity_results = run_equity_comparison(args.start, args.end, args.capital)
            comparison_data = {}
            for eq_sym, fut_sym in [("SPY", "MES"), ("QQQ", "MNQ")]:
                if fut_sym in results and eq_sym in equity_results:
                    er = equity_results[eq_sym]
                    fr = results[fut_sym]
                    comparison_data[f"{eq_sym}/{fut_sym} Return"] = {
                        "equity": f"{er['total_return']*100:+.2f}%",
                        "futures": f"{fr.total_return*100:+.2f}%",
                        "better": "futures" if fr.total_return > er["total_return"] else "equity"
                    }
                    comparison_data[f"{eq_sym}/{fut_sym} P&L"] = {
                        "equity": f"${er['total_pnl']:+,.0f}",
                        "futures": f"${fr.total_pnl:+,.0f}",
                        "better": "futures" if fr.total_pnl > er["total_pnl"] else "equity"
                    }
                    comparison_data[f"{eq_sym}/{fut_sym} Sharpe"] = {
                        "equity": f"{er['sharpe']:.2f}",
                        "futures": f"{fr.sharpe_ratio:.2f}",
                        "better": "futures" if fr.sharpe_ratio > er["sharpe"] else "equity"
                    }

        path = generate_report(report_results, comparison=comparison_data)
        print(f"\nReport saved: {path}")


if __name__ == "__main__":
    main()
