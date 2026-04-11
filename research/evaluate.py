"""Strategy evaluation and comparison.

Pulls historical data, runs walk-forward backtests on all strategies,
and compares them side by side.
"""

import logging
import sys
import os
from datetime import datetime

import pytz

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.data.provider import get_minute_bars
from trading.data.features import prepare_features
from trading.strategies.vwap_reversion import VWAPReversion
from trading.strategies.orb import ORBBreakout
from trading.strategies.rsi_reversion import RSIReversion
from trading.backtest.engine import run_backtest
from trading.backtest.walkforward import walk_forward, format_results
from trading.config import INITIAL_CAPITAL

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_evaluation():
    """Main research pipeline: pull data, test strategies, compare."""
    setup_logging()
    log.info("=" * 70)
    log.info("STRATEGY EVALUATION")
    log.info("=" * 70)

    symbols = ["SPY", "QQQ"]

    # Date range: ~12 months of data for walk-forward
    start = datetime(2025, 1, 2, tzinfo=ET)
    end = datetime(2026, 4, 4, tzinfo=ET)

    # ── Step 1: Pull data ──
    log.info("\n--- STEP 1: Pulling historical data ---")
    data = {}
    for sym in symbols:
        log.info("Fetching %s...", sym)
        df = get_minute_bars(sym, start, end, use_cache=True)
        log.info("  %s: %d bars, %s to %s", sym, len(df),
                 df["dt"].iloc[0].date(), df["dt"].iloc[-1].date())
        data[sym] = df

    # ── Step 2: Feature engineering ──
    log.info("\n--- STEP 2: Feature engineering ---")
    for sym in symbols:
        data[sym] = prepare_features(data[sym])
        log.info("  %s: %d bars with features", sym, len(data[sym]))

    # ── Step 3: Define strategies ──
    strategies = [
        VWAPReversion(entry_std=1.5, exit_std=0.3),
        VWAPReversion(entry_std=2.0, exit_std=0.5),
        ORBBreakout(range_minutes=15, target_multiple=1.5),
        ORBBreakout(range_minutes=30, target_multiple=1.0),
        RSIReversion(rsi_period=14, oversold=25, overbought=75),
        RSIReversion(rsi_period=10, oversold=20, overbought=80),
    ]

    # ── Step 4: Walk-forward testing ──
    log.info("\n--- STEP 3: Walk-forward backtesting ---")
    results = []

    for sym in symbols:
        df = data[sym]
        for strat in strategies:
            log.info("\nTesting %s on %s (params=%s)...", strat.name, sym, strat.get_params())

            # Generate signals
            df_sig = strat.generate_signals(df.copy())

            try:
                wf = walk_forward(
                    df_sig, strat, sym,
                    train_days=60, test_days=20, step_days=20,
                )
                results.append(wf)
                print(format_results(wf))
            except Exception as e:
                log.error("  Walk-forward failed: %s", e)

    # ── Step 5: Comparison ──
    log.info("\n\n" + "=" * 80)
    log.info("STRATEGY COMPARISON (Walk-Forward OOS Results)")
    log.info("=" * 80)

    if not results:
        log.error("No results to compare!")
        return

    # Sort by Sharpe ratio
    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

    header = f"{'Strategy':<25} {'Symbol':<6} {'Trades':>7} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'WinR':>6} {'PF':>6} {'AvgTr':>7} {'Exp%':>5}"
    log.info(header)
    log.info("-" * len(header))

    for r in results:
        log.info(
            f"{r.strategy_name:<25} {r.symbol:<6} {r.total_trades:>7} "
            f"{r.total_return*100:>+7.2f}% {r.sharpe_ratio:>7.2f} "
            f"{r.max_drawdown*100:>6.2f}% {r.win_rate*100:>5.1f}% "
            f"{r.profit_factor:>5.2f} {r.avg_trade_pct:>+6.3f}% {r.exposure_pct:>4.1f}%"
        )

    # ── Step 6: Winner selection ──
    log.info("\n\n--- WINNER SELECTION ---")

    # Filter: need at least 50 trades for statistical relevance
    viable = [r for r in results if r.total_trades >= 50]
    if not viable:
        viable = [r for r in results if r.total_trades >= 20]

    if not viable:
        log.warning("No strategy produced enough trades for statistical confidence.")
        log.warning("Best available:")
        best = results[0]
    else:
        # Rank by Sharpe, but penalize very high values (likely overfit)
        # A realistic intraday Sharpe is 0.5-2.0
        def score(r):
            sharpe_score = min(r.sharpe_ratio, 3.0)  # Cap extreme Sharpes
            trade_score = min(r.total_trades / 100, 1.0)  # Reward more trades
            dd_penalty = max(0, -r.max_drawdown - 0.05) * 10  # Penalize >5% DD
            return sharpe_score * trade_score - dd_penalty

        viable.sort(key=score, reverse=True)
        best = viable[0]

    log.info("Selected: %s on %s", best.strategy_name, best.symbol)
    log.info("  Sharpe: %.2f", best.sharpe_ratio)
    log.info("  Return: %.2f%%", best.total_return * 100)
    log.info("  Win Rate: %.1f%%", best.win_rate * 100)
    log.info("  Profit Factor: %.2f", best.profit_factor)
    log.info("  Max Drawdown: %.2f%%", best.max_drawdown * 100)
    log.info("  Total Trades: %d", best.total_trades)
    log.info("  Avg Trade: %+.3f%%", best.avg_trade_pct)

    # Honest assessment
    log.info("\n--- HONEST ASSESSMENT ---")
    if best.sharpe_ratio < 0.5:
        log.warning("Sharpe < 0.5: Weak or no edge detected. Do NOT trade live without further research.")
    elif best.sharpe_ratio < 1.0:
        log.info("Sharpe 0.5-1.0: Marginal edge. Paper trade extensively before going live.")
    elif best.sharpe_ratio < 2.0:
        log.info("Sharpe 1.0-2.0: Reasonable edge. Paper trade to verify, then consider small live.")
    else:
        log.warning("Sharpe > 2.0: Suspiciously high. Likely overfit. Investigate before trusting.")

    if best.total_trades < 100:
        log.warning("Only %d trades — insufficient sample for statistical confidence.", best.total_trades)

    if best.profit_factor < 1.0:
        log.warning("Profit factor < 1.0: Strategy loses money after costs.")
    elif best.profit_factor < 1.2:
        log.warning("Profit factor 1.0-1.2: Very thin edge, easily eroded by real-world friction.")

    return results


if __name__ == "__main__":
    run_evaluation()
