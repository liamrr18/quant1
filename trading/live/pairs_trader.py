"""Pairs spread live trader.

Subclass of LiveTrader that handles the special requirement of pairs
trading: two instruments' data must be merged before signal generation.

The primary symbol is traded; the secondary symbol provides the
'pair_close' column used for spread calculation.
"""

import logging

import pandas as pd

from trading.live.trader import (
    LiveTrader, ActiveTrade, fetch_live_bars, now_et,
    get_account_equity, get_all_positions,
)
from trading.data.features import prepare_features

log = logging.getLogger(__name__)


class PairsLiveTrader(LiveTrader):
    """Live trader for pairs spread strategies.

    Overrides signal scanning and exit checking to merge two instruments'
    data before generating signals. Only the primary symbol is traded.
    """

    def __init__(self, strategy, primary_symbol: str, secondary_symbol: str,
                 dry_run: bool = False, log_base_dir: str = None):
        super().__init__(
            strategy=strategy,
            symbols=[primary_symbol],
            dry_run=dry_run,
            log_base_dir=log_base_dir,
        )
        self.primary_symbol = primary_symbol
        self.secondary_symbol = secondary_symbol

    def _fetch_merged_bars(self, lookback_minutes: int = 200) -> pd.DataFrame | None:
        """Fetch bars for both instruments and merge into a single DataFrame.

        Returns a DataFrame with 'pair_close' column from the secondary
        instrument joined on timestamp. Returns None if data is insufficient
        or timestamps diverge too much.
        """
        try:
            df_primary = fetch_live_bars(self.primary_symbol, lookback_minutes)
            df_secondary = fetch_live_bars(self.secondary_symbol, lookback_minutes)
        except Exception as e:
            log.error("Error fetching bars for pairs: %s", e)
            return None

        if len(df_primary) < 30 or len(df_secondary) < 30:
            log.debug("SKIP pairs: primary=%d bars, secondary=%d bars (need 30+)",
                      len(df_primary), len(df_secondary))
            return None

        # Check timestamp alignment — skip if latest bars diverge > 2 minutes
        latest_primary = df_primary["dt"].iloc[-1]
        latest_secondary = df_secondary["dt"].iloc[-1]
        diff_minutes = abs((latest_primary - latest_secondary).total_seconds()) / 60
        if diff_minutes > 2:
            log.warning("SKIP pairs: timestamp divergence %.1f minutes "
                        "(primary=%s, secondary=%s)",
                        diff_minutes, latest_primary, latest_secondary)
            return None

        # Merge: add secondary close as 'pair_close' column
        pair_close = df_secondary.set_index("dt")["close"].rename("pair_close")
        df = df_primary.set_index("dt").join(pair_close, how="left").reset_index()
        df["pair_close"] = df["pair_close"].ffill()

        return df

    def _check_strategy_exits(self):
        """Override: check exits using merged pairs data."""
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades[symbol]
            try:
                df = self._fetch_merged_bars(lookback_minutes=200)
                if df is None:
                    log.warning("Cannot check exits for %s: merged data unavailable", symbol)
                    continue

                df = prepare_features(df)
                strat = self._get_strategy(symbol)
                df = strat.generate_signals(df)

                last_signal = int(df["signal"].iloc[-1])
                expected_signal = 1 if trade.direction == "long" else -1

                if last_signal == 0:
                    log.info("STRATEGY EXIT: %s signal -> 0", symbol)
                    self._close_trade(symbol, "strategy_exit")
                elif last_signal == -expected_signal:
                    log.info("STRATEGY FLIP: %s signal flipped to %d", symbol, last_signal)
                    self._close_trade(symbol, "strategy_flip")

            except Exception as e:
                log.error("Error checking strategy exit for %s: %s", symbol, e)

    def _scan_for_signals(self):
        """Override: scan for signals using merged pairs data."""
        # Global position check
        try:
            all_positions = get_all_positions()
            from trading.config import MAX_CONCURRENT_POSITIONS
            if len(all_positions) >= MAX_CONCURRENT_POSITIONS:
                log.warning("GLOBAL LIMIT: %d positions across all instances (max %d)",
                           len(all_positions), MAX_CONCURRENT_POSITIONS)
                return
        except Exception as e:
            log.error("Error checking global positions: %s", e)

        symbol = self.primary_symbol

        if symbol in self.active_trades:
            return  # Exits handled by _check_strategy_exits

        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            log.debug("BLOCKED %s: %s", symbol, reason)
            self._log_signal(symbol, 0, 0, 0, 0, False, "blocked", reason)
            return

        try:
            df = self._fetch_merged_bars(lookback_minutes=200)
            if df is None:
                self._log_signal(symbol, 0, 0, 0, 0, False, "insufficient_data",
                                 "merged data unavailable")
                return

            df = prepare_features(df)
            strat = self._get_strategy(symbol)
            df = strat.generate_signals(df)

            last_signal = int(df["signal"].iloc[-1])
            price = float(df["close"].iloc[-1])

            # Pairs don't have or_high/or_low — log zeros
            if last_signal == 0:
                self._log_signal(symbol, 0, price, 0, 0, False, "no_signal", "")
                return

            shares = self.risk_manager.calculate_shares(price)
            if shares == 0:
                log.info("SKIP %s: shares=0 (price=$%.2f)", symbol, price)
                self._log_signal(symbol, last_signal, price, 0, 0,
                                 False, "zero_shares", f"price={price:.2f}")
                return

            direction = "long" if last_signal > 0 else "short"
            side = "buy" if direction == "long" else "sell"

            log.info("SIGNAL: %s %s %d shares @ ~$%.2f (pairs spread)",
                     direction.upper(), symbol, shares, price)
            self._log_signal(symbol, last_signal, price, 0, 0,
                             False, "entry_signal", f"{direction} {shares} shares")

            if self.dry_run:
                log.info("DRY RUN: would submit %s %d %s", side, shares, symbol)
                return

            from trading.execution.broker import submit_market_order, wait_for_fill
            order_id = submit_market_order(symbol, shares, side)
            if order_id is None:
                log.error("ORDER FAILED: %s %s", side, symbol)
                self._log_trade(symbol, direction, "fill_failed", shares, price, None, "order_failed", None)
                self.risk_manager.record_order_rejection()
                return

            fill = wait_for_fill(order_id, timeout_sec=30)
            if fill and fill.get("status") == "filled":
                fill_price = fill["filled_avg_price"]
                self.active_trades[symbol] = ActiveTrade(
                    symbol=symbol,
                    direction=direction,
                    shares=shares,
                    entry_price=fill_price,
                    entry_time=now_et(),
                    order_id=order_id,
                )
                log.info("FILLED: %s %s %d @ $%.2f", direction.upper(), symbol, shares, fill_price)
                self._log_trade(symbol, direction, "entry", shares, fill_price, None, "filled", order_id)
            else:
                log.error("FILL TIMEOUT: %s %s, status=%s", side, symbol,
                         fill.get("status") if fill else "timeout")
                self._log_trade(symbol, direction, "fill_failed", shares, price, None,
                               "fill_timeout", order_id)

        except Exception as e:
            log.error("Error scanning pairs signal for %s: %s", symbol, e)
