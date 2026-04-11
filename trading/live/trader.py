"""Live trading loop.

Runs during market hours, generates signals from real-time data,
executes through Alpaca, and manages positions with risk controls.
"""

import logging
import time
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import pytz
import pandas as pd

from trading.config import (
    SYMBOLS, TRADE_START, TRADE_END, FORCE_CLOSE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)
from trading.execution.broker import (
    get_client, get_account_equity, get_position, get_all_positions,
    submit_market_order, wait_for_fill, close_position, close_all_positions,
    cancel_all_orders, is_market_open,
)
from trading.risk.manager import RiskManager
from trading.data.provider import get_client as get_data_client
from trading.data.features import prepare_features

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


@dataclass
class ActiveTrade:
    symbol: str
    direction: str
    shares: int
    entry_price: float
    entry_time: datetime
    order_id: str


def now_et() -> datetime:
    return datetime.now(ET)


def time_str_to_minutes(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def current_minute() -> int:
    now = now_et()
    return now.hour * 60 + now.minute


def fetch_live_bars(symbol: str, lookback_minutes: int = 120) -> pd.DataFrame:
    """Fetch recent minute bars for live signal generation."""
    from trading.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    end = datetime.now(pytz.UTC)
    # Fetch extra to ensure enough data after filtering
    start = end - timedelta(minutes=lookback_minutes + 60)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()
    df = df.rename(columns={"timestamp": "dt"})
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])

    if df["dt"].dt.tz is not None:
        df["dt"] = df["dt"].dt.tz_convert(ET)
    else:
        df["dt"] = df["dt"].dt.tz_localize("UTC").dt.tz_convert(ET)

    # Filter to trading hours
    times = df["dt"].dt.strftime("%H:%M")
    df = df[(times >= "09:30") & (times < "16:00")]
    df = df.sort_values("dt").reset_index(drop=True)
    return df


class LiveTrader:
    """Main live trading orchestrator."""

    def __init__(self, strategy, symbols: list[str] = None, dry_run: bool = False):
        self.strategy = strategy
        self.symbols = symbols or SYMBOLS
        self.dry_run = dry_run
        self.active_trades: dict[str, ActiveTrade] = {}
        self.risk_manager: RiskManager | None = None
        self.running = True

    def startup(self):
        """Safe startup: verify account, check positions, initialize risk."""
        log.info("=" * 60)
        log.info("STARTING LIVE TRADER: %s", self.strategy.name)
        log.info("Symbols: %s", self.symbols)
        log.info("Dry run: %s", self.dry_run)
        log.info("=" * 60)

        # Verify API connection
        try:
            equity = get_account_equity()
        except Exception as e:
            log.error("Cannot connect to Alpaca: %s", e)
            sys.exit(1)

        self.risk_manager = RiskManager(equity)

        # Check for existing positions
        positions = get_all_positions()
        if positions:
            log.warning("Found %d existing positions at startup:", len(positions))
            for p in positions:
                log.warning("  %s: %d shares, P&L $%.2f",
                           p["symbol"], p["qty"], p["unrealized_pl"])

        log.info("Startup complete. Equity: $%.0f", equity)

    def shutdown(self):
        """Safe shutdown: close all positions."""
        log.info("Shutting down...")
        self.running = False

        if not self.dry_run:
            # Close all tracked positions
            for symbol, trade in list(self.active_trades.items()):
                log.info("Closing position: %s", symbol)
                close_position(symbol)

            # Safety: close anything else
            remaining = get_all_positions()
            if remaining:
                log.warning("Closing %d remaining positions", len(remaining))
                close_all_positions()

            cancel_all_orders()

        log.info("Shutdown complete.")

    def run(self):
        """Main trading loop. Runs until market close or interrupted."""
        self.startup()

        try:
            self._wait_for_trading_start()
            self._trading_loop()
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        except Exception:
            log.exception("Unexpected error in trading loop")
        finally:
            self.shutdown()

    def _wait_for_trading_start(self):
        """Wait until trade start time."""
        start_min = time_str_to_minutes(TRADE_START)
        while self.running and current_minute() < start_min:
            if not is_market_open():
                log.info("Market not open yet. Waiting...")
                time.sleep(30)
                continue
            remaining = start_min - current_minute()
            log.info("Waiting %d minutes until trade start (%s ET)", remaining, TRADE_START)
            time.sleep(min(remaining * 60, 30))

    def _trading_loop(self):
        """Core loop: generate signals, manage positions, check risk."""
        trade_end_min = time_str_to_minutes(TRADE_END)
        force_close_min = time_str_to_minutes(FORCE_CLOSE)

        while self.running:
            now_min = current_minute()

            # Force close time
            if now_min >= force_close_min:
                log.info("Force close time reached (%s ET)", FORCE_CLOSE)
                if not self.dry_run:
                    for symbol in list(self.active_trades.keys()):
                        self._close_trade(symbol, "eod")
                break

            # Update risk state
            try:
                equity = get_account_equity()
                self.risk_manager.update_equity(equity)
                self.risk_manager.update_positions(len(self.active_trades))
            except Exception as e:
                log.error("Error updating risk state: %s", e)
                self.risk_manager.record_api_error()

            # Check if halted
            if self.risk_manager.state.halted:
                log.warning("Risk manager halted: %s", self.risk_manager.state.halt_reason)
                # Close everything and stop
                if not self.dry_run:
                    for symbol in list(self.active_trades.keys()):
                        self._close_trade(symbol, "halt")
                break

            # Check active positions for stops/targets
            self._check_positions()

            # Generate new signals (only before trade end time)
            if now_min < trade_end_min:
                self._scan_for_signals()

            # Log status every iteration
            status = self.risk_manager.status()
            log.info("Status: pnl=$%.0f, positions=%d, trades=%d, halted=%s",
                     status["daily_pnl"], status["open_positions"],
                     status["trades_today"], status["halted"])

            # Sleep between iterations
            time.sleep(60)  # Check every minute

    def _scan_for_signals(self):
        """Fetch current data and check for entry signals."""
        for symbol in self.symbols:
            if symbol in self.active_trades:
                continue

            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                continue

            try:
                df = fetch_live_bars(symbol, lookback_minutes=200)
                if len(df) < 30:
                    continue

                df = prepare_features(df)
                df = self.strategy.generate_signals(df)

                last_signal = int(df["signal"].iloc[-1])
                if last_signal == 0:
                    continue

                price = float(df["close"].iloc[-1])
                shares = self.risk_manager.calculate_shares(price)
                if shares == 0:
                    continue

                direction = "long" if last_signal > 0 else "short"
                side = "buy" if direction == "long" else "sell"

                log.info("SIGNAL: %s %s %d shares @ ~$%.2f",
                         direction.upper(), symbol, shares, price)

                if self.dry_run:
                    log.info("[DRY RUN] Would %s %d %s", side, shares, symbol)
                    continue

                order_id = submit_market_order(symbol, shares, side)
                if order_id is None:
                    self.risk_manager.record_order_rejection()
                    continue

                fill = wait_for_fill(order_id)
                if fill and fill.get("status") == "filled":
                    self.active_trades[symbol] = ActiveTrade(
                        symbol=symbol,
                        direction=direction,
                        shares=fill["filled_qty"],
                        entry_price=fill["filled_avg_price"],
                        entry_time=now_et(),
                        order_id=order_id,
                    )
                    log.info("FILLED: %s %s %d @ $%.2f",
                             direction.upper(), symbol,
                             fill["filled_qty"], fill["filled_avg_price"])
                else:
                    self.risk_manager.record_order_rejection()

            except Exception as e:
                log.error("Error scanning %s: %s", symbol, e)
                self.risk_manager.record_api_error()

    def _check_positions(self):
        """Check active positions for stop loss / take profit."""
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades[symbol]

            try:
                pos = get_position(symbol)
                if pos is None:
                    log.warning("%s: position disappeared", symbol)
                    del self.active_trades[symbol]
                    continue

                current_price = abs(pos["market_value"]) / abs(pos["qty"]) if pos["qty"] != 0 else trade.entry_price

                if self.risk_manager.check_stop_loss(trade.entry_price, current_price, trade.direction):
                    log.warning("STOP LOSS: %s @ $%.2f (entry $%.2f)",
                               symbol, current_price, trade.entry_price)
                    self._close_trade(symbol, "stop")

                elif self.risk_manager.check_take_profit(trade.entry_price, current_price, trade.direction):
                    log.info("TAKE PROFIT: %s @ $%.2f (entry $%.2f)",
                            symbol, current_price, trade.entry_price)
                    self._close_trade(symbol, "target")

            except Exception as e:
                log.error("Error checking %s: %s", symbol, e)
                self.risk_manager.record_api_error()

    def _close_trade(self, symbol: str, reason: str):
        """Close a trade and record the result."""
        if self.dry_run:
            log.info("[DRY RUN] Would close %s (%s)", symbol, reason)
            if symbol in self.active_trades:
                del self.active_trades[symbol]
            return

        trade = self.active_trades.get(symbol)
        if trade is None:
            return

        order_id = close_position(symbol)
        if order_id:
            fill = wait_for_fill(order_id)
            if fill and fill.get("status") == "filled":
                exit_price = fill["filled_avg_price"]
                if trade.direction == "long":
                    pnl = (exit_price - trade.entry_price) * trade.shares
                else:
                    pnl = (trade.entry_price - exit_price) * trade.shares

                self.risk_manager.record_trade(pnl)
                log.info("CLOSED %s %s: entry=$%.2f exit=$%.2f pnl=$%.2f (%s)",
                         trade.direction.upper(), symbol, trade.entry_price,
                         exit_price, pnl, reason)

        del self.active_trades[symbol]
