"""Live trading loop.

Runs during market hours, generates signals from real-time data,
executes through Alpaca, and manages positions with risk controls.

CRITICAL: Strategy exits (target, stop, stale) are executed by re-running
signal generation each iteration and detecting signal -> 0 transitions.
The risk manager's 2% hard stop/TP is a backstop only.
"""

import csv
import json
import logging
import os
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
    get_ib, get_account_equity, get_position, get_all_positions,
    submit_market_order, wait_for_fill, close_position, close_all_positions,
    cancel_all_orders, is_market_open, _resolve_stock,
)
from trading.risk.manager import RiskManager
from trading.data.features import prepare_features
from trading.alerts import alert_trade, alert_halt, alert_eod_summary, is_enabled as alerts_enabled
from trading.discord_alerts import discord_trade, discord_halt, discord_eod

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

# ── Log directory setup ──
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")


def _ensure_log_dir(base_dir=None):
    base = base_dir or LOG_DIR
    os.makedirs(base, exist_ok=True)
    date_dir = os.path.join(base, now_et().strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)
    return date_dir


def _today_log_dir(base_dir=None):
    base = base_dir or LOG_DIR
    return os.path.join(base, now_et().strftime("%Y-%m-%d"))


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
    """Fetch recent minute bars from IB for live signal generation."""
    ib = get_ib()
    contract = _resolve_stock(symbol)

    duration = "1 D" if lookback_minutes <= 390 else "2 D"
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
    except Exception as e:
        log.error("IB bar fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()

    if not bars:
        log.warning("No bars returned for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame([{
        "dt": b.date,
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": b.volume,
    } for b in bars])

    df["dt"] = pd.to_datetime(df["dt"])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize(ET)
    else:
        df["dt"] = df["dt"].dt.tz_convert(ET)

    # Filter to trading hours
    times = df["dt"].dt.strftime("%H:%M")
    df = df[(times >= "09:30") & (times < "16:00")]
    df = df.sort_values("dt").reset_index(drop=True)

    log.debug("Got %d IB bars for %s", len(df), symbol)
    return df


class LiveTrader:
    """Main live trading orchestrator.

    Key design: the ORB strategy's own exits (target, stop, stale exit) are
    the primary exit mechanism. Each iteration, we re-run signal generation
    for symbols with active trades. If the signal transitions to 0, we close.
    The risk manager's 2% hard stop/TP is a backstop only.
    """

    def __init__(self, strategy=None, symbols: list[str] = None, dry_run: bool = False,
                 strategies: dict = None, log_base_dir: str = None):
        self.strategies = strategies or {}
        self.default_strategy = strategy
        self.symbols = symbols or SYMBOLS
        self.dry_run = dry_run
        self.active_trades: dict[str, ActiveTrade] = {}
        self.risk_manager: RiskManager | None = None
        self.running = True
        self._log_base_dir = log_base_dir or LOG_DIR
        self._trade_timestamps: dict[str, list] = {}  # symbol -> list of trade times (churn detection)

        # Logging state
        self._log_dir = None
        self._trade_journal_path = None
        self._equity_log_path = None
        self._signal_log_path = None
        self._iteration_count = 0
        self._session_start = None

    def _get_strategy(self, symbol: str):
        return self.strategies.get(symbol, self.default_strategy)

    # ── Trade Journal (CSV) ──

    def _init_trade_journal(self):
        self._trade_journal_path = os.path.join(self._log_dir, "trades.csv")
        if not os.path.exists(self._trade_journal_path):
            with open(self._trade_journal_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "symbol", "direction", "action", "shares",
                    "price", "pnl", "reason", "order_id",
                ])

    def _log_trade(self, symbol, direction, action, shares, price, pnl, reason, order_id):
        if self._trade_journal_path is None:
            return
        with open(self._trade_journal_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                now_et().isoformat(), symbol, direction, action, shares,
                f"{price:.2f}", f"{pnl:.2f}" if pnl is not None else "",
                reason, order_id or "",
            ])

    # ── Equity Log ──

    def _init_equity_log(self):
        self._equity_log_path = os.path.join(self._log_dir, "equity.csv")
        if not os.path.exists(self._equity_log_path):
            with open(self._equity_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "equity", "daily_pnl", "open_positions", "trades_today"])

    def _log_equity(self):
        if self._equity_log_path is None or self.risk_manager is None:
            return
        status = self.risk_manager.status()
        with open(self._equity_log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                now_et().isoformat(),
                f"{self.risk_manager.state.current_equity:.2f}",
                f"{status['daily_pnl']:.2f}",
                status["open_positions"],
                status["trades_today"],
            ])

    # ── Signal Log ──

    def _init_signal_log(self):
        self._signal_log_path = os.path.join(self._log_dir, "signals.csv")
        if not os.path.exists(self._signal_log_path):
            with open(self._signal_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "symbol", "signal", "price", "or_high", "or_low",
                    "range_pct", "in_position", "action_taken", "reason",
                ])

    def _log_signal(self, symbol, signal, price, or_high, or_low, in_position,
                    action_taken, reason=""):
        if self._signal_log_path is None:
            return
        range_pct = (or_high - or_low) / or_low * 100 if or_low > 0 and or_high > 0 else 0
        with open(self._signal_log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                now_et().isoformat(), symbol, signal, f"{price:.2f}",
                f"{or_high:.2f}", f"{or_low:.2f}", f"{range_pct:.4f}",
                in_position, action_taken, reason,
            ])

    # ── Startup / Shutdown ──

    def startup(self):
        self._session_start = now_et()
        self._log_dir = _ensure_log_dir(self._log_base_dir)
        self._init_trade_journal()
        self._init_equity_log()
        self._init_signal_log()

        log.info("=" * 70)
        log.info("LIVE TRADER STARTING")
        log.info("  Session: %s", self._session_start.strftime("%Y-%m-%d %H:%M:%S ET"))
        log.info("  Log dir: %s", self._log_dir)
        log.info("  Dry run: %s", self.dry_run)
        log.info("  Symbols: %s", self.symbols)
        for sym in self.symbols:
            strat = self._get_strategy(sym)
            log.info("  %s: %s %s", sym, strat.name, strat.get_params())
        log.info("=" * 70)

        # Verify API connection
        try:
            equity = get_account_equity()
        except Exception as e:
            log.error("Cannot connect to IB: %s", e)
            sys.exit(1)

        self.risk_manager = RiskManager(equity)

        # Reconcile: adopt existing IB positions to prevent duplicate orders
        positions = get_all_positions()
        for p in positions:
            sym = p["symbol"]
            if sym not in self.symbols:
                continue
            if sym in self.active_trades:
                continue
            qty = abs(p["qty"])
            direction = p["side"]
            self.active_trades[sym] = ActiveTrade(
                symbol=sym,
                direction=direction,
                shares=qty,
                entry_price=p.get("avg_entry", 0.0),
                entry_time=now_et(),
                order_id="reconciled",
            )
            log.warning("RECONCILED existing IB position: %s %d %s @ $%.2f",
                        direction.upper(), qty, sym, p.get("avg_entry", 0))

        log.info("Startup complete. Equity: $%.0f", equity)

    def shutdown(self):
        log.info("Shutting down...")
        self.running = False

        if not self.dry_run:
            for symbol, trade in list(self.active_trades.items()):
                log.info("Closing position: %s", symbol)
                close_position(symbol)

            remaining = get_all_positions()
            if remaining:
                log.warning("Closing %d remaining positions", len(remaining))
                close_all_positions()

            cancel_all_orders()

        # End-of-day summary
        self._print_eod_summary()
        log.info("Shutdown complete.")

    def _print_eod_summary(self):
        if self.risk_manager is None:
            return
        status = self.risk_manager.status()
        elapsed = (now_et() - self._session_start).total_seconds() / 60 if self._session_start else 0

        # Override status numbers with values read directly from this
        # strategy's trade journal — defensive against any stale state.
        trades_from_journal = 0
        pnl_from_journal = 0.0
        try:
            import csv as _csv
            if self._trade_journal_path and os.path.exists(self._trade_journal_path):
                with open(self._trade_journal_path) as _f:
                    reader = _csv.DictReader(_f)
                    for row in reader:
                        if row.get("action") == "exit":
                            trades_from_journal += 1
                            try:
                                pnl_from_journal += float(row.get("pnl") or 0)
                            except ValueError:
                                pass
            status = dict(status)
            status["trades_today"] = trades_from_journal
            status["daily_pnl"] = pnl_from_journal
        except Exception as _e:
            log.debug("Could not read trade journal for EOD: %s", _e)

        log.info("=" * 70)
        log.info("END OF DAY SUMMARY")
        log.info("  Date:             %s", now_et().strftime("%Y-%m-%d"))
        log.info("  Session duration: %.0f minutes", elapsed)
        log.info("  Iterations:       %d", self._iteration_count)
        log.info("  Trades today:     %d", status["trades_today"])
        log.info("  Daily P&L:        $%.2f", status["daily_pnl"])
        log.info("  Final equity:     $%.2f", self.risk_manager.state.current_equity)
        log.info("  Consec losses:    %d", status["consecutive_losses"])
        log.info("  Halted:           %s %s", status["halted"],
                 f"({status['halt_reason']})" if status["halt_reason"] else "")
        log.info("  Trade journal:    %s", self._trade_journal_path)
        log.info("  Equity log:       %s", self._equity_log_path)
        log.info("  Signal log:       %s", self._signal_log_path)
        log.info("=" * 70)

        # Also write summary as JSON for programmatic access
        summary = {
            "date": now_et().strftime("%Y-%m-%d"),
            "session_minutes": round(elapsed, 1),
            "iterations": self._iteration_count,
            "trades": status["trades_today"],
            "daily_pnl": round(status["daily_pnl"], 2),
            "final_equity": round(self.risk_manager.state.current_equity, 2),
            "halted": status["halted"],
            "halt_reason": status["halt_reason"],
        }
        summary_path = os.path.join(self._log_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Send EOD alerts (email + Discord summary channel)
        try:
            strat_name = "ORB" if not hasattr(self, "primary_symbol") else "Pairs"
            if hasattr(self, "_log_base_dir") and "opendrive" in str(self._log_base_dir):
                strat_name = "OpenDrive"
            alert_eod_summary({strat_name: summary})
            discord_eod({strat_name: summary})
        except Exception as e:
            log.debug("EOD alert failed: %s", e)

    # ── Main Loop ──

    def run(self):
        self.startup()

        try:
            while self.running:
                self._wait_for_market_open()
                if not self.running:
                    break
                self._wait_for_trading_start()
                if not self.running:
                    break
                self._trading_loop()

                # After session ends, sleep until next day
                if self.running:
                    self._print_eod_summary()
                    now = now_et()
                    tomorrow = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
                    sleep_sec = min((tomorrow - now).total_seconds(), 3600)
                    log.info("Session complete. Sleeping %.0f min until next check.", sleep_sec / 60)
                    time.sleep(sleep_sec)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        except Exception:
            log.exception("Unexpected error in trading loop")
        finally:
            self.shutdown()

    def _wait_for_market_open(self):
        """Wait until the stock market is open (09:30-16:00 ET weekdays)."""
        while self.running and not is_market_open():
            now = now_et()
            weekday = now.weekday()
            if weekday >= 5:
                days_to_mon = 7 - weekday
                log.info("Weekend. Sleeping until Monday...")
                time.sleep(min(days_to_mon * 3600, 3600))
                continue
            # Before market open
            open_min = 9 * 60 + 30
            now_min = now.hour * 60 + now.minute
            if now_min < open_min:
                wait = (open_min - now_min) * 60
                log.info("Market closed. Opens 09:30 ET (in %.0f min). Sleeping...", wait / 60)
                time.sleep(min(wait, 300))
            else:
                # After close, wait until tomorrow
                log.info("Market closed for today. Sleeping...")
                time.sleep(300)

        if self.running:
            log.info("Market is OPEN")
            # Re-initialize log dir for new trading day
            self._log_dir = _ensure_log_dir()
            self._init_trade_journal()
            self._init_equity_log()
            self._init_signal_log()
            # Reset risk manager for new day
            try:
                equity = get_account_equity()
                self.risk_manager = RiskManager(equity)
                log.info("Day reset: equity=$%.0f", equity)
            except Exception as e:
                log.error("Error resetting for new day: %s", e)

    def _wait_for_trading_start(self):
        """Wait until TRADE_START (09:45 ET) — the opening range needs to form first."""
        start_min = time_str_to_minutes(TRADE_START)
        while self.running and current_minute() < start_min:
            remaining = start_min - current_minute()
            log.info("Market open but waiting for trade start (%s ET, %d min remaining)",
                     TRADE_START, remaining)
            time.sleep(min(remaining * 60, 30))

    def _trading_loop(self):
        trade_end_min = time_str_to_minutes(TRADE_END)
        force_close_min = time_str_to_minutes(FORCE_CLOSE)

        while self.running:
            self._iteration_count += 1
            now_min = current_minute()

            # Force close time
            if now_min >= force_close_min:
                log.info("Force close time reached (%s ET)", FORCE_CLOSE)
                if not self.dry_run:
                    for symbol in list(self.active_trades.keys()):
                        self._close_trade(symbol, "eod_force_close")
                break

            # Update risk state
            try:
                equity = get_account_equity()
                self.risk_manager.update_equity(equity)
                self.risk_manager.update_positions(len(self.active_trades))
            except Exception as e:
                log.error("Error updating risk state: %s", e)
                self.risk_manager.record_api_error()

            # Sanity check: verify internal state matches IB every iteration
            if not self.dry_run:
                try:
                    ib_positions = get_all_positions()
                    ib_syms = {p["symbol"] for p in ib_positions if p["qty"] != 0}
                    internal_syms = set(self.active_trades.keys())

                    for sym in ib_syms - internal_syms:
                        if sym in self.symbols:
                            pos = [p for p in ib_positions if p["symbol"] == sym][0]
                            self.active_trades[sym] = ActiveTrade(
                                symbol=sym,
                                direction=pos["side"],
                                shares=abs(pos["qty"]),
                                entry_price=pos.get("avg_entry", 0.0),
                                entry_time=now_et(),
                                order_id="adopted",
                            )
                            log.warning("SANITY: adopted unknown IB position %s %d %s",
                                        pos["side"].upper(), abs(pos["qty"]), sym)

                    for sym in internal_syms - ib_syms:
                        log.warning("SANITY: stale %s not in IB, removing", sym)
                        del self.active_trades[sym]
                except Exception as e:
                    log.error("Sanity check error: %s", e)

            # Check if halted
            if self.risk_manager.state.halted:
                log.warning("Risk manager halted: %s", self.risk_manager.state.halt_reason)
                if not self.dry_run:
                    for symbol in list(self.active_trades.keys()):
                        self._close_trade(symbol, "risk_halt")
                break

            # CRITICAL: Check strategy exit signals for active positions.
            # The ORB strategy encodes exits (target hit, stop hit, stale exit)
            # in the signal going from +/-1 to 0. We must detect this.
            self._check_strategy_exits()

            # Check hard stop/TP backstop from risk manager
            self._check_positions()

            # Generate new signals (only before trade end time)
            if now_min < trade_end_min:
                self._scan_for_signals()

            # Log equity snapshot
            self._log_equity()

            # Log status
            status = self.risk_manager.status()
            positions_str = ", ".join(
                f"{t.direction[0].upper()} {t.symbol}" for t in self.active_trades.values()
            ) or "flat"
            log.info("[iter %d] pnl=$%.0f | pos=%s | trades=%d | %s",
                     self._iteration_count, status["daily_pnl"],
                     positions_str, status["trades_today"],
                     now_et().strftime("%H:%M ET"))

            time.sleep(60)

    # ── Strategy Exit Detection (CRITICAL) ──

    def _check_strategy_exits(self):
        """Re-run strategy signals for active positions. If signal -> 0, exit.

        This is the primary exit mechanism. The ORB strategy's target, stop,
        and stale exit are all encoded in the signal going from +/-1 to 0.
        Without this, the strategy's edge would be destroyed.
        """
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades[symbol]
            try:
                df = fetch_live_bars(symbol, lookback_minutes=200)
                if len(df) < 30:
                    log.warning("%s: insufficient bars (%d) for exit check", symbol, len(df))
                    continue

                df = prepare_features(df)
                strat = self._get_strategy(symbol)
                df = strat.generate_signals(df)

                last_signal = int(df["signal"].iloc[-1])
                price = float(df["close"].iloc[-1])

                # Extract opening range for logging
                or_high = float(df["or_high"].iloc[-1]) if "or_high" in df.columns and pd.notna(df["or_high"].iloc[-1]) else 0
                or_low = float(df["or_low"].iloc[-1]) if "or_low" in df.columns and pd.notna(df["or_low"].iloc[-1]) else 0

                expected_signal = 1 if trade.direction == "long" else -1

                if last_signal == 0:
                    # Strategy says exit
                    log.info("STRATEGY EXIT: %s %s -> signal=0 @ $%.2f (entry $%.2f)",
                             trade.direction.upper(), symbol, price, trade.entry_price)
                    self._log_signal(symbol, last_signal, price, or_high, or_low,
                                     True, "strategy_exit", "signal went to 0")
                    self._close_trade(symbol, "strategy_exit")

                elif last_signal == -expected_signal:
                    # Signal flipped direction — close and let _scan_for_signals re-enter
                    log.info("STRATEGY FLIP: %s %s -> signal=%d @ $%.2f (entry $%.2f)",
                             trade.direction.upper(), symbol, last_signal, price,
                             trade.entry_price)
                    self._log_signal(symbol, last_signal, price, or_high, or_low,
                                     True, "strategy_flip", "signal reversed")
                    self._close_trade(symbol, "strategy_flip")

                else:
                    # Still in position, strategy agrees
                    self._log_signal(symbol, last_signal, price, or_high, or_low,
                                     True, "hold", "strategy confirms position")

            except Exception as e:
                log.error("Error checking strategy exit for %s: %s", symbol, e)
                self.risk_manager.record_api_error()

    # ── Signal Scanning ──

    def _scan_for_signals(self):
        # Global position check: all traders share one Alpaca account
        try:
            all_positions = get_all_positions()
            from trading.config import MAX_CONCURRENT_POSITIONS
            if len(all_positions) >= MAX_CONCURRENT_POSITIONS:
                log.warning("GLOBAL LIMIT: %d positions across all instances (max %d)",
                           len(all_positions), MAX_CONCURRENT_POSITIONS)
                return
        except Exception as e:
            log.error("Error checking global positions: %s", e)

        for symbol in self.symbols:
            if symbol in self.active_trades:
                continue  # Exits handled by _check_strategy_exits

            # Churn circuit breaker: >5 trades in last hour = halt this symbol
            now = now_et()
            recent = self._trade_timestamps.get(symbol, [])
            one_hour_ago = now - timedelta(hours=1)
            recent = [t for t in recent if t > one_hour_ago]
            self._trade_timestamps[symbol] = recent
            if len(recent) >= 5:
                log.warning("CHURN HALT %s: %d trades in last hour, skipping", symbol, len(recent))
                continue

            # Double-check IB for existing position (prevents duplicates on restart)
            try:
                ib_pos = get_position(symbol)
                if ib_pos is not None and ib_pos["qty"] != 0:
                    log.debug("SKIP %s: IB already has position (%d)", symbol, ib_pos["qty"])
                    continue
            except Exception:
                pass

            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                log.debug("BLOCKED %s: %s", symbol, reason)
                self._log_signal(symbol, 0, 0, 0, 0, False, "blocked", reason)
                continue

            try:
                df = fetch_live_bars(symbol, lookback_minutes=200)
                if len(df) < 30:
                    log.debug("SKIP %s: only %d bars (need 30+)", symbol, len(df))
                    self._log_signal(symbol, 0, 0, 0, 0, False, "insufficient_data",
                                     f"only {len(df)} bars")
                    continue

                df = prepare_features(df)
                strat = self._get_strategy(symbol)
                df = strat.generate_signals(df)

                last_signal = int(df["signal"].iloc[-1])
                price = float(df["close"].iloc[-1])

                or_high = float(df["or_high"].iloc[-1]) if "or_high" in df.columns and pd.notna(df["or_high"].iloc[-1]) else 0
                or_low = float(df["or_low"].iloc[-1]) if "or_low" in df.columns and pd.notna(df["or_low"].iloc[-1]) else 0

                if last_signal == 0:
                    self._log_signal(symbol, 0, price, or_high, or_low,
                                     False, "no_signal", "")
                    continue

                shares = self.risk_manager.calculate_shares(price)
                if shares == 0:
                    log.info("SKIP %s: shares=0 (price=$%.2f)", symbol, price)
                    self._log_signal(symbol, last_signal, price, or_high, or_low,
                                     False, "zero_shares", f"price={price:.2f}")
                    continue

                direction = "long" if last_signal > 0 else "short"
                side = "buy" if direction == "long" else "sell"

                log.info("SIGNAL: %s %s %d shares @ ~$%.2f (OR: %.2f-%.2f)",
                         direction.upper(), symbol, shares, price, or_low, or_high)
                self._log_signal(symbol, last_signal, price, or_high, or_low,
                                 False, "entry_signal", f"{direction} {shares} shares")

                if self.dry_run:
                    log.info("[DRY RUN] Would %s %d %s @ ~$%.2f", side, shares, symbol, price)
                    continue

                order_id = submit_market_order(symbol, shares, side)
                if order_id is None:
                    log.warning("ORDER REJECTED: %s %d %s", side, shares, symbol)
                    self.risk_manager.record_order_rejection()
                    self._log_trade(symbol, direction, "rejected", shares, price,
                                    None, "order_rejected", None)
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
                    # Track trade time for churn detection
                    self._trade_timestamps.setdefault(symbol, []).append(now_et())
                    log.info("FILLED: %s %s %d @ $%.2f (order %s)",
                             direction.upper(), symbol,
                             fill["filled_qty"], fill["filled_avg_price"], order_id)
                    self._log_trade(symbol, direction, "entry", fill["filled_qty"],
                                    fill["filled_avg_price"], None, "filled", order_id)
                    strat_label = self._get_strategy(symbol).name
                    discord_trade(strat_label, symbol, direction, "entry",
                                  fill["filled_qty"], fill["filled_avg_price"])
                else:
                    fill_status = fill.get("status", "timeout") if fill else "timeout"
                    log.warning("FILL FAILED: %s %s -> %s", side, symbol, fill_status)
                    self.risk_manager.record_order_rejection()
                    self._log_trade(symbol, direction, "fill_failed", shares, price,
                                    None, fill_status, order_id)

            except Exception as e:
                log.error("Error scanning %s: %s", symbol, e, exc_info=True)
                self.risk_manager.record_api_error()

    # ── Position Checks (hard stop/TP backstop) ──

    def _check_positions(self):
        """Check hard stop/TP from risk manager. This is a BACKSTOP only.
        Primary exits come from _check_strategy_exits()."""
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades[symbol]

            try:
                pos = get_position(symbol)
                if pos is None:
                    log.warning("%s: position disappeared from broker", symbol)
                    self._log_trade(symbol, trade.direction, "exit",
                                    trade.shares, trade.entry_price, 0,
                                    "position_disappeared", trade.order_id)
                    del self.active_trades[symbol]
                    continue

                # Get current price from live bars (IB positions don't include market_value)
                try:
                    price_df = fetch_live_bars(symbol, lookback_minutes=5)
                    if len(price_df) > 0:
                        current_price = float(price_df["close"].iloc[-1])
                    else:
                        current_price = trade.entry_price
                except Exception:
                    current_price = trade.entry_price

                if self.risk_manager.check_stop_loss(trade.entry_price, current_price, trade.direction):
                    log.warning("HARD STOP: %s @ $%.2f (entry $%.2f, -%.1f%%)",
                               symbol, current_price, trade.entry_price,
                               STOP_LOSS_PCT * 100)
                    self._close_trade(symbol, "hard_stop")

                elif self.risk_manager.check_take_profit(trade.entry_price, current_price, trade.direction):
                    log.info("HARD TP: %s @ $%.2f (entry $%.2f, +%.1f%%)",
                            symbol, current_price, trade.entry_price,
                            TAKE_PROFIT_PCT * 100)
                    self._close_trade(symbol, "hard_tp")

            except Exception as e:
                log.error("Error checking position %s: %s", symbol, e)
                self.risk_manager.record_api_error()

    def _close_trade(self, symbol: str, reason: str):
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
                self._log_trade(symbol, trade.direction, "exit", trade.shares,
                                exit_price, pnl, reason, order_id)
                strat_label = self._get_strategy(symbol).name
                discord_trade(strat_label, symbol, trade.direction, "exit",
                              trade.shares, exit_price, pnl=pnl, reason=reason,
                              entry_price=trade.entry_price)
                del self.active_trades[symbol]
            else:
                log.error("Close FAILED for %s (reason=%s), keeping in dict", symbol, reason)
                self._log_trade(symbol, trade.direction, "exit_failed", trade.shares,
                                trade.entry_price, None, reason, order_id)
        else:
            # No IB position found — remove stale tracking
            log.warning("%s: no IB position to close, removing from tracking", symbol)
            del self.active_trades[symbol]
