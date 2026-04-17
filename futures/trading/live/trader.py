"""Live futures trading loop via Interactive Brokers.

Connects to IB TWS/Gateway, trades real MES and MNQ micro futures
contracts during regular trading hours (09:30-16:00 ET), and manages
positions with futures-specific risk controls.

STRATEGY TIMING:
  ORB Breakout:     09:45 - 15:30 entry, force flat 15:50
  VWAP Reversion:   10:00 - 15:00 entry, force flat 15:25

Both strategies share the same risk manager and account but have
separate signal generation. No concurrent positions in the same symbol.

CRITICAL: Strategy exits (target, stop, stale) are executed by re-running
signal generation each iteration and detecting signal -> 0 transitions.
The risk manager's hard stop/TP is a backstop only.
"""

import csv
import json
import logging
import os
import time
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass

import pytz
import pandas as pd

from trading.config import (
    FUTURES_SYMBOLS, TRADE_START, TRADE_END, FORCE_CLOSE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, LOG_DIR,
)
from trading.data.contracts import CONTRACTS, FuturesContract
from trading.execution.broker import (
    get_ib, get_account_equity, get_available_margin, get_position,
    get_all_positions, submit_order, submit_market_order, wait_for_fill,
    close_position, close_all_positions, cancel_all_orders, is_market_open,
    get_contract_month, disconnect, is_connected,
)
from trading.risk.manager import FuturesRiskManager
from trading.data.live_feed import fetch_live_bars
from trading.data.features import prepare_features
from trading.discord_alerts import discord_trade, discord_halt, discord_eod
from trading.alerts import alert_eod_summary

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)
    date_dir = os.path.join(LOG_DIR, now_et().strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)
    return date_dir


# VWAP Reversion timing (minutes since midnight ET)
VWAP_ENTRY_START = 600   # 10:00 AM
VWAP_ENTRY_END = 900     # 3:00 PM
VWAP_FORCE_CLOSE = 925   # 3:25 PM
VWAP_STOP_PCT = 0.003    # 0.3% notional stop for position sizing


@dataclass
class ActiveTrade:
    futures_symbol: str
    direction: str
    contracts: int
    entry_price: float  # Actual futures price from IB
    entry_time: datetime
    order_id: str
    contract_spec: FuturesContract
    strategy: str = "ORB"  # "ORB" or "VWAP"


def now_et() -> datetime:
    return datetime.now(ET)


def time_str_to_minutes(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def current_minute() -> int:
    now = now_et()
    return now.hour * 60 + now.minute


class FuturesLiveTrader:
    """Live trading orchestrator for micro futures via IB."""

    def __init__(self, strategies: dict = None, vwap_strategies: dict = None,
                 symbols: list[str] = None, dry_run: bool = False):
        self.strategies = strategies or {}
        self.vwap_strategies = vwap_strategies or {}
        self.symbols = symbols or FUTURES_SYMBOLS
        self.dry_run = dry_run
        self.active_trades: dict[str, ActiveTrade] = {}
        self.active_vwap_trades: dict[str, ActiveTrade] = {}
        self.risk_manager: FuturesRiskManager | None = None
        self.running = True
        self._trade_timestamps: dict[str, list] = {}  # churn detection

        self._log_dir = None
        self._trade_journal_path = None
        self._equity_log_path = None
        self._signal_log_path = None
        self._iteration_count = 0
        self._session_start = None

    def _get_strategy(self, symbol: str):
        return self.strategies.get(symbol)

    def _get_vwap_strategy(self, symbol: str):
        return self.vwap_strategies.get(symbol)

    def _reconcile_ib_positions(self):
        """Adopt any existing IB positions on startup to prevent duplicates.

        If the process restarts mid-day, IB may still hold positions from the
        previous run. We claim them as VWAP trades (the most likely source
        during cash hours) so the signal loop won't stack new orders on top.
        """
        try:
            positions = get_all_positions()
            for pos in positions:
                sym = pos["symbol"]
                if sym not in self.symbols:
                    continue
                if sym in self.active_trades or sym in self.active_vwap_trades:
                    continue  # Already tracked

                qty = abs(pos["qty"])
                direction = pos["side"]
                contract = self._get_contract(sym)

                # IB's avgCost for futures is (price * multiplier).
                # Convert back to actual price for entry_price so P&L math works.
                raw_avg = pos.get("avg_entry", 0.0)
                multiplier = getattr(contract, "multiplier", 1.0) or 1.0
                entry_price = raw_avg / multiplier if multiplier > 1 else raw_avg

                trade = ActiveTrade(
                    futures_symbol=sym,
                    direction=direction,
                    contracts=qty,
                    entry_price=entry_price,
                    entry_time=now_et(),
                    order_id="reconciled",
                    contract_spec=contract,
                    strategy="VWAP",
                )
                # Assign to VWAP since it's the active mid-day strategy
                self.active_vwap_trades[sym] = trade
                log.warning("RECONCILED existing IB position: %s %d %s @ $%.2f "
                            "(adopted as VWAP trade)",
                            direction.upper(), qty, sym, trade.entry_price)
        except Exception as e:
            log.error("Error reconciling IB positions: %s", e)

    def _get_contract(self, symbol: str) -> FuturesContract:
        return CONTRACTS[symbol]

    # ---- Trade Journal ----

    def _init_trade_journal(self):
        self._trade_journal_path = os.path.join(self._log_dir, "trades.csv")
        if not os.path.exists(self._trade_journal_path):
            with open(self._trade_journal_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "strategy", "futures_symbol", "contract_month",
                    "direction", "action", "contracts", "price", "pnl",
                    "reason", "order_id",
                ])

    def _log_trade(self, futures_sym, direction, action, contracts,
                   price, pnl, reason, order_id, strategy="ORB"):
        if self._trade_journal_path is None:
            return
        contract_month = ""
        try:
            contract_month = get_contract_month(futures_sym)
        except Exception:
            pass
        with open(self._trade_journal_path, "a", newline="") as f:
            csv.writer(f).writerow([
                now_et().isoformat(), strategy, futures_sym, contract_month,
                direction, action, contracts, f"{price:.2f}",
                f"{pnl:.2f}" if pnl is not None else "", reason, order_id or "",
            ])

    # ---- Equity Log ----

    def _init_equity_log(self):
        self._equity_log_path = os.path.join(self._log_dir, "equity.csv")
        if not os.path.exists(self._equity_log_path):
            with open(self._equity_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "equity", "margin_available", "daily_pnl",
                    "open_positions", "trades_today", "circuit_breaker",
                ])

    def _log_equity(self):
        if self._equity_log_path is None or self.risk_manager is None:
            return
        status = self.risk_manager.status()
        margin = 0
        try:
            margin = get_available_margin()
        except Exception:
            pass
        with open(self._equity_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                now_et().isoformat(),
                f"{self.risk_manager.state.current_equity:.2f}",
                f"{margin:.2f}",
                f"{status['daily_pnl']:.2f}",
                status["open_positions"],
                status["trades_today"],
                status["circuit_breaker"],
            ])

    # ---- Signal Log ----

    def _init_signal_log(self):
        self._signal_log_path = os.path.join(self._log_dir, "signals.csv")
        if not os.path.exists(self._signal_log_path):
            with open(self._signal_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "futures_symbol", "signal",
                    "price", "or_high", "or_low", "range_pct",
                    "in_position", "action_taken", "reason",
                ])

    def _log_signal(self, futures_sym, signal, price, or_high, or_low,
                    in_position, action_taken, reason=""):
        if self._signal_log_path is None:
            return
        range_pct = (or_high - or_low) / or_low * 100 if or_low > 0 and or_high > 0 else 0
        with open(self._signal_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                now_et().isoformat(), futures_sym, signal,
                f"{price:.2f}", f"{or_high:.2f}", f"{or_low:.2f}",
                f"{range_pct:.4f}", in_position, action_taken, reason,
            ])

    # ---- Startup / Shutdown ----

    def startup(self):
        self._session_start = now_et()
        self._log_dir = _ensure_log_dir()
        self._init_trade_journal()
        self._init_equity_log()
        self._init_signal_log()

        log.info("=" * 70)
        log.info("FUTURES LIVE TRADER STARTING (Interactive Brokers)")
        log.info("  Session: %s", self._session_start.strftime("%Y-%m-%d %H:%M:%S ET"))
        log.info("  Log dir: %s", self._log_dir)
        log.info("  Dry run: %s", self.dry_run)
        log.info("  Symbols: %s", self.symbols)
        for sym in self.symbols:
            contract = self._get_contract(sym)
            strat = self._get_strategy(sym)
            vwap_strat = self._get_vwap_strategy(sym)
            try:
                month = get_contract_month(sym)
            except Exception:
                month = "unknown"
            log.info("  %s: margin=$%s, multiplier=$%s, contract=%s",
                     sym, contract.margin_intraday, contract.multiplier, month)
            if strat:
                log.info("    ORB: %s", strat.get_params())
            if vwap_strat:
                log.info("    VWAP: %s", vwap_strat.get_params())
        log.info("=" * 70)

        try:
            equity = get_account_equity()
            margin = get_available_margin()
        except Exception as e:
            log.error("Cannot connect to IB: %s", e)
            sys.exit(1)

        self.risk_manager = FuturesRiskManager(equity)
        log.info("Startup complete. Equity: $%.0f, Available margin: $%.0f",
                 equity, margin)

        # ---- Reconcile with existing IB positions ----
        # If the process restarts mid-day, adopt any open positions so we
        # don't pile on duplicate orders.
        self._reconcile_ib_positions()

    def shutdown(self):
        log.info("Shutting down futures trader...")
        self.running = False

        if not self.dry_run:
            for symbol in list(self.active_trades.keys()):
                self._close_trade(symbol, "eod_shutdown")
            for symbol in list(self.active_vwap_trades.keys()):
                self._close_vwap_trade(symbol, "eod_shutdown")

            remaining = get_all_positions()
            if remaining:
                log.warning("Closing %d remaining positions", len(remaining))
                close_all_positions()
            cancel_all_orders()

        self._print_eod_summary()
        disconnect()
        log.info("Shutdown complete.")

    def _print_eod_summary(self):
        if self.risk_manager is None:
            return
        status = self.risk_manager.status()
        elapsed = (now_et() - self._session_start).total_seconds() / 60 if self._session_start else 0

        # Override status numbers with values from trade journal (defensive).
        try:
            import csv as _csv
            trades_ct = 0
            pnl_sum = 0.0
            if self._trade_journal_path and os.path.exists(self._trade_journal_path):
                with open(self._trade_journal_path) as _f:
                    for row in _csv.DictReader(_f):
                        if row.get("action") == "exit":
                            trades_ct += 1
                            try:
                                pnl_sum += float(row.get("pnl") or 0)
                            except ValueError:
                                pass
            status = dict(status)
            status["trades_today"] = trades_ct
            status["daily_pnl"] = pnl_sum
        except Exception as _e:
            log.debug("Could not read journal for EOD: %s", _e)

        log.info("=" * 70)
        log.info("FUTURES END OF DAY SUMMARY")
        log.info("  Date:             %s", now_et().strftime("%Y-%m-%d"))
        log.info("  Session duration: %.0f minutes", elapsed)
        log.info("  Trades today:     %d", status["trades_today"])
        log.info("  Daily P&L:        $%.2f", status["daily_pnl"])
        log.info("  Final equity:     $%.2f", self.risk_manager.state.current_equity)
        log.info("  Circuit breaker:  %s", status["circuit_breaker"])
        log.info("  Halted:           %s %s", status["halted"],
                 f"({status['halt_reason']})" if status["halt_reason"] else "")
        log.info("=" * 70)

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

        try:
            discord_eod({"Futures_ORB+VWAP": summary})
            alert_eod_summary({"Futures_ORB+VWAP": summary})
        except Exception as e:
            log.debug("EOD alert failed: %s", e)

    # ---- Main Loop ----

    def run(self):
        self.startup()
        try:
            while self.running:
                self._wait_for_cash_session()
                if not self.running:
                    break
                self._wait_for_trading_start()
                if not self.running:
                    break
                self._trading_loop()

                # After cash session ends, wait for next day
                if self.running:
                    self._print_eod_summary()
                    # Reset for next day
                    self._wait_until_next_session()
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        except Exception:
            log.exception("Unexpected error in trading loop")
        finally:
            self.shutdown()

    def _wait_for_cash_session(self):
        """Wait until cash session opens (9:30 ET)."""
        cash_open_min = time_str_to_minutes("09:30")

        while self.running:
            now = now_et()
            now_min = current_minute()
            weekday = now.weekday()

            # Weekend: sleep longer
            if weekday in (5, 6):
                if weekday == 5:
                    hours_to_monday = 48 - now.hour
                else:
                    hours_to_monday = 24 - now.hour + (18 - 0)  # Until Sunday 6pm
                log.info("Weekend. Sleeping until futures open...")
                time.sleep(min(hours_to_monday * 3600, 3600))
                continue

            # Before cash open
            if now_min < cash_open_min:
                wait_min = cash_open_min - now_min
                log.info("Waiting for cash session open (09:30 ET, %d min remaining)", wait_min)
                time.sleep(min(wait_min * 60, 300))
                continue

            # Cash session already passed today
            if now_min >= time_str_to_minutes("16:00"):
                log.info("Cash session closed. Waiting for tomorrow.")
                time.sleep(300)
                continue

            # We're in the cash session
            log.info("Cash session is OPEN")
            self._log_dir = _ensure_log_dir()
            self._init_trade_journal()
            self._init_equity_log()
            self._init_signal_log()

            # Reconnect to IB if needed and refresh equity
            try:
                if not is_connected():
                    get_ib()
                equity = get_account_equity()
                self.risk_manager = FuturesRiskManager(equity)
                log.info("Day reset: equity=$%.0f", equity)
            except Exception as e:
                log.error("Error resetting for new day: %s", e)
                time.sleep(60)
                continue

            return

    def _wait_until_next_session(self):
        """Sleep until the next cash session."""
        now = now_et()
        # Sleep until midnight + buffer, then _wait_for_cash_session handles the rest
        tomorrow = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
        sleep_sec = (tomorrow - now).total_seconds()
        sleep_sec = min(sleep_sec, 3600)  # Max 1 hour sleep chunks
        log.info("Session complete. Sleeping %.0f min until next check.", sleep_sec / 60)
        time.sleep(sleep_sec)

    def _wait_for_trading_start(self):
        """Wait until TRADE_START (09:45 ET, after opening range forms)."""
        start_min = time_str_to_minutes(TRADE_START)
        while self.running and current_minute() < start_min:
            remaining = start_min - current_minute()
            log.info("Waiting for trade start (%s ET, %d min remaining)",
                     TRADE_START, remaining)
            time.sleep(min(remaining * 60, 30))

    def _trading_loop(self):
        trade_end_min = time_str_to_minutes(TRADE_END)
        force_close_min = time_str_to_minutes(FORCE_CLOSE)

        while self.running:
            self._iteration_count += 1
            now_min = current_minute()

            # VWAP force close at 3:25 PM (before ORB force close)
            if now_min >= VWAP_FORCE_CLOSE and self.active_vwap_trades:
                log.info("VWAP force close time reached (15:25 ET)")
                if not self.dry_run:
                    for symbol in list(self.active_vwap_trades.keys()):
                        self._close_vwap_trade(symbol, "vwap_eod_force_close")

            # ORB force close at 3:50 PM
            if now_min >= force_close_min:
                log.info("Force close time reached (%s ET)", FORCE_CLOSE)
                if not self.dry_run:
                    for symbol in list(self.active_trades.keys()):
                        self._close_trade(symbol, "eod_force_close")
                break

            # Get real equity from IB
            try:
                if not is_connected():
                    log.warning("IB disconnected, reconnecting...")
                    get_ib()
                equity = get_account_equity()
                self.risk_manager.update_equity(equity)
                self.risk_manager.update_positions(
                    len(self.active_trades) + len(self.active_vwap_trades)
                )
            except Exception as e:
                log.error("Error updating risk state: %s", e)
                self.risk_manager.record_api_error()

            # Sanity check: verify internal state matches IB every iteration
            if not self.dry_run:
                try:
                    ib_positions = get_all_positions()
                    ib_syms = {p["symbol"] for p in ib_positions if p["qty"] != 0}
                    internal_syms = set(self.active_trades.keys()) | set(self.active_vwap_trades.keys())

                    # IB has position we don't know about -> adopt it
                    for sym in ib_syms - internal_syms:
                        if sym in self.symbols:
                            pos = [p for p in ib_positions if p["symbol"] == sym][0]
                            contract = self._get_contract(sym)
                            raw_avg = pos.get("avg_entry", 0.0)
                            multiplier = getattr(contract, "multiplier", 1.0) or 1.0
                            entry_price = raw_avg / multiplier if multiplier > 1 else raw_avg
                            trade = ActiveTrade(
                                futures_symbol=sym,
                                direction=pos["side"],
                                contracts=abs(pos["qty"]),
                                entry_price=entry_price,
                                entry_time=now_et(),
                                order_id="adopted",
                                contract_spec=contract,
                                strategy="VWAP",
                            )
                            self.active_vwap_trades[sym] = trade
                            log.warning("SANITY: adopted unknown IB position %s %d %s @ $%.2f",
                                        pos["side"].upper(), abs(pos["qty"]), sym, entry_price)

                    # We think we have position but IB doesn't -> remove stale
                    for sym in internal_syms - ib_syms:
                        if sym in self.active_trades:
                            log.warning("SANITY: stale ORB %s not in IB, removing", sym)
                            del self.active_trades[sym]
                        if sym in self.active_vwap_trades:
                            log.warning("SANITY: stale VWAP %s not in IB, removing", sym)
                            del self.active_vwap_trades[sym]
                except Exception as e:
                    log.error("Sanity check error: %s", e)

            if self.risk_manager.state.halted:
                log.warning("Risk manager halted: %s", self.risk_manager.state.halt_reason)
                if not self.dry_run:
                    for symbol in list(self.active_trades.keys()):
                        self._close_trade(symbol, "risk_halt")
                    for symbol in list(self.active_vwap_trades.keys()):
                        self._close_vwap_trade(symbol, "risk_halt")
                break

            # Check exits for both strategies
            self._check_strategy_exits()
            self._check_vwap_exits()
            self._check_positions()

            # ORB signal scanning (before TRADE_END)
            if now_min < trade_end_min:
                self._scan_for_signals()

            # VWAP signal scanning (10:00 AM - 3:00 PM)
            if VWAP_ENTRY_START <= now_min < VWAP_ENTRY_END:
                self._scan_for_vwap_signals()

            self._log_equity()

            # Status line shows both ORB and VWAP positions
            status = self.risk_manager.status()
            orb_pos = [f"ORB:{t.direction[0].upper()} {t.futures_symbol}"
                       for t in self.active_trades.values()]
            vwap_pos = [f"VWAP:{t.direction[0].upper()} {t.futures_symbol}"
                        for t in self.active_vwap_trades.values()]
            positions_str = ", ".join(orb_pos + vwap_pos) or "flat"
            log.info("[iter %d] pnl=$%.0f | pos=%s | trades=%d | %s",
                     self._iteration_count, status["daily_pnl"],
                     positions_str, status["trades_today"],
                     now_et().strftime("%H:%M ET"))

            time.sleep(60)

    # ---- Strategy Exit Detection ----

    def _check_strategy_exits(self):
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades[symbol]
            try:
                df = fetch_live_bars(symbol, lookback_minutes=200)
                if len(df) < 30:
                    continue

                df = prepare_features(df)
                strat = self._get_strategy(symbol)
                df = strat.generate_signals(df)

                last_signal = int(df["signal"].iloc[-1])
                price = float(df["close"].iloc[-1])

                or_high = float(df["or_high"].iloc[-1]) if pd.notna(df.get("or_high", pd.Series([None])).iloc[-1]) else 0
                or_low = float(df["or_low"].iloc[-1]) if pd.notna(df.get("or_low", pd.Series([None])).iloc[-1]) else 0

                if last_signal == 0:
                    log.info("STRATEGY EXIT: %s %s -> signal=0 @ $%.2f",
                             trade.direction.upper(), symbol, price)
                    self._log_signal(symbol, last_signal, price,
                                     or_high, or_low, True, "strategy_exit", "signal went to 0")
                    self._close_trade(symbol, "strategy_exit")
                elif last_signal == -( 1 if trade.direction == "long" else -1):
                    log.info("STRATEGY FLIP: %s %s -> signal=%d",
                             trade.direction.upper(), symbol, last_signal)
                    self._log_signal(symbol, last_signal, price,
                                     or_high, or_low, True, "strategy_flip", "signal reversed")
                    self._close_trade(symbol, "strategy_flip")
                else:
                    self._log_signal(symbol, last_signal, price,
                                     or_high, or_low, True, "hold", "strategy confirms")

            except Exception as e:
                log.error("Error checking strategy exit for %s: %s", symbol, e)
                self.risk_manager.record_api_error()

    # ---- Signal Scanning ----

    def _scan_for_signals(self):
        for symbol in self.symbols:
            if symbol in self.active_trades or symbol in self.active_vwap_trades:
                continue

            # Churn circuit breaker: >5 trades in last hour = halt
            now = now_et()
            recent = self._trade_timestamps.get(symbol, [])
            one_hour_ago = now - timedelta(hours=1)
            recent = [t for t in recent if t > one_hour_ago]
            self._trade_timestamps[symbol] = recent
            if len(recent) >= 5:
                log.warning("CHURN HALT %s: %d trades in last hour", symbol, len(recent))
                continue

            # Double-check IB for existing position (prevents duplicates on restart)
            try:
                ib_pos = get_position(symbol)
                if ib_pos is not None and ib_pos["qty"] != 0:
                    log.debug("SKIP ORB %s: IB already has position (%d)", symbol, ib_pos["qty"])
                    continue
            except Exception:
                pass

            strat = self._get_strategy(symbol)
            if strat is None:
                continue  # No ORB strategy for this symbol (e.g. MGC)

            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                continue

            contract = self._get_contract(symbol)

            try:
                df = fetch_live_bars(symbol, lookback_minutes=200)
                if len(df) < 30:
                    continue

                df = prepare_features(df)
                df = strat.generate_signals(df)

                last_signal = int(df["signal"].iloc[-1])
                price = float(df["close"].iloc[-1])

                or_high = float(df["or_high"].iloc[-1]) if "or_high" in df.columns and pd.notna(df["or_high"].iloc[-1]) else 0
                or_low = float(df["or_low"].iloc[-1]) if "or_low" in df.columns and pd.notna(df["or_low"].iloc[-1]) else 0

                if last_signal == 0:
                    self._log_signal(symbol, 0, price, or_high, or_low,
                                     False, "no_signal", "")
                    continue

                # Calculate stop for position sizing
                if last_signal > 0:
                    stop_price = or_low
                else:
                    stop_price = or_high

                contracts = self.risk_manager.calculate_contracts(
                    price, stop_price, contract, symbol
                )
                if contracts == 0:
                    continue

                # Margin check using real IB margin data
                ok, margin_msg = self.risk_manager.check_margin(contracts, contract)
                if not ok:
                    log.warning("MARGIN BLOCK %s: %s", symbol, margin_msg)
                    continue

                direction = "long" if last_signal > 0 else "short"
                side = "buy" if direction == "long" else "sell"

                log.info("SIGNAL: %s %s %d contracts @ ~$%.2f",
                         direction.upper(), symbol, contracts, price)
                self._log_signal(symbol, last_signal, price, or_high, or_low,
                                 False, "entry_signal", f"{direction} {contracts} contracts")

                if self.dry_run:
                    log.info("[DRY RUN] Would %s %d %s", side, contracts, symbol)
                    continue

                order_id, fill_type = submit_order(symbol, contracts, side)
                if order_id is None:
                    self.risk_manager.record_order_rejection()
                    continue

                fill = wait_for_fill(order_id)
                if fill and fill.get("status") == "filled":
                    self.active_trades[symbol] = ActiveTrade(
                        futures_symbol=symbol,
                        direction=direction,
                        contracts=fill["filled_qty"],
                        entry_price=fill["filled_avg_price"],
                        entry_time=now_et(),
                        order_id=order_id,
                        contract_spec=contract,
                    )
                    log.info("FILLED: %s %s %d contracts @ $%.2f (fill_type=%s)",
                             direction.upper(), symbol, fill["filled_qty"],
                             fill["filled_avg_price"], fill_type)
                    self._log_trade(symbol, direction, "entry",
                                    fill["filled_qty"], fill["filled_avg_price"],
                                    None, "filled", order_id)
                    discord_trade("Futures_ORB", symbol, direction, "entry",
                                  fill["filled_qty"], fill["filled_avg_price"])

            except Exception as e:
                log.error("Error scanning %s: %s", symbol, e, exc_info=True)
                self.risk_manager.record_api_error()

    # ---- Position Checks (backstop) ----

    def _check_positions(self):
        # ORB positions
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades[symbol]
            try:
                pos = get_position(symbol)
                if pos is None:
                    log.warning("%s: ORB position disappeared from IB", symbol)
                    del self.active_trades[symbol]
                    continue

                try:
                    df = fetch_live_bars(symbol, lookback_minutes=5)
                    if len(df) > 0:
                        current_price = float(df["close"].iloc[-1])
                    else:
                        current_price = trade.entry_price
                except Exception:
                    current_price = trade.entry_price

                if self.risk_manager.check_stop_loss(trade.entry_price, current_price, trade.direction):
                    log.warning("ORB HARD STOP: %s @ $%.2f", symbol, current_price)
                    self._close_trade(symbol, "hard_stop")
                elif self.risk_manager.check_take_profit(trade.entry_price, current_price, trade.direction):
                    self._close_trade(symbol, "hard_tp")

            except Exception as e:
                log.error("Error checking ORB position %s: %s", symbol, e)
                self.risk_manager.record_api_error()

        # VWAP positions
        for symbol in list(self.active_vwap_trades.keys()):
            trade = self.active_vwap_trades[symbol]
            try:
                pos = get_position(symbol)
                if pos is None:
                    log.warning("%s: VWAP position disappeared from IB", symbol)
                    del self.active_vwap_trades[symbol]
                    continue

                try:
                    df = fetch_live_bars(symbol, lookback_minutes=5)
                    if len(df) > 0:
                        current_price = float(df["close"].iloc[-1])
                    else:
                        current_price = trade.entry_price
                except Exception:
                    current_price = trade.entry_price

                if self.risk_manager.check_stop_loss(trade.entry_price, current_price, trade.direction):
                    log.warning("VWAP HARD STOP: %s @ $%.2f", symbol, current_price)
                    self._close_vwap_trade(symbol, "hard_stop")
                elif self.risk_manager.check_take_profit(trade.entry_price, current_price, trade.direction):
                    self._close_vwap_trade(symbol, "hard_tp")

            except Exception as e:
                log.error("Error checking VWAP position %s: %s", symbol, e)
                self.risk_manager.record_api_error()

    def _close_trade(self, symbol: str, reason: str):
        if self.dry_run:
            log.info("[DRY RUN] Would close ORB %s (%s)", symbol, reason)
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
                    price_diff = exit_price - trade.entry_price
                else:
                    price_diff = trade.entry_price - exit_price

                futures_pnl = price_diff * trade.contract_spec.point_value * trade.contracts
                costs = (trade.contract_spec.commission_per_side * 2) * trade.contracts
                futures_pnl -= costs

                self.risk_manager.record_trade(futures_pnl)
                log.info("ORB CLOSED %s %s: entry=$%.2f exit=$%.2f pnl=$%.2f (%s)",
                         trade.direction.upper(), symbol, trade.entry_price,
                         exit_price, futures_pnl, reason)
                self._log_trade(symbol, trade.direction,
                                "exit", trade.contracts, exit_price,
                                futures_pnl, reason, order_id, strategy="ORB")
                discord_trade("Futures_ORB", symbol, trade.direction, "exit",
                              trade.contracts, exit_price, pnl=futures_pnl,
                              reason=reason, entry_price=trade.entry_price)
                del self.active_trades[symbol]
            else:
                log.error("ORB close FAILED for %s (reason=%s), keeping in dict", symbol, reason)
        else:
            # No IB position found — remove from dict to avoid stale state
            log.warning("ORB %s: no IB position to close, removing from tracking", symbol)
            del self.active_trades[symbol]

    # ---- VWAP Reversion Exit Detection ----

    def _check_vwap_exits(self):
        """Check VWAP strategy exit signals (reversion to VWAP or timeout)."""
        for symbol in list(self.active_vwap_trades.keys()):
            trade = self.active_vwap_trades[symbol]
            try:
                # VWAP needs full day's bars for running VWAP calculation
                df = fetch_live_bars(symbol, lookback_minutes=400)
                if len(df) < 60:
                    continue

                vwap_strat = self._get_vwap_strategy(symbol)
                df = vwap_strat.generate_signals(df)

                last_signal = int(df["signal"].iloc[-1])
                price = float(df["close"].iloc[-1])

                if last_signal == 0:
                    log.info("VWAP EXIT: %s %s -> signal=0 @ $%.2f",
                             trade.direction.upper(), symbol, price)
                    self._close_vwap_trade(symbol, "vwap_strategy_exit")
                elif last_signal == -(1 if trade.direction == "long" else -1):
                    log.info("VWAP FLIP: %s %s -> signal=%d",
                             trade.direction.upper(), symbol, last_signal)
                    self._close_vwap_trade(symbol, "vwap_strategy_flip")

            except Exception as e:
                log.error("Error checking VWAP exit for %s: %s", symbol, e)
                self.risk_manager.record_api_error()

    # ---- VWAP Reversion Signal Scanning ----

    def _scan_for_vwap_signals(self):
        """Scan for VWAP Reversion entry signals (10 AM - 3 PM)."""
        for symbol in self.symbols:
            if symbol in self.active_vwap_trades or symbol in self.active_trades:
                continue

            # Churn circuit breaker
            now = now_et()
            recent = self._trade_timestamps.get(symbol, [])
            one_hour_ago = now - timedelta(hours=1)
            recent = [t for t in recent if t > one_hour_ago]
            self._trade_timestamps[symbol] = recent
            if len(recent) >= 5:
                log.warning("CHURN HALT VWAP %s: %d trades in last hour", symbol, len(recent))
                continue

            # Double-check IB for existing position (prevents duplicates on restart)
            try:
                ib_pos = get_position(symbol)
                if ib_pos is not None and ib_pos["qty"] != 0:
                    log.debug("SKIP VWAP %s: IB already has position (%d)", symbol, ib_pos["qty"])
                    continue
            except Exception:
                pass

            if not self._get_vwap_strategy(symbol):
                continue

            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                continue

            contract = self._get_contract(symbol)

            try:
                # VWAP needs full day's bars for running VWAP calculation
                df = fetch_live_bars(symbol, lookback_minutes=400)
                if len(df) < 60:
                    continue

                vwap_strat = self._get_vwap_strategy(symbol)
                df = vwap_strat.generate_signals(df)

                last_signal = int(df["signal"].iloc[-1])
                price = float(df["close"].iloc[-1])

                if last_signal == 0:
                    continue

                # Position sizing: fixed 0.3% stop for VWAP mean-reversion
                stop_distance = price * VWAP_STOP_PCT
                if last_signal > 0:
                    stop_price = price - stop_distance
                else:
                    stop_price = price + stop_distance

                contracts = self.risk_manager.calculate_contracts(
                    price, stop_price, contract, symbol
                )
                if contracts == 0:
                    continue

                ok, margin_msg = self.risk_manager.check_margin(contracts, contract)
                if not ok:
                    log.warning("VWAP MARGIN BLOCK %s: %s", symbol, margin_msg)
                    continue

                direction = "long" if last_signal > 0 else "short"
                side = "buy" if direction == "long" else "sell"

                log.info("VWAP SIGNAL: %s %s %d contracts @ ~$%.2f",
                         direction.upper(), symbol, contracts, price)

                if self.dry_run:
                    log.info("[DRY RUN] Would %s %d %s (VWAP)", side, contracts, symbol)
                    continue

                order_id, fill_type = submit_order(symbol, contracts, side)
                if order_id is None:
                    self.risk_manager.record_order_rejection()
                    continue

                fill = wait_for_fill(order_id)
                if fill and fill.get("status") == "filled":
                    self.active_vwap_trades[symbol] = ActiveTrade(
                        futures_symbol=symbol,
                        direction=direction,
                        contracts=fill["filled_qty"],
                        entry_price=fill["filled_avg_price"],
                        entry_time=now_et(),
                        order_id=order_id,
                        contract_spec=contract,
                        strategy="VWAP",
                    )
                    self._trade_timestamps.setdefault(symbol, []).append(now_et())
                    log.info("VWAP FILLED: %s %s %d contracts @ $%.2f (fill_type=%s)",
                             direction.upper(), symbol, fill["filled_qty"],
                             fill["filled_avg_price"], fill_type)
                    self._log_trade(symbol, direction, "entry",
                                    fill["filled_qty"], fill["filled_avg_price"],
                                    None, "vwap_filled", order_id,
                                    strategy="VWAP")
                    discord_trade("Futures_VWAP", symbol, direction, "entry",
                                  fill["filled_qty"], fill["filled_avg_price"])

            except Exception as e:
                log.error("Error scanning VWAP %s: %s", symbol, e, exc_info=True)
                self.risk_manager.record_api_error()

    def _close_vwap_trade(self, symbol: str, reason: str):
        """Close a VWAP Reversion position."""
        if self.dry_run:
            log.info("[DRY RUN] Would close VWAP %s (%s)", symbol, reason)
            if symbol in self.active_vwap_trades:
                del self.active_vwap_trades[symbol]
            return

        trade = self.active_vwap_trades.get(symbol)
        if trade is None:
            return

        order_id = close_position(symbol)
        if order_id:
            fill = wait_for_fill(order_id)
            if fill and fill.get("status") == "filled":
                exit_price = fill["filled_avg_price"]

                if trade.direction == "long":
                    price_diff = exit_price - trade.entry_price
                else:
                    price_diff = trade.entry_price - exit_price

                futures_pnl = price_diff * trade.contract_spec.point_value * trade.contracts
                costs = (trade.contract_spec.commission_per_side * 2) * trade.contracts
                futures_pnl -= costs

                self.risk_manager.record_trade(futures_pnl)
                log.info("VWAP CLOSED %s %s: entry=$%.2f exit=$%.2f pnl=$%.2f (%s)",
                         trade.direction.upper(), symbol, trade.entry_price,
                         exit_price, futures_pnl, reason)
                self._log_trade(symbol, trade.direction,
                                "exit", trade.contracts, exit_price,
                                futures_pnl, reason, order_id,
                                strategy="VWAP")
                discord_trade("Futures_VWAP", symbol, trade.direction, "exit",
                              trade.contracts, exit_price, pnl=futures_pnl,
                              reason=reason, entry_price=trade.entry_price)
                del self.active_vwap_trades[symbol]
            else:
                log.error("VWAP close FAILED for %s (reason=%s), keeping in dict", symbol, reason)
        else:
            log.warning("VWAP %s: no IB position to close, removing from tracking", symbol)
            del self.active_vwap_trades[symbol]
