#!/usr/bin/env python3
"""Overnight mean-reversion trader for MNQ via Interactive Brokers.

Runs separately from the cash-hours ORB trader. Trades during dead hours
(8 PM - 2 AM ET) and forces flat by 9:25 AM before cash ORB starts.

Uses a different IB clientId (3) to avoid conflicts with the ORB trader (1).

Usage:
    python run_overnight.py                # Paper trading (live)
    python run_overnight.py --dry-run      # Signals only, no orders
    python run_overnight.py --port 4002    # Use IB Gateway port
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

from trading.config import (
    IB_HOST, IB_PORT, LOG_DIR,
    OVERNIGHT_TRADE_START, OVERNIGHT_TRADE_END, OVERNIGHT_FORCE_CLOSE,
    OVERNIGHT_REVERSION_DEFAULTS, INITIAL_CAPITAL,
    DISCORD_WEBHOOK_URL,
)

ET = pytz.timezone("America/New_York")
OVERNIGHT_CLIENT_ID = 3
SYMBOL = "MNQ"

# CRITICAL: Set clientId=3 BEFORE broker module is imported, so the
# module-level `from trading.config import IB_CLIENT_ID` picks up 3.
import trading.config as _cfg
_cfg.IB_CLIENT_ID = OVERNIGHT_CLIENT_ID

from trading.strategies.overnight_reversion import OvernightReversion
from trading.data.contracts import CONTRACTS
from trading.execution.broker import (
    get_ib, get_account_equity, get_available_margin, get_position,
    submit_market_order, wait_for_fill, close_position, close_all_positions,
    cancel_all_orders, get_contract_month, disconnect, is_connected,
)
from trading.data.live_feed import fetch_live_bars
from trading.discord_alerts import _send as send_discord


def now_et() -> datetime:
    return datetime.now(ET)


def current_minute() -> int:
    now = now_et()
    return now.hour * 60 + now.minute


def time_str_to_minutes(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def in_trading_window() -> bool:
    """Check if we're in the overnight trading window (8 PM - 2 AM ET)."""
    h = now_et().hour
    return h >= 20 or h < 2


def in_force_close_zone() -> bool:
    """Check if we're past the entry cutoff but before cash open."""
    h = now_et().hour
    m = now_et().minute
    # 2 AM through 9:25 AM
    return (2 <= h < 9) or (h == 9 and m < 25)


def should_be_flat() -> bool:
    """Check if all positions must be closed (9:25 AM or later)."""
    mod = current_minute()
    return 9 * 60 + 25 <= mod < 20 * 60  # 9:25 AM to 8 PM


class OvernightTrader:
    """Live overnight reversion trader for MNQ."""

    def __init__(self, dry_run: bool = False, port: int = None, client_id: int = None):
        self.dry_run = dry_run
        self.port = port or IB_PORT
        self.client_id = client_id or OVERNIGHT_CLIENT_ID
        self.symbol = SYMBOL
        self.contract = CONTRACTS[self.symbol]
        self.strategy = OvernightReversion(**OVERNIGHT_REVERSION_DEFAULTS)
        self.running = True

        self.position = 0  # contracts: +1 long, -1 short, 0 flat
        self.entry_price = 0.0
        self.entry_time = None
        self.session_pnl = 0.0
        self.trades_tonight = 0
        self._log_dir = None

    def _ensure_log_dir(self):
        log_dir = os.path.join(LOG_DIR, now_et().strftime("%Y-%m-%d"))
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir
        return log_dir

    def _log_trade(self, direction, action, contracts, price, pnl, reason):
        if self._log_dir is None:
            return
        path = os.path.join(self._log_dir, "overnight_trades.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "symbol", "strategy", "direction",
                            "action", "contracts", "price", "pnl", "reason"])
            w.writerow([
                now_et().isoformat(), self.symbol, "overnight_reversion",
                direction, action, contracts, f"{price:.2f}",
                f"{pnl:.2f}" if pnl is not None else "", reason,
            ])

    def _discord_alert(self, message: str, webhook: str = None):
        """Send alert to Discord. Default = trade webhook."""
        try:
            send_discord(f"[ON_Reversion MNQ] {message}", webhook=webhook)
        except Exception as e:
            log.debug("Discord alert failed: %s", e)

    def connect(self):
        """Connect to IB with overnight-specific clientId."""
        import trading.config as cfg
        cfg.IB_PORT = self.port
        cfg.IB_CLIENT_ID = self.client_id

        log.info("Connecting to IB at %s:%d (clientId=%d)", IB_HOST, self.port, self.client_id)
        try:
            ib = get_ib()
            equity = get_account_equity()
            log.info("Connected. Equity: $%.0f", equity)
            return True
        except Exception as e:
            log.error("Cannot connect to IB: %s", e)
            return False

    def enter_trade(self, direction: str, price: float):
        """Enter a position."""
        contracts = 1  # Conservative: 1 contract during overnight

        if self.dry_run:
            log.info("DRY RUN: Would enter %s %d %s at %.2f",
                     direction, contracts, self.symbol, price)
        else:
            try:
                side_buy_sell = "buy" if direction == "long" else "sell"
                order_id = submit_market_order(self.symbol, contracts, side_buy_sell)
                fill = wait_for_fill(order_id, timeout_sec=30) if order_id else None
                if fill and fill.get("status") == "filled":
                    price = fill["filled_avg_price"]
                    log.info("FILLED: %s %d %s at %.2f", side_buy_sell.upper(), contracts, self.symbol, price)
                elif order_id:
                    # Order submitted but didn't confirm fill in time — trust IB and continue.
                    # The sanity check will prevent duplicate entries.
                    log.warning("Fill not confirmed in 30s; proceeding assuming filled")
                else:
                    log.warning("Order not filled within timeout")
                    return
            except Exception as e:
                log.error("Order failed: %s", e)
                return

        self.position = 1 if direction == "long" else -1
        self.entry_price = price
        self.entry_time = now_et()

        self._log_trade(direction, "ENTRY", contracts, price, None, "signal")
        self._discord_alert(f"ENTRY {direction.upper()} {contracts} @ {price:.2f}")

    def exit_trade(self, price: float, reason: str):
        """Exit current position."""
        if self.position == 0:
            return

        direction = "long" if self.position > 0 else "short"
        pnl_per = (price - self.entry_price) * self.position * self.contract.point_value
        costs = self.contract.commission_per_side * 2 + self.contract.tick_value * 2
        pnl = pnl_per - costs

        if not self.dry_run:
            try:
                side_bs = "sell" if self.position > 0 else "buy"
                order_id = submit_market_order(self.symbol, 1, side_bs)
                fill = wait_for_fill(order_id, timeout_sec=10) if order_id else None
                if fill and fill.get("status") == "filled":
                    price = fill["filled_avg_price"]
            except Exception as e:
                log.error("Exit order failed: %s", e)

        self.session_pnl += pnl
        self.trades_tonight += 1

        log.info("EXIT %s at %.2f (reason=%s, pnl=$%.2f)", direction, price, reason, pnl)
        self._log_trade(direction, "EXIT", 1, price, pnl, reason)
        self._discord_alert(f"EXIT {direction.upper()} @ {price:.2f} | {reason} | P&L ${pnl:+.0f}")

        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None

    def run_iteration(self):
        """One iteration of the overnight trading loop."""
        try:
            # Fetch recent bars
            bars = fetch_live_bars(self.symbol, lookback_minutes=400, use_rth=False)
            if bars is None or len(bars) < 30:
                return

            df = bars.copy()
            if "dt" not in df.columns and "date" in df.columns:
                df = df.rename(columns={"date": "dt"})
            if not hasattr(df["dt"].dt, "tz") or df["dt"].dt.tz is None:
                df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize("UTC").dt.tz_convert(ET)
            else:
                df["dt"] = df["dt"].dt.tz_convert(ET)

            # Generate signals
            df = self.strategy.generate_signals(df)

            # Get latest signal
            latest = df.iloc[-1]
            signal = int(latest["signal"])
            price = float(latest["close"])

            # Act on signal
            if should_be_flat():
                if self.position != 0:
                    self.exit_trade(price, "force_close_9:25AM")
                return

            if in_force_close_zone():
                if self.position != 0:
                    self.exit_trade(price, "past_entry_cutoff")
                return

            if in_trading_window():
                if self.position == 0 and signal != 0:
                    direction = "long" if signal > 0 else "short"
                    self.enter_trade(direction, price)
                elif self.position != 0 and signal == 0:
                    self.exit_trade(price, "signal_revert")
                elif self.position != 0 and signal != 0 and signal != self.position:
                    self.exit_trade(price, "signal_flip")
                    direction = "long" if signal > 0 else "short"
                    self.enter_trade(direction, price)

        except Exception as e:
            log.error("Iteration error: %s", e)

    def run(self):
        """Main overnight trading loop."""
        self._ensure_log_dir()

        if not self.connect():
            return

        # Reconcile: adopt any existing MNQ position on startup.
        # Note: get_position() already divides avgCost by point_value, so
        # avg_entry is the actual price.
        try:
            existing = get_position(self.symbol)
            if existing is not None and existing["qty"] != 0:
                qty = existing["qty"]
                self.position = 1 if qty > 0 else -1
                self.entry_price = existing.get("avg_entry", 0.0)
                self.entry_time = now_et()
                log.warning("RECONCILED existing %s position: %s %d @ $%.2f",
                            self.symbol, "long" if qty > 0 else "short", abs(qty), self.entry_price)
        except Exception as e:
            log.error("Reconciliation failed: %s", e)

        log.info("=" * 60)
        log.info("OVERNIGHT REVERSION TRADER STARTED")
        log.info("  Symbol:    %s", self.symbol)
        log.info("  Strategy:  %s %s", self.strategy.name, self.strategy.get_params())
        log.info("  Window:    %s - %s ET", OVERNIGHT_TRADE_START, OVERNIGHT_TRADE_END)
        log.info("  Flat by:   %s ET", OVERNIGHT_FORCE_CLOSE)
        log.info("  Dry run:   %s", self.dry_run)
        log.info("  ClientId:  %d", self.client_id)
        log.info("=" * 60)

        # Startup message intentionally NOT sent to trade webhook (noise).
        # Trade entries/exits still alert normally via _discord_alert().
        log.info("Startup complete (no Discord alert for startup)")

        try:
            while self.running:
                now = now_et()
                h = now.hour

                # Active trading window: 8 PM - 9:25 AM
                if h >= 20 or h < 9 or (h == 9 and now.minute < 25):
                    self.run_iteration()
                    time.sleep(60)  # 1-minute bars, check every minute
                elif 9 <= h < 20:
                    # Daytime: sleep until 8 PM
                    if h < 20:
                        target = now.replace(hour=20, minute=0, second=0)
                        if target <= now:
                            target += timedelta(days=1)
                        wait = (target - now).total_seconds()
                        wait = min(wait, 3600)  # Max 1hr chunks
                        log.info("Daytime. Sleeping %.0f min until 8 PM ET.", wait / 60)

                        # EOD summary if we traded
                        if self.trades_tonight > 0:
                            self._print_session_summary()
                            self.session_pnl = 0.0
                            self.trades_tonight = 0

                        time.sleep(wait)
                        self._ensure_log_dir()

                        # Reconnect if needed
                        if not is_connected():
                            self.connect()

        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            if self.position != 0:
                try:
                    price = float(fetch_live_bars(self.symbol, lookback_minutes=60, use_rth=False).iloc[-1]["close"])
                    self.exit_trade(price, "shutdown")
                except Exception:
                    pass
            self._print_session_summary()
            disconnect()
            log.info("Overnight trader stopped.")

    def _print_session_summary(self):
        log.info("=" * 60)
        log.info("OVERNIGHT SESSION SUMMARY")
        log.info("  Date:      %s", now_et().strftime("%Y-%m-%d"))
        log.info("  Trades:    %d", self.trades_tonight)
        log.info("  P&L:       $%.2f", self.session_pnl)
        log.info("=" * 60)

        # Session summary goes to the SUMMARY webhook
        from trading.discord_alerts import SUMMARY_WEBHOOK
        self._discord_alert(
            f"Session complete: {self.trades_tonight} trades, P&L ${self.session_pnl:+.0f}",
            webhook=SUMMARY_WEBHOOK,
        )

        if self._log_dir:
            summary = {
                "date": now_et().strftime("%Y-%m-%d"),
                "strategy": "overnight_reversion",
                "symbol": self.symbol,
                "trades": self.trades_tonight,
                "session_pnl": round(self.session_pnl, 2),
            }
            path = os.path.join(self._log_dir, "overnight_summary.json")
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)


log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Overnight Reversion Trader (MNQ via IB)")
    parser.add_argument("--dry-run", action="store_true", help="Signals only, no orders")
    parser.add_argument("--port", type=int, default=None, help="IB port")
    parser.add_argument("--client-id", type=int, default=OVERNIGHT_CLIENT_ID,
                        help="IB client ID (default: 2)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    today = now_et().strftime("%Y-%m-%d")
    log_dir = os.path.join(LOG_DIR, today)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "overnight.log")),
        ],
    )

    trader = OvernightTrader(
        dry_run=args.dry_run,
        port=args.port,
        client_id=args.client_id,
    )
    trader.run()


if __name__ == "__main__":
    main()
