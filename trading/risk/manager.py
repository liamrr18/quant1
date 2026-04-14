"""Risk management: position sizing, daily loss limits, circuit breakers."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import pytz

from trading.config import (
    MAX_POSITION_PCT, MAX_DAILY_LOSS_PCT, MAX_CONCURRENT_POSITIONS,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


@dataclass
class RiskState:
    """Tracks risk state for the current trading session."""
    starting_equity: float = 0.0
    current_equity: float = 0.0
    daily_pnl: float = 0.0
    trades_today: int = 0
    open_positions: int = 0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    order_rejections: int = 0
    api_errors: int = 0
    last_error_time: float = 0.0


class RiskManager:
    """Centralized risk management for the trading session."""

    def __init__(self, starting_equity: float):
        self.state = RiskState(
            starting_equity=starting_equity,
            current_equity=starting_equity,
        )
        self.max_daily_loss = starting_equity * MAX_DAILY_LOSS_PCT
        log.info("RiskManager: equity=$%.0f, max_daily_loss=$%.0f",
                 starting_equity, self.max_daily_loss)

    def calculate_shares(self, price: float) -> int:
        """Calculate position size based on current equity and risk limits."""
        if price <= 0 or self.state.halted:
            return 0

        equity = self.state.current_equity
        max_notional = equity * MAX_POSITION_PCT
        shares = int(max_notional / price)

        # Reduce size if we're already down today
        if self.state.daily_pnl < 0:
            loss_ratio = abs(self.state.daily_pnl) / self.max_daily_loss
            if loss_ratio > 0.5:
                shares = int(shares * 0.5)  # Half size when 50%+ of daily loss used
                log.info("Reduced size to %d shares (daily loss %.0f%% of limit)",
                         shares, loss_ratio * 100)

        return max(shares, 0)

    def can_trade(self) -> tuple[bool, str]:
        """Check if we're allowed to open a new position."""
        if self.state.halted:
            return False, f"HALTED: {self.state.halt_reason}"

        if self.state.daily_pnl <= -self.max_daily_loss:
            self.halt("Daily loss limit hit: $%.0f" % self.state.daily_pnl)
            return False, "Daily loss limit reached"

        if self.state.open_positions >= MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS})"

        if self.state.consecutive_losses >= 5:
            self.halt("5 consecutive losses")
            return False, "Consecutive loss limit"

        return True, "OK"

    def check_stop_loss(self, entry_price: float, current_price: float,
                        direction: str) -> bool:
        """Check if stop loss is hit."""
        if entry_price <= 0:
            return False
        if direction == "long":
            return current_price <= entry_price * (1 - STOP_LOSS_PCT)
        else:
            return current_price >= entry_price * (1 + STOP_LOSS_PCT)

    def check_take_profit(self, entry_price: float, current_price: float,
                          direction: str) -> bool:
        """Check if take profit is hit."""
        if entry_price <= 0:
            return False
        if direction == "long":
            return current_price >= entry_price * (1 + TAKE_PROFIT_PCT)
        else:
            return current_price <= entry_price * (1 - TAKE_PROFIT_PCT)

    def record_trade(self, pnl: float):
        """Record a completed trade."""
        self.state.trades_today += 1
        self.state.daily_pnl += pnl

        if pnl > 0:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

        log.info("Trade P&L: $%.2f | Daily P&L: $%.2f | Trades: %d",
                 pnl, self.state.daily_pnl, self.state.trades_today)

        # Check daily loss limit
        if self.state.daily_pnl <= -self.max_daily_loss:
            self.halt("Daily loss limit reached: $%.0f" % self.state.daily_pnl)

    def record_order_rejection(self):
        """Track order rejections for circuit breaker."""
        self.state.order_rejections += 1
        if self.state.order_rejections >= 5:
            self.halt("Too many order rejections (%d)" % self.state.order_rejections)

    def record_api_error(self):
        """Track API errors for circuit breaker."""
        self.state.api_errors += 1
        now = time.time()

        # If 3+ errors in 60 seconds, halt
        if self.state.api_errors >= 3:
            if now - self.state.last_error_time < 60:
                self.halt("Rapid API errors (%d in <60s)" % self.state.api_errors)
            else:
                self.state.api_errors = 1  # Reset if errors are spread out

        self.state.last_error_time = now

    def halt(self, reason: str):
        """Halt all trading."""
        self.state.halted = True
        self.state.halt_reason = reason
        log.warning("TRADING HALTED: %s", reason)

    def update_equity(self, equity: float):
        """Update current equity."""
        self.state.current_equity = equity
        self.state.daily_pnl = equity - self.state.starting_equity

    def update_positions(self, count: int):
        """Update open position count."""
        self.state.open_positions = count

    def status(self) -> dict:
        """Get current risk state."""
        return {
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "daily_pnl": self.state.daily_pnl,
            "daily_loss_limit": self.max_daily_loss,
            "pnl_vs_limit": self.state.daily_pnl / self.max_daily_loss * 100 if self.max_daily_loss > 0 else 0,
            "trades_today": self.state.trades_today,
            "open_positions": self.state.open_positions,
            "consecutive_losses": self.state.consecutive_losses,
        }
