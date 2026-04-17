"""Futures risk management: margin-aware sizing, daily loss limits, circuit breakers.

KEY DIFFERENCES FROM EQUITY RISK MANAGER:
- Position sizing by contracts, not shares
- Margin requirement checks before entry
- Per-trade max loss in dollars (based on stop distance)
- Circuit breaker: reduce size at 3% daily loss, halt at 5%
- Margin safety buffer: require 2x margin available
"""

import logging
import time
from dataclasses import dataclass

import pytz

from trading.config import (
    MAX_RISK_PER_TRADE_PCT, MAX_DAILY_LOSS_PCT, CIRCUIT_BREAKER_PCT,
    MAX_CONTRACTS_MES, MAX_CONTRACTS_MNQ, MAX_CONTRACTS_MGC,
    MARGIN_SAFETY_MULTIPLE,
    MAX_CONCURRENT_POSITIONS, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)
from trading.data.contracts import FuturesContract, CONTRACTS, total_cost_per_contract

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

MAX_CONTRACTS = {"MES": MAX_CONTRACTS_MES, "MNQ": MAX_CONTRACTS_MNQ, "MGC": MAX_CONTRACTS_MGC}


@dataclass
class RiskState:
    """Tracks risk state for the current futures trading session."""
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
    circuit_breaker_active: bool = False


class FuturesRiskManager:
    """Risk management for futures trading with margin awareness."""

    def __init__(self, starting_equity: float):
        self.state = RiskState(
            starting_equity=starting_equity,
            current_equity=starting_equity,
        )
        self.max_daily_loss = starting_equity * MAX_DAILY_LOSS_PCT
        self.circuit_breaker_level = starting_equity * CIRCUIT_BREAKER_PCT
        log.info("FuturesRiskManager: equity=$%.0f, daily_loss_limit=$%.0f, "
                 "circuit_breaker=$%.0f",
                 starting_equity, self.max_daily_loss, self.circuit_breaker_level)

    def calculate_contracts(self, entry_price: float, stop_price: float,
                            contract: FuturesContract,
                            futures_symbol: str) -> int:
        """Calculate position size in contracts.

        Sizing formula:
            contracts = min(
                max_contracts_cap,
                floor(risk_dollars / risk_per_contract),
                floor(available_margin / margin_per_contract)
            )

        Risk per contract = |entry - stop| * multiplier + costs
        """
        if entry_price <= 0 or stop_price <= 0 or self.state.halted:
            return 0

        equity = self.state.current_equity
        risk_dollars = equity * MAX_RISK_PER_TRADE_PCT

        # Apply circuit breaker: half size if daily loss > 3%
        if self.state.circuit_breaker_active:
            risk_dollars *= 0.5
            log.info("Circuit breaker active: reducing risk to $%.0f", risk_dollars)

        # Dollar risk per contract
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0

        risk_per_contract = stop_distance * contract.multiplier
        cost_per_contract = total_cost_per_contract(contract)
        effective_risk = risk_per_contract + cost_per_contract

        contracts_by_risk = int(risk_dollars / effective_risk)

        # Margin constraint with safety buffer
        available_margin = equity / MARGIN_SAFETY_MULTIPLE
        contracts_by_margin = int(available_margin / contract.margin_intraday)

        # Hard cap
        max_cap = MAX_CONTRACTS.get(futures_symbol, 5)  # default 5 for safety

        contracts = min(contracts_by_risk, contracts_by_margin, max_cap)

        log.info("Sizing %s: risk=$%.0f, risk/contract=$%.0f, "
                 "by_risk=%d, by_margin=%d, cap=%d -> %d contracts",
                 futures_symbol, risk_dollars, effective_risk,
                 contracts_by_risk, contracts_by_margin, max_cap, contracts)

        return max(contracts, 0)

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

    def check_margin(self, contracts: int, contract: FuturesContract) -> tuple[bool, str]:
        """Verify sufficient margin for a new position."""
        required = contracts * contract.margin_intraday
        available = self.state.current_equity / MARGIN_SAFETY_MULTIPLE

        if required > available:
            return False, (f"Insufficient margin: need ${required:,.0f} "
                          f"(available ${available:,.0f} with {MARGIN_SAFETY_MULTIPLE}x buffer)")
        return True, "OK"

    def check_stop_loss(self, entry_price: float, current_price: float,
                        direction: str) -> bool:
        if entry_price <= 0:
            return False
        if direction == "long":
            return current_price <= entry_price * (1 - STOP_LOSS_PCT)
        else:
            return current_price >= entry_price * (1 + STOP_LOSS_PCT)

    def check_take_profit(self, entry_price: float, current_price: float,
                          direction: str) -> bool:
        if entry_price <= 0:
            return False
        if direction == "long":
            return current_price >= entry_price * (1 + TAKE_PROFIT_PCT)
        else:
            return current_price <= entry_price * (1 - TAKE_PROFIT_PCT)

    def record_trade(self, pnl: float):
        """Record a completed trade and check circuit breakers."""
        self.state.trades_today += 1
        self.state.daily_pnl += pnl

        if pnl > 0:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

        log.info("Trade P&L: $%.2f | Daily P&L: $%.2f | Trades: %d",
                 pnl, self.state.daily_pnl, self.state.trades_today)

        # Circuit breaker: reduce size at 3% daily loss
        if self.state.daily_pnl <= -self.circuit_breaker_level:
            if not self.state.circuit_breaker_active:
                self.state.circuit_breaker_active = True
                log.warning("CIRCUIT BREAKER: daily loss $%.0f exceeds %.0f%% threshold. "
                           "Reducing position size by 50%%.",
                           self.state.daily_pnl, CIRCUIT_BREAKER_PCT * 100)

        # Hard halt at daily loss limit
        if self.state.daily_pnl <= -self.max_daily_loss:
            self.halt("Daily loss limit reached: $%.0f" % self.state.daily_pnl)

    def record_order_rejection(self):
        self.state.order_rejections += 1
        if self.state.order_rejections >= 5:
            self.halt("Too many order rejections (%d)" % self.state.order_rejections)

    def record_api_error(self):
        self.state.api_errors += 1
        now = time.time()
        if self.state.api_errors >= 3:
            if now - self.state.last_error_time < 60:
                self.halt("Rapid API errors (%d in <60s)" % self.state.api_errors)
            else:
                self.state.api_errors = 1
        self.state.last_error_time = now

    def halt(self, reason: str):
        self.state.halted = True
        self.state.halt_reason = reason
        log.warning("FUTURES TRADING HALTED: %s", reason)

    def update_equity(self, equity: float):
        self.state.current_equity = equity
        self.state.daily_pnl = equity - self.state.starting_equity

    def update_positions(self, count: int):
        self.state.open_positions = count

    def status(self) -> dict:
        return {
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "daily_pnl": self.state.daily_pnl,
            "daily_loss_limit": self.max_daily_loss,
            "pnl_vs_limit": self.state.daily_pnl / self.max_daily_loss * 100 if self.max_daily_loss > 0 else 0,
            "trades_today": self.state.trades_today,
            "open_positions": self.state.open_positions,
            "consecutive_losses": self.state.consecutive_losses,
            "circuit_breaker": self.state.circuit_breaker_active,
        }
