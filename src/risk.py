"""Position sizing and risk management."""

import logging
from src.config import RISK_PER_TRADE_PCT, MAX_POSITION_PCT, STOP_LOSS_PCT

log = logging.getLogger(__name__)


def calculate_shares(equity: float, entry_price: float) -> int:
    """Calculate how many shares to short based on risk parameters.

    Uses the smaller of:
    - RISK_PER_TRADE_PCT of equity / stop_loss_distance (risk-based sizing)
    - MAX_POSITION_PCT of equity / entry_price (max position cap)
    """
    if entry_price <= 0 or equity <= 0:
        return 0

    # Risk-based: risk_amount / stop_distance_per_share
    risk_amount = equity * (RISK_PER_TRADE_PCT / 100.0)
    stop_distance = entry_price * (STOP_LOSS_PCT / 100.0)
    risk_shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0

    # Position cap
    max_notional = equity * (MAX_POSITION_PCT / 100.0)
    cap_shares = int(max_notional / entry_price)

    shares = min(risk_shares, cap_shares)
    shares = max(shares, 0)

    log.info(
        "Position sizing: equity=%.2f, price=%.2f, risk_shares=%d, cap_shares=%d -> %d shares",
        equity, entry_price, risk_shares, cap_shares, shares,
    )
    return shares


def check_stop_loss(entry_price: float, current_price: float) -> bool:
    """Check if current price has hit the stop loss (for a short position).

    For a short, stop loss triggers when price moves UP beyond the threshold.
    """
    if entry_price <= 0:
        return False
    move_pct = (current_price / entry_price - 1.0) * 100.0
    hit = move_pct >= STOP_LOSS_PCT
    if hit:
        log.warning("STOP LOSS HIT: entry=%.2f, current=%.2f, move=+%.3f%%", entry_price, current_price, move_pct)
    return hit
