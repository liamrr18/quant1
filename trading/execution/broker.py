"""Alpaca broker interface for live/paper trading."""

import logging
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus

from trading.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

log = logging.getLogger(__name__)

_client: TradingClient | None = None


def get_client() -> TradingClient:
    global _client
    if _client is None:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
        _client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=ALPACA_PAPER,
        )
        acct = _client.get_account()
        log.info("Alpaca connected: paper=%s, equity=$%s, buying_power=$%s",
                 ALPACA_PAPER, acct.equity, acct.buying_power)
    return _client


def get_account_equity() -> float:
    return float(get_client().get_account().equity)


def get_buying_power() -> float:
    return float(get_client().get_account().buying_power)


def get_position(symbol: str) -> dict | None:
    """Get current position for a symbol, or None if flat."""
    try:
        pos = get_client().get_open_position(symbol)
        return {
            "symbol": pos.symbol,
            "qty": int(pos.qty),
            "side": pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
            "avg_entry": float(pos.avg_entry_price),
            "market_value": float(pos.market_value),
            "unrealized_pl": float(pos.unrealized_pl),
        }
    except Exception:
        return None


def get_all_positions() -> list[dict]:
    """Get all open positions."""
    positions = get_client().get_all_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": int(p.qty),
            "side": p.side.value if hasattr(p.side, 'value') else str(p.side),
            "avg_entry": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        }
        for p in positions
    ]


def submit_market_order(symbol: str, qty: int, side: str,
                        max_retries: int = 3) -> str | None:
    """Submit a market order with retry logic.

    Args:
        symbol: Ticker symbol
        qty: Number of shares (positive)
        side: "buy" or "sell"
        max_retries: Number of retry attempts on failure

    Returns:
        Order ID string, or None on failure
    """
    if qty <= 0:
        log.warning("Cannot submit order with qty=%d", qty)
        return None

    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    for attempt in range(max_retries):
        try:
            order = get_client().submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info("ORDER %s %d %s -> id=%s status=%s",
                     side.upper(), qty, symbol, order.id, order.status)
            return str(order.id)

        except Exception as e:
            log.error("Order failed (attempt %d/%d): %s %d %s -> %s",
                      attempt + 1, max_retries, side, qty, symbol, e)
            if attempt < max_retries - 1:
                time.sleep(1)

    log.error("Order FAILED after %d retries: %s %d %s", max_retries, side, qty, symbol)
    return None


def wait_for_fill(order_id: str, timeout_sec: int = 30) -> dict | None:
    """Wait for an order to fill. Returns fill info or None on timeout."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            order = get_client().get_order_by_id(order_id)
            status = order.status
            if status == OrderStatus.FILLED:
                return {
                    "id": str(order.id),
                    "status": "filled",
                    "filled_qty": int(order.filled_qty),
                    "filled_avg_price": float(order.filled_avg_price),
                }
            elif status in (OrderStatus.CANCELED, OrderStatus.REJECTED,
                          OrderStatus.EXPIRED):
                log.warning("Order %s ended with status: %s", order_id, status)
                return {"id": str(order.id), "status": str(status)}
        except Exception as e:
            log.warning("Error checking order %s: %s", order_id, e)

        time.sleep(0.5)

    log.warning("Order %s timed out after %ds", order_id, timeout_sec)
    return None


def close_position(symbol: str) -> str | None:
    """Close any open position in a symbol. Returns order ID."""
    pos = get_position(symbol)
    if pos is None:
        return None

    qty = abs(pos["qty"])
    side = "sell" if pos["side"] == "long" else "buy"
    log.info("Closing %s: %s %d shares", symbol, side, qty)
    return submit_market_order(symbol, qty, side)


def close_all_positions():
    """Emergency: close everything."""
    log.warning("CLOSING ALL POSITIONS")
    try:
        get_client().close_all_positions(cancel_orders=True)
    except Exception as e:
        log.error("Error closing all positions: %s", e)


def cancel_all_orders():
    """Cancel all open orders."""
    try:
        get_client().cancel_orders()
        log.info("All open orders cancelled")
    except Exception as e:
        log.error("Error cancelling orders: %s", e)


def is_market_open() -> bool:
    """Check if the market is currently open."""
    try:
        clock = get_client().get_clock()
        return clock.is_open
    except Exception:
        return False
