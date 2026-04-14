"""Alpaca broker interface for live/paper trading.

Order flow: limit at midpoint (5s timeout) -> cancel -> market fallback.
Every order logs fill_type = "limit" or "market_fallback" for tracking.
"""

import logging
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

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


_data_client: StockHistoricalDataClient | None = None


def _get_data_client() -> StockHistoricalDataClient:
    """Get or create the data client for quotes."""
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _data_client


def get_midpoint(symbol: str) -> float | None:
    """Get current bid/ask midpoint price for a symbol."""
    try:
        from alpaca.data.enums import DataFeed
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        quote = _get_data_client().get_stock_latest_quote(req)
        q = quote[symbol]
        bid = float(q.bid_price)
        ask = float(q.ask_price)
        if bid > 0 and ask > 0:
            mid = round((bid + ask) / 2, 2)
            log.debug("Quote %s: bid=$%.2f ask=$%.2f mid=$%.2f", symbol, bid, ask, mid)
            return mid
        return None
    except Exception as e:
        log.debug("Failed to get quote for %s: %s", symbol, e)
        return None


def submit_order(symbol: str, qty: int, side: str,
                 limit_timeout_sec: int = 5, max_retries: int = 3) -> tuple[str | None, str]:
    """Submit order: try limit at midpoint first, fall back to market.

    Returns:
        (order_id, fill_type) where fill_type is "limit", "market_fallback",
        or "market" (if midpoint unavailable). Returns (None, "failed") on failure.
    """
    if qty <= 0:
        log.warning("Cannot submit order with qty=%d", qty)
        return None, "failed"

    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    # Step 1: Try limit order at midpoint
    midpoint = get_midpoint(symbol)
    if midpoint is not None and midpoint > 0:
        try:
            limit_order = get_client().submit_order(
                LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=midpoint,
                )
            )
            limit_id = str(limit_order.id)
            log.info("LIMIT ORDER %s %d %s @ $%.2f -> id=%s",
                     side.upper(), qty, symbol, midpoint, limit_id)

            # Wait up to limit_timeout_sec for fill
            start = time.time()
            while time.time() - start < limit_timeout_sec:
                try:
                    order = get_client().get_order_by_id(limit_id)
                    if order.status == OrderStatus.FILLED:
                        log.info("LIMIT FILLED %s %d %s @ $%s (fill_type=limit)",
                                 side.upper(), qty, symbol, order.filled_avg_price)
                        return limit_id, "limit"
                    elif order.status in (OrderStatus.CANCELED, OrderStatus.REJECTED,
                                          OrderStatus.EXPIRED):
                        log.info("Limit order %s ended: %s", limit_id, order.status)
                        break
                except Exception as e:
                    log.debug("Error checking limit order: %s", e)
                time.sleep(0.5)

            # Step 2: Limit didn't fill — cancel it
            try:
                get_client().cancel_order_by_id(limit_id)
                log.info("Cancelled unfilled limit order %s after %ds",
                         limit_id, limit_timeout_sec)
            except Exception as e:
                # May already be filled or cancelled
                log.debug("Cancel limit order error (may be filled): %s", e)
                # Check one more time if it filled during cancel
                try:
                    order = get_client().get_order_by_id(limit_id)
                    if order.status == OrderStatus.FILLED:
                        log.info("LIMIT FILLED (during cancel) %s %d %s @ $%s (fill_type=limit)",
                                 side.upper(), qty, symbol, order.filled_avg_price)
                        return limit_id, "limit"
                except Exception:
                    pass

            log.info("Limit order timeout — falling back to market order")

        except Exception as e:
            log.warning("Limit order failed: %s — falling back to market", e)

    # Step 3: Market order fallback
    fill_type = "market_fallback" if midpoint is not None else "market"
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
            log.info("MARKET ORDER %s %d %s -> id=%s (fill_type=%s)",
                     side.upper(), qty, symbol, order.id, fill_type)
            return str(order.id), fill_type

        except Exception as e:
            log.error("Market order failed (attempt %d/%d): %s %d %s -> %s",
                      attempt + 1, max_retries, side, qty, symbol, e)
            if attempt < max_retries - 1:
                time.sleep(1)

    log.error("Order FAILED after all attempts: %s %d %s", side, qty, symbol)
    return None, "failed"


def submit_market_order(symbol: str, qty: int, side: str,
                        max_retries: int = 3) -> str | None:
    """Submit order using limit-first-then-market flow.

    Backward compatible: returns just the order ID (or None).
    The fill_type is logged but not returned to preserve the interface.
    """
    order_id, fill_type = submit_order(symbol, qty, side,
                                       limit_timeout_sec=5,
                                       max_retries=max_retries)
    return order_id


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
