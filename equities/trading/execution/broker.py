"""IB TWS broker interface for equity live/paper trading.

Order flow: limit at midpoint (5s timeout) -> cancel -> market fallback.
Every order logs fill_type = "limit" or "market_fallback" for tracking.

Adapted from the futures broker (ib_insync). Replaces Alpaca.
"""

import logging
import time
import threading
from datetime import datetime

import pytz
from ib_insync import IB, Stock, MarketOrder, LimitOrder

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

_ib: IB | None = None
_lock = threading.Lock()
_stock_cache: dict[str, Stock] = {}


def _resolve_stock(symbol: str) -> Stock:
    """Resolve a stock symbol to a qualified IB contract."""
    if symbol in _stock_cache:
        return _stock_cache[symbol]

    ib = get_ib()
    contract = Stock(symbol, "SMART", "USD")
    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            _stock_cache[symbol] = qualified[0]
            log.info("Resolved stock %s -> conId=%s", symbol, qualified[0].conId)
            return qualified[0]
    except Exception as e:
        log.error("Failed to qualify stock %s: %s", symbol, e)

    raise RuntimeError(f"Cannot resolve IB stock contract for {symbol}")


def get_ib() -> IB:
    """Get or create IB connection with auto-reconnect."""
    global _ib
    with _lock:
        if _ib is not None and _ib.isConnected():
            return _ib

        if _ib is not None:
            log.warning("IB connection lost, reconnecting...")
            try:
                _ib.disconnect()
            except Exception:
                pass

        _ib = IB()
        _ib.errorEvent += _on_error

        import trading.config as _cfg
        _host = _cfg.IB_HOST
        _port = _cfg.IB_PORT
        _client_id = _cfg.IB_CLIENT_ID
        _timeout = _cfg.IB_TIMEOUT

        for attempt in range(3):
            try:
                _ib.connect(
                    host=_host,
                    port=_port,
                    clientId=_client_id,
                    timeout=_timeout,
                    readonly=False,
                )
                _ib.reqMarketDataType(1)  # LIVE data
                log.info("Market data type set to LIVE (non-delayed)")

                acct = _ib.accountSummary()
                equity = _extract_account_value(acct, "NetLiquidation")
                log.info("IB connected: host=%s, port=%d, clientId=%d, equity=$%.2f",
                         _host, _port, _client_id, equity)
                return _ib
            except Exception as e:
                log.error("IB connection attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(5)

        raise RuntimeError(
            f"Cannot connect to IB at {_host}:{_port}. "
            f"Ensure TWS or IB Gateway is running with API enabled."
        )


# Keep old name as alias so existing code calling get_client() still works
get_client = get_ib


def disconnect():
    """Cleanly disconnect from IB."""
    global _ib
    with _lock:
        if _ib is not None:
            try:
                _ib.disconnect()
            except Exception:
                pass
            _ib = None
            log.info("IB disconnected")


def is_connected() -> bool:
    """Check if IB connection is alive."""
    return _ib is not None and _ib.isConnected()


def _on_error(reqId, errorCode, errorString, contract):
    """Handle IB error events."""
    non_critical = {2104, 2106, 2158, 2119, 10168}
    if errorCode in non_critical:
        log.debug("IB info [%d]: %s", errorCode, errorString)
    else:
        log.warning("IB error [%d] reqId=%s: %s (contract=%s)",
                     errorCode, reqId, errorString, contract)


def _extract_account_value(summary: list, tag: str) -> float:
    """Extract a value from IB account summary."""
    for item in summary:
        if item.tag == tag:
            try:
                return float(item.value)
            except (ValueError, TypeError):
                pass
    return 0.0


def get_account_equity() -> float:
    """Get net liquidation value from IB."""
    ib = get_ib()
    summary = ib.accountSummary()
    return _extract_account_value(summary, "NetLiquidation")


def get_buying_power() -> float:
    """Get buying power from IB."""
    ib = get_ib()
    summary = ib.accountSummary()
    return _extract_account_value(summary, "AvailableFunds")


def get_midpoint(symbol: str) -> float | None:
    """Get current bid/ask midpoint for a stock."""
    ib = get_ib()
    contract = _resolve_stock(symbol)
    try:
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(2)
        bid = ticker.bid
        ask = ticker.ask
        ib.cancelMktData(contract)

        if bid is not None and ask is not None and bid > 0 and ask > 0:
            if bid != bid or ask != ask:  # NaN check
                return None
            mid = round((bid + ask) / 2, 2)
            log.debug("Quote %s: bid=%.2f ask=%.2f mid=%.2f", symbol, bid, ask, mid)
            return mid
        return None
    except Exception as e:
        log.debug("Failed to get quote for %s: %s", symbol, e)
        return None


def get_position(symbol: str) -> dict | None:
    """Get current position for a stock, or None if flat."""
    ib = get_ib()
    positions = ib.positions()

    for pos in positions:
        if (isinstance(pos.contract, Stock) and
                pos.contract.symbol == symbol):
            qty = int(pos.position)
            if qty == 0:
                continue
            return {
                "symbol": symbol,
                "qty": qty,
                "side": "long" if qty > 0 else "short",
                "avg_entry": float(pos.avgCost),
                "market_value": 0.0,  # IB doesn't provide this directly in positions()
                "unrealized_pl": 0.0,
            }

    return None


def get_all_positions() -> list[dict]:
    """Get all open stock positions."""
    ib = get_ib()
    positions = ib.positions()
    result = []
    for pos in positions:
        if isinstance(pos.contract, Stock) and int(pos.position) != 0:
            qty = int(pos.position)
            result.append({
                "symbol": pos.contract.symbol,
                "qty": qty,
                "side": "long" if qty > 0 else "short",
                "avg_entry": float(pos.avgCost),
                "market_value": 0.0,
                "unrealized_pl": 0.0,
            })
    return result


def submit_order(symbol: str, qty: int, side: str,
                 limit_timeout_sec: int = 5, max_retries: int = 3) -> tuple[str | None, str]:
    """Submit order: try limit at midpoint first, fall back to market.

    Returns (order_id, fill_type) where fill_type is "limit", "market_fallback",
    "market", or "failed".
    """
    if qty <= 0:
        log.warning("Cannot submit order with qty=%d", qty)
        return None, "failed"

    ib = get_ib()
    contract = _resolve_stock(symbol)
    action = "BUY" if side == "buy" else "SELL"

    # Step 1: Try limit at midpoint
    midpoint = get_midpoint(symbol)
    if midpoint is not None and midpoint > 0:
        try:
            limit_order = LimitOrder(action, qty, midpoint)
            trade = ib.placeOrder(contract, limit_order)
            log.info("LIMIT ORDER %s %d %s @ $%.2f -> orderId=%s",
                     action, qty, symbol, midpoint, trade.order.orderId)

            start = time.time()
            while time.time() - start < limit_timeout_sec:
                ib.sleep(0.5)
                if trade.orderStatus.status == "Filled":
                    log.info("LIMIT FILLED %s %d %s @ $%.2f (fill_type=limit)",
                             action, qty, symbol, trade.orderStatus.avgFillPrice)
                    return str(trade.order.orderId), "limit"
                elif trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
                    break

            # Cancel unfilled limit
            try:
                ib.cancelOrder(trade.order)
                ib.sleep(1)
            except Exception:
                if trade.orderStatus.status == "Filled":
                    return str(trade.order.orderId), "limit"

            log.info("Limit timeout — falling back to market order")

        except Exception as e:
            log.warning("Limit order failed: %s — falling back to market", e)

    # Step 2: Market fallback
    fill_type = "market_fallback" if midpoint is not None else "market"
    for attempt in range(max_retries):
        try:
            market_order = MarketOrder(action, qty)
            trade = ib.placeOrder(contract, market_order)
            log.info("MARKET ORDER %s %d %s -> orderId=%s (fill_type=%s)",
                     action, qty, symbol, trade.order.orderId, fill_type)

            start = time.time()
            while time.time() - start < 10:
                ib.sleep(0.5)
                if trade.orderStatus.status == "Filled":
                    return str(trade.order.orderId), fill_type
                elif trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
                    break

            if trade.orderStatus.status in ("Submitted", "PreSubmitted"):
                return str(trade.order.orderId), fill_type

        except Exception as e:
            log.error("Market order failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(1)

    log.error("Order FAILED after all attempts: %s %d %s", side, qty, symbol)
    return None, "failed"


def submit_market_order(symbol: str, qty: int, side: str,
                        max_retries: int = 3) -> str | None:
    """Submit order using limit-first-then-market flow. Returns order ID."""
    order_id, fill_type = submit_order(symbol, qty, side,
                                       limit_timeout_sec=5,
                                       max_retries=max_retries)
    return order_id


def wait_for_fill(order_id: str, timeout_sec: int = 30) -> dict | None:
    """Wait for an order to fill. Returns fill info or None on timeout."""
    ib = get_ib()
    start = time.time()

    while time.time() - start < timeout_sec:
        ib.sleep(0.5)
        for trade in ib.trades():
            if str(trade.order.orderId) == order_id:
                if trade.orderStatus.status == "Filled":
                    return {
                        "id": order_id,
                        "status": "filled",
                        "filled_qty": int(trade.orderStatus.filled),
                        "filled_avg_price": float(trade.orderStatus.avgFillPrice),
                    }
                elif trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
                    return {
                        "id": order_id,
                        "status": trade.orderStatus.status.lower(),
                    }

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
    """Emergency: close all stock positions."""
    log.warning("CLOSING ALL STOCK POSITIONS")
    ib = get_ib()
    positions = ib.positions()
    for pos in positions:
        if isinstance(pos.contract, Stock) and int(pos.position) != 0:
            qty = abs(int(pos.position))
            action = "SELL" if pos.position > 0 else "BUY"
            try:
                order = MarketOrder(action, qty)
                ib.placeOrder(pos.contract, order)
                log.info("Closing %s %d %s", action, qty, pos.contract.symbol)
            except Exception as e:
                log.error("Error closing position %s: %s", pos.contract.symbol, e)

    try:
        ib.reqGlobalCancel()
    except Exception:
        pass


def cancel_all_orders():
    """Cancel all open orders."""
    try:
        ib = get_ib()
        ib.reqGlobalCancel()
        log.info("All open orders cancelled")
    except Exception as e:
        log.error("Error cancelling orders: %s", e)


def is_market_open() -> bool:
    """Check if the US stock market is currently open (09:30-16:00 ET weekdays)."""
    now = datetime.now(ET)
    weekday = now.weekday()
    if weekday >= 5:  # Weekend
        return False
    hour = now.hour
    minute = now.minute
    mins = hour * 60 + minute
    return 9 * 60 + 30 <= mins < 16 * 60  # 09:30-16:00
