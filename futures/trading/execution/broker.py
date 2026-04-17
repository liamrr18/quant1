"""Futures broker interface via Interactive Brokers.

Connects to IB TWS or IB Gateway for real MES/MNQ micro futures trading.
Handles contract resolution, front-month rollover, order submission,
position tracking, and account data via ib_insync.

Contract rollover: always trades front month, rolls to next quarterly
when current contract is within 3 days of expiration.
"""

import logging
import time
import threading
from datetime import datetime, timedelta

import pytz
from ib_insync import (
    IB, Future, MarketOrder, LimitOrder, Trade as IBTrade,
    util,
)

from trading.config import IB_HOST, IB_PORT, IB_CLIENT_ID, IB_TIMEOUT
from trading.data.contracts import CONTRACTS

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

# Quarterly expiry months
ROLL_MONTHS = [3, 6, 9, 12]

_ib: IB | None = None
_lock = threading.Lock()

# Cache resolved contracts to avoid repeated lookups
_contract_cache: dict[str, Future] = {}


def _front_month_expiry(now: datetime = None) -> str:
    """Get the front-month quarterly expiry in YYYYMM format.

    Rolls to next quarter if current front month is within 3 days of expiry.
    CME futures expire on the 3rd Friday of the contract month.
    """
    if now is None:
        now = datetime.now(ET)

    year = now.year
    month = now.month

    # Find current or next quarterly month
    for m in ROLL_MONTHS:
        if m >= month:
            contract_month = m
            contract_year = year
            break
    else:
        contract_month = ROLL_MONTHS[0]
        contract_year = year + 1

    # Find 3rd Friday of contract month
    import calendar
    cal = calendar.monthcalendar(contract_year, contract_month)
    # Find all Fridays (index 4)
    fridays = [week[4] for week in cal if week[4] != 0]
    third_friday = fridays[2]
    expiry_date = datetime(contract_year, contract_month, third_friday, tzinfo=ET)

    # If within 3 days of expiry, roll to next quarter
    if (expiry_date - now).days <= 3:
        idx = ROLL_MONTHS.index(contract_month)
        if idx + 1 < len(ROLL_MONTHS):
            contract_month = ROLL_MONTHS[idx + 1]
        else:
            contract_month = ROLL_MONTHS[0]
            contract_year += 1

    return f"{contract_year}{contract_month:02d}"


_EXCHANGE_MAP = {
    "MGC": "COMEX",
    "MCL": "NYMEX",
}


def _resolve_ib_contract(symbol: str) -> Future:
    """Resolve a futures symbol to a qualified IB contract.

    Uses front-month expiry with automatic rollover.
    """
    if symbol in _contract_cache:
        cached = _contract_cache[symbol]
        expiry_yyyymm = _front_month_expiry()
        if cached.lastTradeDateOrContractMonth[:6] == expiry_yyyymm:
            return cached

    exchange = _EXCHANGE_MAP.get(symbol, "CME")
    expiry = _front_month_expiry()
    contract = Future(symbol=symbol, exchange=exchange, lastTradeDateOrContractMonth=expiry)

    ib = get_ib()
    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            resolved = qualified[0]
            _contract_cache[symbol] = resolved
            log.info("Resolved %s -> %s (expiry: %s, conId: %s)",
                     symbol, resolved.localSymbol, resolved.lastTradeDateOrContractMonth,
                     resolved.conId)
            return resolved
    except Exception as e:
        log.error("Failed to qualify %s contract: %s", symbol, e)

    # Fallback: try without specifying expiry and let IB pick front month
    contract = Future(symbol=symbol, exchange=exchange)
    try:
        details = ib.reqContractDetails(contract)
        if details:
            # Pick the nearest expiry
            details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
            resolved = details[0].contract
            ib.qualifyContracts(resolved)
            _contract_cache[symbol] = resolved
            log.info("Resolved %s (fallback) -> %s (expiry: %s)",
                     symbol, resolved.localSymbol, resolved.lastTradeDateOrContractMonth)
            return resolved
    except Exception as e:
        log.error("Fallback contract resolution failed for %s: %s", symbol, e)

    raise RuntimeError(f"Cannot resolve IB contract for {symbol}")


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

        # Re-read config at call time so runtime overrides (e.g. overnight
        # clientId=2) take effect even after this module was imported.
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
                # Request live (non-delayed) market data
                # Type 1 = live, 3 = delayed, 4 = frozen delayed
                _ib.reqMarketDataType(1)
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


def _on_error(reqId, errorCode, errorString, contract):
    """Handle IB error events."""
    # Filter out non-critical messages
    non_critical = {2104, 2106, 2158, 2119}  # Market data farm messages
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


def is_connected() -> bool:
    """Check if IB connection is alive."""
    return _ib is not None and _ib.isConnected()


def get_account_equity() -> float:
    """Get net liquidation value from IB."""
    ib = get_ib()
    summary = ib.accountSummary()
    return _extract_account_value(summary, "NetLiquidation")


def get_available_margin() -> float:
    """Get available funds / buying power from IB."""
    ib = get_ib()
    summary = ib.accountSummary()
    return _extract_account_value(summary, "AvailableFunds")


def get_buying_power() -> float:
    """Get buying power from IB."""
    return get_available_margin()


def get_account_info() -> dict:
    """Get full account summary."""
    ib = get_ib()
    summary = ib.accountSummary()
    return {
        "equity": _extract_account_value(summary, "NetLiquidation"),
        "available_margin": _extract_account_value(summary, "AvailableFunds"),
        "initial_margin": _extract_account_value(summary, "InitMarginReq"),
        "maint_margin": _extract_account_value(summary, "MaintMarginReq"),
        "unrealized_pnl": _extract_account_value(summary, "UnrealizedPnL"),
        "realized_pnl": _extract_account_value(summary, "RealizedPnL"),
    }


def get_contract_month(symbol: str) -> str:
    """Get the current front-month contract identifier."""
    contract = _resolve_ib_contract(symbol)
    return contract.lastTradeDateOrContractMonth


def get_midpoint(symbol: str) -> float | None:
    """Get current bid/ask midpoint for a futures contract."""
    ib = get_ib()
    contract = _resolve_ib_contract(symbol)
    try:
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(2)  # Wait for data
        bid = ticker.bid
        ask = ticker.ask
        ib.cancelMktData(contract)

        if bid is not None and ask is not None and bid > 0 and ask > 0:
            # These are sometimes nan
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
    """Get current position for a futures symbol."""
    ib = get_ib()
    positions = ib.positions()

    for pos in positions:
        if (pos.contract.symbol == symbol and
                isinstance(pos.contract, Future)):
            qty = int(pos.position)
            if qty == 0:
                continue
            return {
                "symbol": symbol,
                "qty": qty,
                "side": "long" if qty > 0 else "short",
                "avg_entry": float(pos.avgCost) / CONTRACTS[symbol].point_value if symbol in CONTRACTS else float(pos.avgCost),
                "unrealized_pl": float(pos.avgCost),  # IB reports cost, not unrealized PnL here
            }

    return None


def get_all_positions() -> list[dict]:
    """Get all open futures positions."""
    ib = get_ib()
    positions = ib.positions()
    result = []
    for pos in positions:
        if isinstance(pos.contract, Future) and int(pos.position) != 0:
            qty = int(pos.position)
            result.append({
                "symbol": pos.contract.symbol,
                "qty": qty,
                "side": "long" if qty > 0 else "short",
                "avg_entry": float(pos.avgCost),
            })
    return result


def submit_order(symbol: str, qty: int, side: str,
                 limit_timeout_sec: int = 5, max_retries: int = 3) -> tuple[str | None, str]:
    """Submit an order with limit-then-market fallback.

    Tries a limit order at midpoint first. If not filled within
    limit_timeout_sec, cancels and falls back to market order.

    Returns (order_id, fill_type) where fill_type is 'limit', 'market',
    'market_fallback', or 'failed'.
    """
    if qty <= 0:
        log.warning("Cannot submit order with qty=%d", qty)
        return None, "failed"

    ib = get_ib()
    contract = _resolve_ib_contract(symbol)
    action = "BUY" if side == "buy" else "SELL"

    # Try limit at midpoint first
    midpoint = get_midpoint(symbol)
    if midpoint is not None and midpoint > 0:
        try:
            limit_order = LimitOrder(action, qty, midpoint)
            trade = ib.placeOrder(contract, limit_order)
            log.info("LIMIT ORDER %s %d %s @ %.2f -> orderId=%s",
                     action, qty, symbol, midpoint, trade.order.orderId)

            # Wait for fill
            start = time.time()
            while time.time() - start < limit_timeout_sec:
                ib.sleep(0.5)
                if trade.orderStatus.status == "Filled":
                    fill_price = trade.orderStatus.avgFillPrice
                    log.info("LIMIT FILLED %s %d %s @ %.2f",
                             action, qty, symbol, fill_price)
                    return str(trade.order.orderId), "limit"
                elif trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
                    break

            # Cancel unfilled limit
            try:
                ib.cancelOrder(trade.order)
                ib.sleep(1)
            except Exception:
                # Check if it filled during cancellation
                if trade.orderStatus.status == "Filled":
                    return str(trade.order.orderId), "limit"

        except Exception as e:
            log.warning("Limit order failed: %s", e)

    # Market fallback
    fill_type = "market_fallback" if midpoint is not None else "market"
    for attempt in range(max_retries):
        try:
            market_order = MarketOrder(action, qty)
            trade = ib.placeOrder(contract, market_order)
            log.info("MARKET ORDER %s %d %s -> orderId=%s (fill_type=%s)",
                     action, qty, symbol, trade.order.orderId, fill_type)

            # Wait briefly for fill confirmation
            start = time.time()
            while time.time() - start < 10:
                ib.sleep(0.5)
                if trade.orderStatus.status == "Filled":
                    fill_price = trade.orderStatus.avgFillPrice
                    log.info("MARKET FILLED %s %d %s @ %.2f",
                             action, qty, symbol, fill_price)
                    return str(trade.order.orderId), fill_type
                elif trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
                    log.warning("Market order %s was %s", trade.order.orderId,
                                trade.orderStatus.status)
                    break

            # If still pending, return the order ID anyway
            if trade.orderStatus.status in ("Submitted", "PreSubmitted"):
                return str(trade.order.orderId), fill_type

        except Exception as e:
            log.error("Market order failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(1)

    return None, "failed"


def submit_market_order(symbol: str, qty: int, side: str,
                        max_retries: int = 3) -> str | None:
    """Submit a market order directly (no limit attempt)."""
    if qty <= 0:
        return None

    ib = get_ib()
    contract = _resolve_ib_contract(symbol)
    action = "BUY" if side == "buy" else "SELL"

    for attempt in range(max_retries):
        try:
            order = MarketOrder(action, qty)
            trade = ib.placeOrder(contract, order)
            log.info("MARKET ORDER %s %d %s -> orderId=%s",
                     action, qty, symbol, trade.order.orderId)
            return str(trade.order.orderId)
        except Exception as e:
            log.error("Market order failed (attempt %d/%d): %s",
                      attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def wait_for_fill(order_id: str, timeout_sec: int = 30) -> dict | None:
    """Wait for an order to fill."""
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
                elif trade.orderStatus.status in ("Cancelled", "ApiCancelled",
                                                    "Inactive"):
                    return {
                        "id": order_id,
                        "status": trade.orderStatus.status.lower(),
                    }

    log.warning("Order %s timed out after %ds", order_id, timeout_sec)
    return None


def close_position(symbol: str) -> str | None:
    """Close an existing position by symbol."""
    pos = get_position(symbol)
    if pos is None:
        return None

    qty = abs(pos["qty"])
    side = "sell" if pos["side"] == "long" else "buy"
    return submit_market_order(symbol, qty, side)


def close_all_positions():
    """Close all open futures positions."""
    log.warning("CLOSING ALL FUTURES POSITIONS")
    ib = get_ib()
    positions = ib.positions()
    for pos in positions:
        if isinstance(pos.contract, Future) and int(pos.position) != 0:
            qty = abs(int(pos.position))
            action = "SELL" if pos.position > 0 else "BUY"
            try:
                order = MarketOrder(action, qty)
                ib.placeOrder(pos.contract, order)
                log.info("Closing %s %d %s", action, qty, pos.contract.symbol)
            except Exception as e:
                log.error("Error closing position %s: %s", pos.contract.symbol, e)


def cancel_all_orders():
    """Cancel all open orders."""
    ib = get_ib()
    try:
        ib.reqGlobalCancel()
        log.info("All open orders cancelled")
    except Exception as e:
        log.error("Error cancelling orders: %s", e)


def is_market_open() -> bool:
    """Check if futures market is currently open.

    CME E-mini/Micro futures trade Sunday 6pm - Friday 5pm ET
    with a daily maintenance break 5pm-6pm ET Monday-Thursday.
    """
    now = datetime.now(ET)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour
    minute = now.minute

    # Saturday: closed all day
    if weekday == 5:
        return False

    # Sunday: opens at 6pm ET
    if weekday == 6:
        return hour >= 18

    # Friday: closes at 5pm ET
    if weekday == 4:
        return hour < 17

    # Mon-Thu: daily maintenance break 5pm-6pm ET
    if hour == 17:
        return False

    return True
