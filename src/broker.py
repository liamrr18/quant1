"""Alpaca broker interface for order placement and position management."""

import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

log = logging.getLogger(__name__)

_client: TradingClient | None = None


def get_client() -> TradingClient:
    global _client
    if _client is None:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
        _client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=ALPACA_PAPER,
        )
        log.info("Alpaca client initialized (paper=%s)", ALPACA_PAPER)
    return _client


def get_account_equity() -> float:
    account = get_client().get_account()
    return float(account.equity)


def get_buying_power() -> float:
    account = get_client().get_account()
    return float(account.buying_power)


def get_position(symbol: str) -> dict | None:
    """Get current position for a symbol, or None if flat."""
    try:
        pos = get_client().get_open_position(symbol)
        return {
            "symbol": pos.symbol,
            "qty": int(pos.qty),
            "side": pos.side.value,
            "avg_entry": float(pos.avg_entry_price),
            "market_value": float(pos.market_value),
            "unrealized_pl": float(pos.unrealized_pl),
            "unrealized_plpc": float(pos.unrealized_plpc),
        }
    except Exception:
        return None


def submit_short(symbol: str, qty: int) -> str:
    """Submit a market sell-short order. Returns the order ID."""
    log.info("Submitting SHORT %d shares of %s", qty, symbol)
    order = get_client().submit_order(
        MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
    )
    log.info("Order submitted: id=%s status=%s", order.id, order.status)
    return str(order.id)


def submit_cover(symbol: str, qty: int) -> str:
    """Submit a market buy-to-cover order. Returns the order ID."""
    log.info("Submitting COVER %d shares of %s", qty, symbol)
    order = get_client().submit_order(
        MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    )
    log.info("Order submitted: id=%s status=%s", order.id, order.status)
    return str(order.id)


def close_position(symbol: str) -> str | None:
    """Close any open position in the symbol. Returns order ID or None if flat."""
    pos = get_position(symbol)
    if pos is None:
        log.info("No position in %s to close", symbol)
        return None

    qty = abs(pos["qty"])
    if pos["side"] == "short":
        return submit_cover(symbol, qty)
    else:
        # Long position - sell it
        log.info("Closing LONG %d shares of %s", qty, symbol)
        order = get_client().submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        return str(order.id)
