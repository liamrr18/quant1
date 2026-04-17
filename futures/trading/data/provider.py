"""Data provider for futures backtesting and live trading.

PROXY DATA APPROACH:
  MES and MNQ micro futures track the S&P 500 and Nasdaq 100 indices
  respectively. During regular trading hours (09:30-16:00 ET), SPY and
  QQQ ETFs track these same indices tick-for-tick.

  Since Alpaca does not provide historical minute-bar data for futures
  contracts, we use SPY data as proxy for MES and QQQ data as proxy
  for MNQ. The backtest engine handles the conversion from ETF price
  movements to futures P&L using the correct point values and multipliers.

  This proxy is highly accurate for our use case because:
  1. ORB strategy trades only during RTH (09:45-15:30 ET)
  2. We hold intraday only (no overnight basis risk)
  3. The correlation between ETF and futures during RTH is >0.999
  4. Typical basis (futures premium/discount) is <0.05% intraday

DIVERGENCE RISKS (documented, not applicable to our strategy):
  - Extended hours: futures trade Sun 6pm - Fri 5pm, ETFs don't
  - Quarterly roll: 3rd Friday of Mar/Jun/Sep/Dec
  - Extreme volatility: futures may temporarily diverge during halts
  - Dividends: minor, ETFs adjust while futures don't (small effect)
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trading.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, CACHE_DIR, EQUITY_CACHE_DIR
from trading.data.contracts import CONTRACTS

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

_client = None

PROXY_MAP = {
    "MES": "SPY",
    "MNQ": "QQQ",
}


def get_client() -> StockHistoricalDataClient:
    global _client
    if _client is None:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
        _client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _client


def _resolve_proxy(symbol: str) -> str:
    """Map futures symbol to its ETF proxy."""
    return PROXY_MAP.get(symbol, symbol)


def fetch_minute_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch 1-minute bars from Alpaca using ETF proxy."""
    proxy = _resolve_proxy(symbol)
    log.info("Fetching %s minute bars (proxy=%s): %s to %s",
             symbol, proxy, start.date(), end.date())
    client = get_client()

    all_bars = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=7), end)
        req = StockBarsRequest(
            symbol_or_symbols=proxy,
            timeframe=TimeFrame.Minute,
            start=chunk_start,
            end=chunk_end,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if not df.empty:
            all_bars.append(df)
        chunk_start = chunk_end

    if not all_bars:
        raise RuntimeError(f"No data returned for {symbol} (proxy={proxy})")

    result = pd.concat(all_bars)
    result = result.reset_index()

    result = result.rename(columns={"timestamp": "dt"})
    if result["dt"].dt.tz is not None:
        result["dt"] = result["dt"].dt.tz_convert(ET)
    else:
        result["dt"] = result["dt"].dt.tz_localize("UTC").dt.tz_convert(ET)

    # Keep only regular trading hours
    result["time"] = result["dt"].dt.strftime("%H:%M")
    result = result[(result["time"] >= "09:30") & (result["time"] < "16:00")]
    result = result.drop(columns=["time"])
    result = result.sort_values("dt").reset_index(drop=True)

    if "symbol" in result.columns:
        result = result.drop(columns=["symbol"])

    log.info("Got %d bars for %s", len(result), symbol)
    return result


def save_cache(df: pd.DataFrame, symbol: str, label: str):
    """Save DataFrame to CSV cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_{label}.csv")
    df.to_csv(path, index=False)
    log.info("Cached %d rows to %s", len(df), path)


def load_cache(symbol: str, label: str) -> pd.DataFrame | None:
    """Load cached DataFrame. Checks futures cache first, then equity cache."""
    # Check futures cache
    path = os.path.join(CACHE_DIR, f"{symbol}_{label}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)
        log.info("Loaded %d rows from futures cache: %s", len(df), path)
        return df

    # Check equity cache using proxy symbol
    proxy = _resolve_proxy(symbol)
    equity_path = os.path.join(EQUITY_CACHE_DIR, f"{proxy}_{label}.csv")
    if os.path.exists(equity_path):
        df = pd.read_csv(equity_path)
        df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)
        log.info("Loaded %d rows from equity cache (proxy %s): %s",
                 len(df), proxy, equity_path)
        return df

    return None


def get_minute_bars(symbol: str, start: datetime, end: datetime,
                    use_cache: bool = True) -> pd.DataFrame:
    """Get minute bars, using cache if available."""
    label = f"1min_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"

    if use_cache:
        cached = load_cache(symbol, label)
        if cached is not None:
            return cached

    df = fetch_minute_bars(symbol, start, end)
    if use_cache:
        save_cache(df, symbol, label)
    return df
