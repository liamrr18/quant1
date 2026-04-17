"""Live data feed from Interactive Brokers for futures trading.

Provides real-time minute bars for MES and MNQ during live trading.
Falls back to SPY/QQQ Alpaca data if IB bars are unavailable.

The backtesting data provider (provider.py) is NOT modified — it
continues to use cached CSVs from Alpaca.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytz
from ib_insync import Future, util

from trading.execution.broker import get_ib, _resolve_ib_contract

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


def fetch_ib_bars(symbol: str, lookback_minutes: int = 200,
                  use_rth: bool = True) -> pd.DataFrame | None:
    """Fetch recent minute bars from IB for a futures symbol.

    Returns a DataFrame with columns: dt, open, high, low, close, volume
    or None if IB data is unavailable.

    Args:
        use_rth: If True, only regular trading hours bars (9:30-16:00).
                 Set False for overnight/extended-hours strategies.
    """
    try:
        ib = get_ib()
        contract = _resolve_ib_contract(symbol)

        # For ~200 1-minute bars, request 1 day of data
        duration = "1 D" if lookback_minutes <= 390 else "2 D"

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=use_rth,
            formatDate=1,
        )

        if not bars:
            log.warning("No IB bars returned for %s", symbol)
            return None

        df = util.df(bars)
        if df is None or df.empty:
            return None

        # Rename columns to match our expected format
        df = df.rename(columns={"date": "dt"})

        # Ensure timezone-aware datetime
        if not hasattr(df["dt"].dtype, "tz") or df["dt"].dt.tz is None:
            df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(ET)
        else:
            df["dt"] = pd.to_datetime(df["dt"]).dt.tz_convert(ET)

        # Filter to regular trading hours only if use_rth
        if use_rth:
            times = df["dt"].dt.strftime("%H:%M")
            df = df[(times >= "09:30") & (times < "16:00")]
        df = df.sort_values("dt").reset_index(drop=True)

        # Ensure required columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                log.warning("Missing column %s in IB bars for %s", col, symbol)
                return None

        df = df[["dt", "open", "high", "low", "close", "volume"]]
        log.info("Got %d IB bars for %s", len(df), symbol)
        return df

    except Exception as e:
        log.warning("IB bar fetch failed for %s: %s", symbol, e)
        return None


def fetch_live_bars(symbol: str, lookback_minutes: int = 200,
                    use_rth: bool = True) -> pd.DataFrame:
    """Fetch live minute bars, trying IB first then falling back to Alpaca proxy.

    Args:
        use_rth: Pass False for overnight/extended-hours strategies.
    """
    df = fetch_ib_bars(symbol, lookback_minutes, use_rth=use_rth)
    if df is not None and len(df) >= 30:
        return df

    # Fall back to Alpaca SPY/QQQ proxy data
    log.warning("IB bars unavailable for %s, falling back to Alpaca proxy", symbol)
    return _fetch_alpaca_proxy_bars(symbol, lookback_minutes)


def _fetch_alpaca_proxy_bars(symbol: str, lookback_minutes: int = 200) -> pd.DataFrame:
    """Fetch bars from Alpaca using ETF proxy (SPY/QQQ)."""
    from trading.data.contracts import CONTRACTS

    proxy_map = {"MES": "SPY", "MNQ": "QQQ"}
    proxy = proxy_map.get(symbol, symbol)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        from trading.config import ALPACA_API_KEY, ALPACA_SECRET_KEY

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        end = datetime.now(pytz.UTC)
        start = end - timedelta(minutes=lookback_minutes + 60)

        req = StockBarsRequest(
            symbol_or_symbols=proxy,
            timeframe=TimeFrame.Minute,
            start=start, end=end,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        df = bars.df.reset_index()
        df = df.rename(columns={"timestamp": "dt"})
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        if df["dt"].dt.tz is not None:
            df["dt"] = df["dt"].dt.tz_convert(ET)
        else:
            df["dt"] = df["dt"].dt.tz_localize("UTC").dt.tz_convert(ET)

        times = df["dt"].dt.strftime("%H:%M")
        df = df[(times >= "09:30") & (times < "16:00")]
        df = df.sort_values("dt").reset_index(drop=True)

        log.info("Alpaca proxy fallback: got %d bars for %s (proxy=%s)", len(df), symbol, proxy)
        return df

    except Exception as e:
        log.error("Alpaca proxy fallback also failed for %s: %s", symbol, e)
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])
