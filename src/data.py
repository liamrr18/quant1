"""Fetch market data from Massive (Polygon) API."""

from datetime import datetime, timedelta

import pandas as pd
from massive import RESTClient

from src.config import MASSIVE_API_KEY, SYMBOL


def get_client() -> RESTClient:
    if not MASSIVE_API_KEY:
        raise RuntimeError("MASSIVE_API_KEY not set in .env")
    return RESTClient(api_key=MASSIVE_API_KEY)


def fetch_bars(symbol: str, start: str, end: str, timespan: str = "minute") -> pd.DataFrame:
    """Fetch OHLCV bars from Massive API.

    Args:
        symbol: Ticker symbol (e.g. "SPY")
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        timespan: Bar size ("minute", "hour", "day")

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, vwap
    """
    client = get_client()
    bars = list(
        client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan=timespan,
            from_=start,
            to=end,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
    )

    if not bars:
        raise RuntimeError(f"No bars returned for {symbol} from {start} to {end}")

    rows = [
        {
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "vwap": getattr(b, "vwap", None),
        }
        for b in bars
    ]

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    return df


def fetch_daily_bars(symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch daily bars for regime filter calculations."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")
    return fetch_bars(symbol, start, end, timespan="day")


def fetch_recent_minutes(symbol: str, days: int = 3) -> pd.DataFrame:
    """Fetch recent minute bars covering today + prior days for gap calculation."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days + 2)).strftime("%Y-%m-%d")
    return fetch_bars(symbol, start, end, timespan="minute")


def get_previous_close(minute_df: pd.DataFrame, today_str: str) -> float | None:
    """Get yesterday's 15:59 close from minute bars."""
    df = minute_df.copy()
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["time"] = df["dt"].dt.strftime("%H:%M")

    prior_dates = sorted([d for d in df["date"].unique() if d < today_str])
    if not prior_dates:
        return None

    yesterday = prior_dates[-1]
    close_bar = df[(df["date"] == yesterday) & (df["time"] == "15:59")]
    if close_bar.empty:
        return None
    return float(close_bar.iloc[0]["close"])


def get_bar_at_time(minute_df: pd.DataFrame, date_str: str, time_str: str) -> dict | None:
    """Get a specific bar by date and time."""
    df = minute_df.copy()
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["time"] = df["dt"].dt.strftime("%H:%M")

    bar = df[(df["date"] == date_str) & (df["time"] == time_str)]
    if bar.empty:
        return None
    return bar.iloc[0].to_dict()
