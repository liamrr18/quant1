
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from massive import RESTClient


DATA_DIR = Path("data/gap_edge")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_client() -> RESTClient:
    load_dotenv()
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY not found in .env or environment.")
    return RESTClient(api_key=api_key)


def fetch_minute_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    client = get_client()

    bars = list(
        client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="minute",
            from_=start,
            to=end,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
    )

    rows = []
    for b in bars:
        rows.append(
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "vwap": getattr(b, "vwap", None),
                "transactions": getattr(b, "transactions", None),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No bars returned from Massive.")
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dt"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    out["session_date"] = out["dt"].dt.strftime("%Y-%m-%d")
    out["time"] = out["dt"].dt.strftime("%H:%M")
    out["dow"] = out["dt"].dt.day_name()
    return out


def safe_sign(x: float) -> int:
    if pd.isna(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def gap_bucket(abs_gap_pct: float) -> str:
    if pd.isna(abs_gap_pct):
        return "unknown"
    if abs_gap_pct < 0.30:
        return "<0.30%"
    if abs_gap_pct < 0.60:
        return "0.30-0.60%"
    if abs_gap_pct < 1.00:
        return "0.60-1.00%"
    return ">=1.00%"
