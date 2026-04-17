"""Feature engineering for intraday trading strategies.

Identical to the equity version - all features are computed on price
data which is the same whether we're using SPY/QQQ directly or as
a proxy for MES/MNQ futures.
"""

import numpy as np
import pandas as pd


def add_session_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["dt"].dt.date
    df["time"] = df["dt"].dt.strftime("%H:%M")
    df["minute_of_day"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_col = "vwap" if "vwap" in df.columns else "close"
    df["_vp"] = df[price_col] * df["volume"]
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["cum_vp"] = df.groupby("date")["_vp"].cumsum()
    df["intraday_vwap"] = df["cum_vp"] / df["cum_vol"].replace(0, np.nan)
    df = df.drop(columns=["cum_vol", "cum_vp", "_vp"])
    return df


def add_vwap_deviation(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    diff = df["close"] - df["intraday_vwap"]
    rolling_std = diff.rolling(lookback, min_periods=5).std()
    df["vwap_dev"] = diff / rolling_std.replace(0, np.nan)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_ema(df: pd.DataFrame, span: int, col: str = "close") -> pd.DataFrame:
    df = df.copy()
    df[f"ema_{span}"] = df[col].ewm(span=span, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["bb_mid"] = df["close"].rolling(period, min_periods=5).mean()
    bb_std = df["close"].rolling(period, min_periods=5).std()
    df["bb_upper"] = df["bb_mid"] + num_std * bb_std
    df["bb_lower"] = df["bb_mid"] - num_std * bb_std
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_prev = abs(df["high"] - df["close"].shift(1))
    low_prev = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period, min_periods=5).mean()
    return df


def add_atr_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_pct"] = df["atr"] / df["close"] * 100
    daily_atr = df.groupby("date")["atr_pct"].last()
    daily_rank = daily_atr.rolling(20, min_periods=5).rank(pct=True) * 100
    df["atr_percentile"] = df["date"].map(daily_rank)
    return df


def add_relative_volume(df: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    df = df.copy()
    avg_vol = df.groupby("time")["volume"].transform(
        lambda x: x.rolling(lookback_days * 1, min_periods=1).mean().shift(1)
    )
    df["rel_volume"] = df["volume"] / avg_vol.replace(0, np.nan)
    df["rel_volume"] = df["rel_volume"].fillna(1.0)
    return df


def add_opening_range(df: pd.DataFrame, range_minutes: int = 15) -> pd.DataFrame:
    df = df.copy()
    df["or_high"] = np.nan
    df["or_low"] = np.nan
    or_end_minute = 9 * 60 + 30 + range_minutes

    for date, group in df.groupby("date"):
        or_bars = group[group["minute_of_day"] < or_end_minute]
        if len(or_bars) == 0:
            continue
        or_h = or_bars["high"].max()
        or_l = or_bars["low"].min()
        mask = df["date"] == date
        df.loc[mask, "or_high"] = or_h
        df.loc[mask, "or_low"] = or_l

    return df


def add_prev_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    daily_close = df.groupby("date")["close"].last()
    dates = sorted(df["date"].unique())
    prev_map = {}
    for i, d in enumerate(dates):
        if i > 0:
            prev_map[d] = daily_close[dates[i - 1]]
    df["prev_close"] = df["date"].map(prev_map)
    df["gap_pct"] = (df.groupby("date")["open"].transform("first") / df["prev_close"] - 1) * 100
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = add_session_info(df)
    df = add_vwap(df)
    df = add_vwap_deviation(df)
    df = add_rsi(df)
    df = add_ema(df, 8)
    df = add_ema(df, 21)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_atr_pct(df)
    df = add_relative_volume(df)
    df = add_opening_range(df)
    df = add_prev_close(df)
    return df
