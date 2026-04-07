
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, add_time_columns, fetch_minute_bars, gap_bucket, safe_sign


PREMARKET_START = "04:00"
PREMARKET_END = "09:29"
OPEN_BAR = "09:30"
BAR_5M = "09:34"
BAR_15M = "09:44"
BAR_30M = "09:59"
BAR_60M = "10:29"
BAR_NOON = "11:59"
CLOSE_BAR = "15:59"


def get_close_before_open(pre_df: pd.DataFrame) -> float | None:
    bar = pre_df.loc[pre_df["time"] == PREMARKET_END]
    if bar.empty:
        return None
    return float(bar.iloc[0]["close"])


def get_bar_value(df: pd.DataFrame, time_str: str, col: str) -> float | None:
    bar = df.loc[df["time"] == time_str]
    if bar.empty:
        return None
    return float(bar.iloc[0][col])


def get_slice(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    return df[(df["time"] >= start_time) & (df["time"] <= end_time)].copy()


def gap_fill_stats(intraday_df: pd.DataFrame, prev_close: float, open_px: float, gap_sign: int):
    if gap_sign == 0 or intraday_df.empty:
        return {
            "gap_fill_fraction_by_10": np.nan,
            "gap_fill_fraction_by_noon": np.nan,
            "gap_fill_fraction_by_close": np.nan,
            "gap_fills_by_10": np.nan,
            "gap_fills_by_noon": np.nan,
            "gap_fills_by_close": np.nan,
        }

    def one_period(end_time: str):
        window = intraday_df[intraday_df["time"] <= end_time].copy()
        if window.empty:
            return np.nan, np.nan

        if gap_sign > 0:
            gap_size = max(open_px - prev_close, 1e-9)
            best_retrace = max(open_px - float(window["low"].min()), 0.0)
            fill_frac = float(np.clip(best_retrace / gap_size, 0.0, 1.0))
            filled = int(float(window["low"].min()) <= prev_close)
        else:
            gap_size = max(prev_close - open_px, 1e-9)
            best_retrace = max(float(window["high"].max()) - open_px, 0.0)
            fill_frac = float(np.clip(best_retrace / gap_size, 0.0, 1.0))
            filled = int(float(window["high"].max()) >= prev_close)
        return fill_frac, filled

    frac10, fill10 = one_period("09:59")
    frac_noon, fill_noon = one_period("11:59")
    frac_close, fill_close = one_period("15:59")

    return {
        "gap_fill_fraction_by_10": frac10,
        "gap_fill_fraction_by_noon": frac_noon,
        "gap_fill_fraction_by_close": frac_close,
        "gap_fills_by_10": fill10,
        "gap_fills_by_noon": fill_noon,
        "gap_fills_by_close": fill_close,
    }


def build_daily_dataset(minute_df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_columns(minute_df)
    sessions = sorted(df["session_date"].unique())

    rows = []
    for i in range(1, len(sessions)):
        prev_day = sessions[i - 1]
        day = sessions[i]

        prev_df = df[df["session_date"] == prev_day].copy()
        day_df = df[df["session_date"] == day].copy()

        prev_close = get_bar_value(prev_df, CLOSE_BAR, "close")
        open_px = get_bar_value(day_df, OPEN_BAR, "open")
        close_px = get_bar_value(day_df, CLOSE_BAR, "close")

        if prev_close is None or open_px is None or close_px is None:
            continue

        pm_df = get_slice(day_df, PREMARKET_START, PREMARKET_END)
        core_df = get_slice(day_df, OPEN_BAR, CLOSE_BAR)

        if core_df.empty:
            continue

        pm_last = get_close_before_open(pm_df) if not pm_df.empty else np.nan
        pm_high = float(pm_df["high"].max()) if not pm_df.empty else np.nan
        pm_low = float(pm_df["low"].min()) if not pm_df.empty else np.nan
        pm_volume = float(pm_df["volume"].sum()) if not pm_df.empty else 0.0
        pm_mid = (pm_high + pm_low) / 2 if pd.notna(pm_high) and pd.notna(pm_low) else np.nan

        ret_5m_close = get_bar_value(core_df, BAR_5M, "close")
        ret_15m_close = get_bar_value(core_df, BAR_15M, "close")
        ret_30m_close = get_bar_value(core_df, BAR_30M, "close")
        ret_60m_close = get_bar_value(core_df, BAR_60M, "close")
        ret_noon_close = get_bar_value(core_df, BAR_NOON, "close")

        gap_pct = (open_px / prev_close - 1.0) * 100.0
        gap_sign = safe_sign(gap_pct)

        open_to_5m = ((ret_5m_close / open_px - 1.0) * 100.0) if ret_5m_close else np.nan
        open_to_15m = ((ret_15m_close / open_px - 1.0) * 100.0) if ret_15m_close else np.nan
        open_to_30m = ((ret_30m_close / open_px - 1.0) * 100.0) if ret_30m_close else np.nan
        open_to_60m = ((ret_60m_close / open_px - 1.0) * 100.0) if ret_60m_close else np.nan
        open_to_noon = ((ret_noon_close / open_px - 1.0) * 100.0) if ret_noon_close else np.nan
        open_to_close = ((close_px / open_px - 1.0) * 100.0)

        first5_confirms = int(safe_sign(open_to_5m) == gap_sign) if gap_sign != 0 and pd.notna(open_to_5m) else np.nan
        pm_ret = ((pm_last / prev_close - 1.0) * 100.0) if pm_last and prev_close else np.nan
        pm_confirms = int(safe_sign(pm_ret) == gap_sign) if gap_sign != 0 and pd.notna(pm_ret) else np.nan

        continues_15m = int(safe_sign(open_to_15m) == gap_sign) if gap_sign != 0 and pd.notna(open_to_15m) else np.nan
        continues_close = int(safe_sign(open_to_close) == gap_sign) if gap_sign != 0 and pd.notna(open_to_close) else np.nan

        fill_stats = gap_fill_stats(core_df, prev_close, open_px, gap_sign)

        rows.append(
            {
                "session_date": day,
                "dow": str(day_df.iloc[0]["dow"]),
                "prev_close": prev_close,
                "open_0930": open_px,
                "close_1559": close_px,
                "gap_pct": gap_pct,
                "abs_gap_pct": abs(gap_pct),
                "gap_direction": "up" if gap_sign > 0 else ("down" if gap_sign < 0 else "flat"),
                "abs_gap_bucket": gap_bucket(abs(gap_pct)),
                "premarket_last_close": pm_last,
                "premarket_return_pct": pm_ret,
                "premarket_high": pm_high,
                "premarket_low": pm_low,
                "premarket_mid": pm_mid,
                "premarket_range_pct": ((pm_high - pm_low) / prev_close * 100.0) if pd.notna(pm_high) and pd.notna(pm_low) else np.nan,
                "premarket_volume": pm_volume,
                "open_vs_premarket_mid_pct": ((open_px - pm_mid) / prev_close * 100.0) if pd.notna(pm_mid) else np.nan,
                "open_vs_premarket_high_pct": ((open_px - pm_high) / prev_close * 100.0) if pd.notna(pm_high) else np.nan,
                "open_vs_premarket_low_pct": ((open_px - pm_low) / prev_close * 100.0) if pd.notna(pm_low) else np.nan,
                "open_to_5m_return_pct": open_to_5m,
                "open_to_15m_return_pct": open_to_15m,
                "open_to_30m_return_pct": open_to_30m,
                "open_to_60m_return_pct": open_to_60m,
                "open_to_noon_return_pct": open_to_noon,
                "open_to_close_return_pct": open_to_close,
                "gap_continues_15m": continues_15m,
                "gap_continues_close": continues_close,
                "first5_confirms_gap": first5_confirms,
                "pm_return_confirms_gap": pm_confirms,
                **fill_stats,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--save-raw", action="store_true", help="also save raw minute bars")
    args = parser.parse_args()

    raw = fetch_minute_bars(args.symbol, args.start, args.end)
    dataset = build_daily_dataset(raw)

    raw_path = DATA_DIR / f"{args.symbol.lower()}_minute_raw.csv"
    ds_path = DATA_DIR / f"{args.symbol.lower()}_gap_daily_dataset.csv"

    if args.save_raw:
        raw.to_csv(raw_path, index=False)
        print(f"saved raw minute bars -> {raw_path}")

    dataset.to_csv(ds_path, index=False)
    print(f"saved daily gap dataset -> {ds_path}")
    print(f"rows: {len(dataset)}")
    if not dataset.empty:
        print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()
