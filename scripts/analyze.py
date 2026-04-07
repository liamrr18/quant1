
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR


DATASET_PATH = DATA_DIR / "spy_gap_daily_dataset.csv"


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    pm_vol_med = out["premarket_volume"].median()
    pm_range_med = out["premarket_range_pct"].median()

    out["pm_volume_bucket"] = np.where(out["premarket_volume"] >= pm_vol_med, "high_pm_vol", "low_pm_vol")
    out["pm_range_bucket"] = np.where(out["premarket_range_pct"] >= pm_range_med, "wide_pm_range", "tight_pm_range")

    out["gap_open_to_15m_signed_move_pct"] = np.sign(out["gap_pct"]) * out["open_to_15m_return_pct"]
    out["gap_open_to_close_signed_move_pct"] = np.sign(out["gap_pct"]) * out["open_to_close_return_pct"]

    out["fade_15m"] = 1 - out["gap_continues_15m"]
    out["fade_close"] = 1 - out["gap_continues_close"]

    return out


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby(["gap_direction", "abs_gap_bucket"], dropna=False)
        .agg(
            count=("session_date", "count"),
            mean_gap_pct=("gap_pct", "mean"),
            mean_open_to_15m_pct=("open_to_15m_return_pct", "mean"),
            mean_open_to_close_pct=("open_to_close_return_pct", "mean"),
            continue_15m_rate=("gap_continues_15m", "mean"),
            continue_close_rate=("gap_continues_close", "mean"),
            fill_by_10_rate=("gap_fills_by_10", "mean"),
            fill_by_noon_rate=("gap_fills_by_noon", "mean"),
            fill_by_close_rate=("gap_fills_by_close", "mean"),
            avg_fill_frac_close=("gap_fill_fraction_by_close", "mean"),
        )
        .reset_index()
        .sort_values(["gap_direction", "abs_gap_bucket"])
    )

    for c in [
        "continue_15m_rate",
        "continue_close_rate",
        "fill_by_10_rate",
        "fill_by_noon_rate",
        "fill_by_close_rate",
    ]:
        grp[c] = (grp[c] * 100.0).round(2)

    return grp


def rule_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "gap_direction",
        "abs_gap_bucket",
        "first5_confirms_gap",
        "pm_return_confirms_gap",
        "pm_volume_bucket",
        "pm_range_bucket",
    ]

    grp = (
        df.groupby(cols, dropna=False)
        .agg(
            count=("session_date", "count"),
            mean_gap_pct=("gap_pct", "mean"),
            signed_15m_move_pct=("gap_open_to_15m_signed_move_pct", "mean"),
            signed_close_move_pct=("gap_open_to_close_signed_move_pct", "mean"),
            continue_15m_rate=("gap_continues_15m", "mean"),
            continue_close_rate=("gap_continues_close", "mean"),
            fill_by_10_rate=("gap_fills_by_10", "mean"),
            fill_by_close_rate=("gap_fills_by_close", "mean"),
        )
        .reset_index()
    )

    for c in ["continue_15m_rate", "continue_close_rate", "fill_by_10_rate", "fill_by_close_rate"]:
        grp[c] = grp[c] * 100.0

    return grp.sort_values(["count", "signed_close_move_pct"], ascending=[False, False])


def best_continuation_setups(rules: pd.DataFrame) -> pd.DataFrame:
    out = rules.copy()
    out = out[out["count"] >= 20].copy()
    out = out.sort_values(
        ["signed_close_move_pct", "continue_close_rate", "count"],
        ascending=[False, False, False],
    )
    return out.head(20)


def best_fade_setups(rules: pd.DataFrame) -> pd.DataFrame:
    out = rules.copy()
    out = out[out["count"] >= 20].copy()
    out["fade_close_rate"] = 100.0 - out["continue_close_rate"]
    out["signed_fade_move_pct"] = -out["signed_close_move_pct"]
    out = out.sort_values(
        ["signed_fade_move_pct", "fade_close_rate", "count"],
        ascending=[False, False, False],
    )
    return out.head(20)


def print_high_level_takeaways(df: pd.DataFrame, overall: pd.DataFrame):
    print("\nHIGH-LEVEL TAKEAWAYS\n")

    if df.empty:
        print("Dataset is empty.")
        return

    baseline_15m = float(df["gap_continues_15m"].mean() * 100.0)
    baseline_close = float(df["gap_continues_close"].mean() * 100.0)

    print(f"Overall continuation rate to 15m:  {baseline_15m:.2f}%")
    print(f"Overall continuation rate to close: {baseline_close:.2f}%")

    if not overall.empty:
        best_cont = overall.sort_values("continue_close_rate", ascending=False).iloc[0]
        best_fill = overall.sort_values("fill_by_close_rate", ascending=False).iloc[0]

        print(
            f"\nBest raw continuation bucket: {best_cont['gap_direction']} / {best_cont['abs_gap_bucket']} "
            f"| n={int(best_cont['count'])} | continue_close={best_cont['continue_close_rate']:.2f}% "
            f"| mean open->close={best_cont['mean_open_to_close_pct']:.3f}%"
        )
        print(
            f"Best raw fade/fill bucket: {best_fill['gap_direction']} / {best_fill['abs_gap_bucket']} "
            f"| n={int(best_fill['count'])} | fill_by_close={best_fill['fill_by_close_rate']:.2f}% "
            f"| mean open->close={best_fill['mean_open_to_close_pct']:.3f}%"
        )


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found. Run build_gap_dataset first.")

    df = pd.read_csv(DATASET_PATH)
    df = add_analysis_columns(df)

    overall = overall_summary(df)
    rules = rule_summary(df)
    cont = best_continuation_setups(rules)
    fade = best_fade_setups(rules)

    overall_path = DATA_DIR / "gap_overall_summary.csv"
    rules_path = DATA_DIR / "gap_rule_summary.csv"
    cont_path = DATA_DIR / "best_continuation_setups.csv"
    fade_path = DATA_DIR / "best_fade_setups.csv"

    overall.to_csv(overall_path, index=False)
    rules.to_csv(rules_path, index=False)
    cont.to_csv(cont_path, index=False)
    fade.to_csv(fade_path, index=False)

    print(f"saved -> {overall_path}")
    print(f"saved -> {rules_path}")
    print(f"saved -> {cont_path}")
    print(f"saved -> {fade_path}")

    print_high_level_takeaways(df, overall)

    print("\nTOP CONTINUATION CANDIDATES\n")
    if cont.empty:
        print("No continuation candidates met the sample-size filter.")
    else:
        print(cont.head(10).to_string(index=False))

    print("\nTOP FADE CANDIDATES\n")
    if fade.empty:
        print("No fade candidates met the sample-size filter.")
    else:
        print(fade.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
