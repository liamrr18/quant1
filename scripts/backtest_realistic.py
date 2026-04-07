import os
import pandas as pd

DATA_PATH = r"data\gap_edge\spy_gap_daily_dataset.csv"
OUT_DIR = r"data\gap_edge"
OUT_PATH = os.path.join(OUT_DIR, "gap_fade_realistic_backtest.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "gap_fade_realistic_summary.csv")

MAX_GAP_PCT = 0.20
SLIPPAGE_PCT = 0.0002  # 2 bps each side
MIN_TRADES_FOR_KEEP = 10


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.sort_values("session_date").reset_index(drop=True)

    # Use only information known before today's open
    prev_session_close = df["close_1559"].shift(1)

    # 20-day trend filter: only short fade when market is below its 20-day average
    df["ma20"] = prev_session_close.rolling(20).mean()
    df["trend_short_ok"] = prev_session_close < df["ma20"]

    # 20-day realized volatility from prior daily close-to-close returns
    daily_ret = df["close_1559"].pct_change() * 100.0
    df["vol20"] = daily_ret.shift(1).rolling(20).std()

    # Compare current vol regime to prior expanding median vol
    df["vol20_median_expanding"] = (
        df["vol20"].expanding(min_periods=20).median().shift(1)
    )
    df["vol_high_ok"] = df["vol20"] > df["vol20_median_expanding"]

    return df


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_ret_pct": 0.0,
            "median_ret_pct": 0.0,
            "total_ret_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
        }

    wins = trades.loc[trades["trade_return_pct"] > 0, "trade_return_pct"]
    losses = trades.loc[trades["trade_return_pct"] <= 0, "trade_return_pct"]

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return {
        "trades": int(len(trades)),
        "win_rate": float(trades["win"].mean()),
        "avg_ret_pct": float(trades["trade_return_pct"].mean()),
        "median_ret_pct": float(trades["trade_return_pct"].median()),
        "total_ret_pct": float(trades["trade_return_pct"].sum()),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(trades["drawdown_pct"].min()),
    }


def run_variant(
    df: pd.DataFrame,
    name: str,
    use_trend_filter: bool,
    use_vol_filter: bool,
):
    trades = df.copy()

    # Base edge
    trades = trades[trades["gap_direction"] == "up"].copy()
    trades = trades[trades["abs_gap_pct"] < MAX_GAP_PCT].copy()
    trades = trades[trades["first5_confirms_gap"] == 0].copy()

    # Regime filters
    if use_trend_filter:
        trades = trades[trades["trend_short_ok"] == True].copy()

    if use_vol_filter:
        trades = trades[trades["vol_high_ok"] == True].copy()

    trades = trades.dropna(subset=["ma20", "vol20", "vol20_median_expanding"]).copy()

    if trades.empty:
        return pd.DataFrame(), {
            "variant": name,
            "use_trend_filter": use_trend_filter,
            "use_vol_filter": use_vol_filter,
            "trades": 0,
            "win_rate": 0.0,
            "avg_ret_pct": 0.0,
            "median_ret_pct": 0.0,
            "total_ret_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
        }

    # Short at 9:35 using move from 5m close to noon
    trades["trade_return_pct"] = -(
        trades["open_to_noon_return_pct"] - trades["open_to_5m_return_pct"]
    )

    # Round-trip slippage
    trades["trade_return_pct"] = trades["trade_return_pct"] - (2 * SLIPPAGE_PCT * 100)

    trades["win"] = (trades["trade_return_pct"] > 0).astype(int)
    trades["cum_return_pct"] = trades["trade_return_pct"].cumsum()
    trades["equity_peak"] = trades["cum_return_pct"].cummax()
    trades["drawdown_pct"] = trades["cum_return_pct"] - trades["equity_peak"]

    stats = summarize(trades)
    stats["variant"] = name
    stats["use_trend_filter"] = use_trend_filter
    stats["use_vol_filter"] = use_vol_filter

    return trades, stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = add_regime_features(df)

    variants = [
        ("baseline", False, False),
        ("trend_only", True, False),
        ("vol_only", False, True),
        ("trend_and_vol", True, True),
    ]

    results = []
    trade_tables = {}

    for name, use_trend, use_vol in variants:
        trades, stats = run_variant(df, name, use_trend, use_vol)
        results.append(stats)
        trade_tables[name] = trades

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(
        ["profit_factor", "total_ret_pct", "trades"],
        ascending=[False, False, False]
    )

    print("Realistic Gap Fade Backtest Variants")
    print("------------------------------------")
    print(f"MAX_GAP_PCT: {MAX_GAP_PCT}")
    print(f"SLIPPAGE_PCT (each side): {SLIPPAGE_PCT * 100:.4f}%\n")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved summary -> {SUMMARY_PATH}")

    # Pick best variant that still has enough trades
    eligible = summary_df[summary_df["trades"] >= MIN_TRADES_FOR_KEEP].copy()

    if eligible.empty:
        print(f"\nNo variant had at least {MIN_TRADES_FOR_KEEP} trades.")
        return

    best_variant = eligible.iloc[0]["variant"]
    best_trades = trade_tables[best_variant].copy()

    cols = [
        "session_date",
        "gap_pct",
        "abs_gap_pct",
        "first5_confirms_gap",
        "trend_short_ok",
        "vol_high_ok",
        "ma20",
        "vol20",
        "vol20_median_expanding",
        "open_to_5m_return_pct",
        "open_to_noon_return_pct",
        "trade_return_pct",
        "cum_return_pct",
        "drawdown_pct",
    ]
    best_trades[cols].to_csv(OUT_PATH, index=False)

    print(f"\nBest kept variant -> {best_variant}")
    print(f"Saved trades -> {OUT_PATH}")


if __name__ == "__main__":
    main()