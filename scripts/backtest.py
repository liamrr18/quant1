import os
import pandas as pd

DATA_PATH = r"data\gap_edge\spy_gap_daily_dataset.csv"
OUT_PATH = r"data\gap_edge\gap_fade_backtest_results.csv"

# Base rule
GAP_DIRECTION = "up"
MAX_GAP_PCT = 0.30           # absolute gap less than 0.30%
REQUIRE_FIRST5_FAIL = True   # first 5m does NOT confirm the gap
ENTRY_MODE = "5m_close"      # "open" or "5m_close"
EXIT_MODE = "close"          # "close", "noon", "15m", "gap_fill"
SLIPPAGE_PCT = 0.0002        # 2 bps each side


def run_backtest():
    df = pd.read_csv(DATA_PATH)

    # candidate setup
    trades = df.copy()
    trades = trades[trades["gap_direction"] == GAP_DIRECTION].copy()
    trades = trades[trades["abs_gap_pct"] < MAX_GAP_PCT].copy()

    if REQUIRE_FIRST5_FAIL:
        trades = trades[trades["first5_confirms_gap"] == 0].copy()

    if trades.empty:
        print("No trades found for this setup.")
        return

    # entry return anchor
    if ENTRY_MODE == "open":
        # short from open
        if EXIT_MODE == "close":
            trades["trade_return_pct"] = -trades["open_to_close_return_pct"]
        elif EXIT_MODE == "noon":
            trades["trade_return_pct"] = -trades["open_to_noon_return_pct"]
        elif EXIT_MODE == "15m":
            trades["trade_return_pct"] = -trades["open_to_15m_return_pct"]
        elif EXIT_MODE == "gap_fill":
            # crude approximation:
            # if filled by close, assume capture equals gap size
            # otherwise mark to close
            trades["trade_return_pct"] = trades.apply(
                lambda r: r["gap_pct"] if r["gap_fills_by_close"] == 1 else -r["open_to_close_return_pct"],
                axis=1
            )
        else:
            raise ValueError("Invalid EXIT_MODE")

    elif ENTRY_MODE == "5m_close":
        # for fade setup, use move after 5m
        if EXIT_MODE == "close":
            trades["trade_return_pct"] = -(trades["open_to_close_return_pct"] - trades["open_to_5m_return_pct"])
        elif EXIT_MODE == "noon":
            trades["trade_return_pct"] = -(trades["open_to_noon_return_pct"] - trades["open_to_5m_return_pct"])
        elif EXIT_MODE == "15m":
            trades["trade_return_pct"] = -(trades["open_to_15m_return_pct"] - trades["open_to_5m_return_pct"])
        elif EXIT_MODE == "gap_fill":
            trades["trade_return_pct"] = trades.apply(
                lambda r: max(
                    0.0,
                    r["gap_pct"] + r["open_to_5m_return_pct"]
                ) if r["gap_fills_by_close"] == 1 else -(r["open_to_close_return_pct"] - r["open_to_5m_return_pct"]),
                axis=1
            )
        else:
            raise ValueError("Invalid EXIT_MODE")
    else:
        raise ValueError("Invalid ENTRY_MODE")

    # subtract round-trip slippage
    trades["trade_return_pct"] = trades["trade_return_pct"] - (2 * SLIPPAGE_PCT * 100)

    trades["win"] = (trades["trade_return_pct"] > 0).astype(int)
    trades["cum_return_pct"] = trades["trade_return_pct"].cumsum()
    trades["equity_peak"] = trades["cum_return_pct"].cummax()
    trades["drawdown_pct"] = trades["cum_return_pct"] - trades["equity_peak"]

    wins = trades.loc[trades["trade_return_pct"] > 0, "trade_return_pct"]
    losses = trades.loc[trades["trade_return_pct"] <= 0, "trade_return_pct"]

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    print("Gap Fade Backtest")
    print("-----------------")
    print(f"Trades: {len(trades)}")
    print(f"Win rate: {trades['win'].mean():.2%}")
    print(f"Average trade return: {trades['trade_return_pct'].mean():.4f}%")
    print(f"Median trade return: {trades['trade_return_pct'].median():.4f}%")
    print(f"Total return (sum): {trades['trade_return_pct'].sum():.4f}%")
    print(f"Profit factor: {profit_factor:.4f}")
    print(f"Max drawdown: {trades['drawdown_pct'].min():.4f}%")

    cols = [
        "session_date", "gap_pct", "abs_gap_pct", "first5_confirms_gap",
        "open_to_5m_return_pct", "open_to_15m_return_pct",
        "open_to_noon_return_pct", "open_to_close_return_pct",
        "gap_fills_by_close", "trade_return_pct", "cum_return_pct", "drawdown_pct"
    ]
    trades[cols].to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    run_backtest()