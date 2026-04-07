import pandas as pd

DATA_PATH = r"data\gap_edge\spy_gap_daily_dataset.csv"
OUT_PATH = r"data\gap_edge\gap_fade_sweep.csv"

SLIPPAGE_PCT = 0.0002


def simulate(df, max_gap_pct, require_first5_fail, entry_mode, exit_mode):
    trades = df.copy()
    trades = trades[trades["gap_direction"] == "up"].copy()
    trades = trades[trades["abs_gap_pct"] < max_gap_pct].copy()

    if require_first5_fail:
        trades = trades[trades["first5_confirms_gap"] == 0].copy()

    if trades.empty:
        return None

    if entry_mode == "open":
        if exit_mode == "close":
            trades["ret"] = -trades["open_to_close_return_pct"]
        elif exit_mode == "noon":
            trades["ret"] = -trades["open_to_noon_return_pct"]
        elif exit_mode == "15m":
            trades["ret"] = -trades["open_to_15m_return_pct"]
        else:
            return None
    elif entry_mode == "5m_close":
        if exit_mode == "close":
            trades["ret"] = -(trades["open_to_close_return_pct"] - trades["open_to_5m_return_pct"])
        elif exit_mode == "noon":
            trades["ret"] = -(trades["open_to_noon_return_pct"] - trades["open_to_5m_return_pct"])
        elif exit_mode == "15m":
            trades["ret"] = -(trades["open_to_15m_return_pct"] - trades["open_to_5m_return_pct"])
        else:
            return None
    else:
        return None

    trades["ret"] = trades["ret"] - (2 * SLIPPAGE_PCT * 100)

    wins = trades.loc[trades["ret"] > 0, "ret"]
    losses = trades.loc[trades["ret"] <= 0, "ret"]

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    equity = trades["ret"].cumsum()
    dd = (equity - equity.cummax()).min()

    return {
        "max_gap_pct": max_gap_pct,
        "require_first5_fail": require_first5_fail,
        "entry_mode": entry_mode,
        "exit_mode": exit_mode,
        "trades": len(trades),
        "win_rate": float((trades["ret"] > 0).mean()),
        "avg_ret_pct": float(trades["ret"].mean()),
        "median_ret_pct": float(trades["ret"].median()),
        "total_ret_pct": float(trades["ret"].sum()),
        "profit_factor": float(pf),
        "max_drawdown_pct": float(dd),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    results = []
    for max_gap_pct in [0.10, 0.20, 0.30]:
        for require_first5_fail in [True, False]:
            for entry_mode in ["open", "5m_close"]:
                for exit_mode in ["15m", "noon", "close"]:
                    row = simulate(df, max_gap_pct, require_first5_fail, entry_mode, exit_mode)
                    if row:
                        results.append(row)

    out = pd.DataFrame(results)
    out = out.sort_values(["total_ret_pct", "profit_factor", "win_rate"], ascending=False)
    out.to_csv(OUT_PATH, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()