#!/usr/bin/env python3
"""Phase 3: Analyze overnight market structure for MES and MNQ.

Computes volume, volatility, and structural patterns during the
overnight session (6 PM - 9:30 AM ET) to identify where exploitable
edges might exist before testing any strategies.
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")


def load_full_data(symbol: str) -> pd.DataFrame:
    """Load full-session data (including overnight)."""
    path = os.path.join(CACHE_DIR, f"{symbol}_futures_full_1min.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No full-session data: {path}")

    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(ET)

    df["hour"] = df["dt"].dt.hour
    df["minute"] = df["dt"].dt.minute
    df["time_str"] = df["dt"].dt.strftime("%H:%M")
    df["date"] = df["dt"].dt.date
    df["weekday"] = df["dt"].dt.weekday  # 0=Mon

    # Session classification
    df["session"] = "overnight"
    cash_mask = (df["time_str"] >= "09:30") & (df["time_str"] < "16:00")
    df.loc[cash_mask, "session"] = "cash"

    # Sub-session classification for overnight
    df["sub_session"] = "other"
    df.loc[(df["hour"] >= 18) | (df["hour"] < 2), "sub_session"] = "evening"      # 6pm-2am
    df.loc[(df["hour"] >= 2) & (df["hour"] < 4), "sub_session"] = "asia_close"    # 2-4am
    df.loc[(df["hour"] >= 3) & (df["hour"] < 5), "sub_session"] = "euro_open"     # 3-5am
    df.loc[(df["hour"] >= 7) & (df["hour"] < 9), "sub_session"] = "premarket"     # 7-9am
    df.loc[(df["hour"] == 8) & (df["minute"] >= 25) & (df["minute"] <= 40), "sub_session"] = "econ_data"  # 8:25-8:40
    df.loc[cash_mask, "sub_session"] = "cash"

    df["bar_range"] = df["high"] - df["low"]

    log.info("Loaded %d bars for %s (%s to %s)", len(df), symbol,
             df["dt"].iloc[0].strftime("%Y-%m-%d"), df["dt"].iloc[-1].strftime("%Y-%m-%d"))

    return df


def analyze_volume_by_hour(df: pd.DataFrame, symbol: str):
    """Average volume by hour from 6 PM through next day."""
    print(f"\n  --- Volume by Hour ({symbol}) ---")
    print(f"  {'Hour':<8} {'Avg Vol':>10} {'Med Vol':>10} {'% of Daily':>12} {'Session'}")

    # Create hour order starting from 18 (6pm)
    hour_order = list(range(18, 24)) + list(range(0, 17))

    # Daily total volume for percentages
    daily_vol = df.groupby("date")["volume"].sum()
    avg_daily = daily_vol.mean()

    for h in hour_order:
        mask = df["hour"] == h
        if not mask.any():
            continue
        hourly = df[mask]["volume"]
        avg_v = hourly.mean()
        med_v = hourly.median()
        pct = avg_v * 60 / avg_daily * 100 if avg_daily > 0 else 0

        session = "cash" if 9 <= h < 16 else "overnight"
        if h == 9:
            session = "open"
        elif h == 15:
            session = "close"
        elif 3 <= h <= 4:
            session = "euro"
        elif 7 <= h <= 8:
            session = "pre-mkt"
        elif 18 <= h <= 21:
            session = "evening"
        elif 22 <= h or h <= 2:
            session = "dead"

        bar = "#" * int(pct / 2)
        print(f"  {h:02d}:00    {avg_v:>10,.0f} {med_v:>10,.0f}   {pct:>5.1f}%     {session:<8} {bar}")


def analyze_volatility_by_hour(df: pd.DataFrame, symbol: str):
    """Average bar range (high-low) by hour."""
    print(f"\n  --- Bar Range by Hour ({symbol}) ---")
    print(f"  {'Hour':<8} {'Avg Range':>10} {'Med Range':>10} {'Avg Range%':>12}")

    hour_order = list(range(18, 24)) + list(range(0, 17))

    for h in hour_order:
        mask = df["hour"] == h
        if not mask.any():
            continue
        ranges = df[mask]["bar_range"]
        prices = df[mask]["close"]
        avg_r = ranges.mean()
        med_r = ranges.median()
        avg_pct = (ranges / prices * 100).mean()

        bar = "#" * int(avg_pct * 500)
        print(f"  {h:02d}:00    {avg_r:>10.4f} {med_r:>10.4f}   {avg_pct:>8.4f}%   {bar}")


def analyze_overnight_returns(df: pd.DataFrame, symbol: str):
    """Distribution of overnight returns."""
    print(f"\n  --- Overnight Return Distribution ({symbol}) ---")

    dates = sorted(df["date"].unique())
    overnight_returns = []
    cash_ranges = []
    overnight_ranges = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        # Previous day close (last cash bar)
        prev_cash = df[(df["date"] == prev_date) & (df["session"] == "cash")]
        if prev_cash.empty:
            continue
        prev_close = prev_cash["close"].iloc[-1]

        # Current day open (first cash bar)
        curr_cash = df[(df["date"] == curr_date) & (df["session"] == "cash")]
        if curr_cash.empty:
            continue
        curr_open = curr_cash["open"].iloc[0]

        # Overnight return: prev close -> current open
        on_ret = (curr_open - prev_close) / prev_close * 100
        overnight_returns.append(on_ret)

        # Cash session range
        cash_range = curr_cash["high"].max() - curr_cash["low"].min()
        cash_ranges.append(cash_range)

        # Overnight range (bars between prev close and curr open)
        on_bars = df[(df["dt"] > prev_cash["dt"].iloc[-1]) & (df["dt"] < curr_cash["dt"].iloc[0])]
        if not on_bars.empty:
            on_range = on_bars["high"].max() - on_bars["low"].min()
            overnight_ranges.append(on_range)

    on_ret = np.array(overnight_returns)

    print(f"  Observations:         {len(on_ret)}")
    print(f"  Mean return:          {on_ret.mean():+.4f}%")
    print(f"  Median return:        {np.median(on_ret):+.4f}%")
    print(f"  Std dev:              {on_ret.std():.4f}%")
    print(f"  Skew:                 {pd.Series(on_ret).skew():.3f}")
    print(f"  % positive:           {(on_ret > 0).mean()*100:.1f}%")
    print(f"  % negative:           {(on_ret < 0).mean()*100:.1f}%")
    print(f"  Max up:               {on_ret.max():+.4f}%")
    print(f"  Max down:             {on_ret.min():+.4f}%")

    # Return distribution
    print(f"\n  Return distribution:")
    for threshold in [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]:
        pct = (on_ret <= threshold).mean() * 100
        print(f"    <= {threshold:+.2f}%: {pct:5.1f}%")

    # Overnight vs cash range
    if overnight_ranges and cash_ranges:
        on_r = np.array(overnight_ranges)
        cash_r = np.array(cash_ranges[:len(on_r)])
        total_r = on_r + cash_r
        on_pct = on_r / total_r * 100

        print(f"\n  Overnight vs Cash Range:")
        print(f"  Avg overnight range:  {on_r.mean():.4f}")
        print(f"  Avg cash range:       {cash_r.mean():.4f}")
        print(f"  Overnight % of total: {on_pct.mean():.1f}%")

    return on_ret


def analyze_overnight_cash_correlation(df: pd.DataFrame, symbol: str, on_returns: np.ndarray):
    """Correlation between overnight direction and cash session direction."""
    print(f"\n  --- Overnight->Cash Correlation ({symbol}) ---")

    dates = sorted(df["date"].unique())
    pairs = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev_cash = df[(df["date"] == prev_date) & (df["session"] == "cash")]
        curr_cash = df[(df["date"] == curr_date) & (df["session"] == "cash")]
        if prev_cash.empty or curr_cash.empty:
            continue

        prev_close = prev_cash["close"].iloc[-1]
        curr_open = curr_cash["open"].iloc[0]
        curr_close = curr_cash["close"].iloc[-1]

        on_ret = (curr_open - prev_close) / prev_close * 100
        cash_ret = (curr_close - curr_open) / curr_open * 100

        pairs.append((on_ret, cash_ret))

    if not pairs:
        print("  No data")
        return

    on = np.array([p[0] for p in pairs])
    cash = np.array([p[1] for p in pairs])

    corr = np.corrcoef(on, cash)[0, 1]
    print(f"  Pearson correlation:    {corr:.4f}")

    # Conditional analysis
    up_nights = on > 0
    down_nights = on < 0

    if up_nights.any():
        cash_after_up = cash[up_nights].mean()
        print(f"  Cash return after UP overnight:   {cash_after_up:+.4f}% (n={up_nights.sum()})")
    if down_nights.any():
        cash_after_down = cash[down_nights].mean()
        print(f"  Cash return after DOWN overnight: {cash_after_down:+.4f}% (n={down_nights.sum()})")

    # Big gap analysis
    big_up = on > 0.3
    big_down = on < -0.3
    if big_up.any():
        print(f"  Cash after gap UP >0.3%:  {cash[big_up].mean():+.4f}% (n={big_up.sum()})")
    if big_down.any():
        print(f"  Cash after gap DOWN <-0.3%: {cash[big_down].mean():+.4f}% (n={big_down.sum()})")

    # Does overnight predict direction?
    same_dir = ((on > 0) & (cash > 0)) | ((on < 0) & (cash < 0))
    print(f"  Same direction:         {same_dir.mean()*100:.1f}%")
    print(f"  Opposite direction:     {(~same_dir).mean()*100:.1f}%")


def analyze_structural_events(df: pd.DataFrame, symbol: str):
    """Identify volume/volatility spikes around key events."""
    print(f"\n  --- Structural Events ({symbol}) ---")

    events = [
        ("Asian close (3:30-4:30 AM)", 3, 30, 4, 30),
        ("European open (3:00-3:30 AM)", 3, 0, 3, 30),
        ("European equities (3:00-5:00 AM)", 3, 0, 5, 0),
        ("Pre-market (7:00-8:00 AM)", 7, 0, 8, 0),
        ("Econ data (8:25-8:45 AM)", 8, 25, 8, 45),
        ("Pre-open (8:45-9:30 AM)", 8, 45, 9, 30),
        ("Dead zone (10 PM-2 AM)", 22, 0, 2, 0),
        ("Evening (6-8 PM)", 18, 0, 20, 0),
    ]

    print(f"  {'Event':<35} {'Avg Vol':>8} {'Avg Rng':>8} {'Rel Vol':>8} {'Rel Rng':>8}")
    print(f"  {'-'*75}")

    # Baseline: average bar during cash hours
    cash_vol = df[df["session"] == "cash"]["volume"].mean()
    cash_rng = df[df["session"] == "cash"]["bar_range"].mean()

    for name, h1, m1, h2, m2 in events:
        time_min_start = h1 * 60 + m1
        time_min_end = h2 * 60 + m2

        min_of_day = df["hour"] * 60 + df["minute"]

        if time_min_start < time_min_end:
            mask = (min_of_day >= time_min_start) & (min_of_day < time_min_end)
        else:
            # Wraps past midnight
            mask = (min_of_day >= time_min_start) | (min_of_day < time_min_end)

        if not mask.any():
            continue

        avg_vol = df[mask]["volume"].mean()
        avg_rng = df[mask]["bar_range"].mean()
        rel_vol = avg_vol / cash_vol if cash_vol > 0 else 0
        rel_rng = avg_rng / cash_rng if cash_rng > 0 else 0

        print(f"  {name:<35} {avg_vol:>8,.0f} {avg_rng:>8.4f} {rel_vol:>7.2f}x {rel_rng:>7.2f}x")


def main():
    print("=" * 70)
    print("  PHASE 3: OVERNIGHT MARKET STRUCTURE ANALYSIS")
    print("=" * 70)

    for symbol in ["MES", "MNQ"]:
        try:
            df = load_full_data(symbol)
        except FileNotFoundError as e:
            print(f"\n  SKIPPING {symbol}: {e}")
            continue

        print(f"\n{'='*70}")
        print(f"  {symbol} OVERNIGHT ANALYSIS")
        print(f"{'='*70}")
        print(f"  Total bars: {len(df):,}")
        print(f"  Cash bars:  {(df['session']=='cash').sum():,}")
        print(f"  Overnight:  {(df['session']=='overnight').sum():,}")
        print(f"  Date range: {df['dt'].iloc[0].strftime('%Y-%m-%d')} to "
              f"{df['dt'].iloc[-1].strftime('%Y-%m-%d')}")

        analyze_volume_by_hour(df, symbol)
        analyze_volatility_by_hour(df, symbol)
        on_ret = analyze_overnight_returns(df, symbol)
        analyze_overnight_cash_correlation(df, symbol, on_ret)
        analyze_structural_events(df, symbol)

    print(f"\n{'='*70}")
    print("  STRATEGY IMPLICATIONS")
    print("=" * 70)
    print("  See the analysis above to determine which overnight windows")
    print("  have sufficient volume and volatility to support a strategy.")
    print("  Key questions:")
    print("  1. Where is volume concentrated? (European open? Pre-market?)")
    print("  2. Does overnight mean-revert or trend?")
    print("  3. How big is overnight range vs cash range?")
    print("  4. Does overnight direction predict cash direction?")
    print("=" * 70)


if __name__ == "__main__":
    main()
