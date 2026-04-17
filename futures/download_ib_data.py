#!/usr/bin/env python3
"""Download full-session historical 1-minute bars from IB for MES and MNQ.

Chains quarterly contracts together to build a continuous dataset.
Covers the full 23-hour session (6 PM - 5 PM ET), not just regular hours.
Handles IB pacing limits gracefully with retry logic.

Usage:
    python download_ib_data.py
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytz
from ib_insync import IB, Future, util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")

# Quarterly contract months: H=Mar, M=Jun, U=Sep, Z=Dec
# Each contract is the front month for ~3 months before its expiry
QUARTERLY_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}

# IB limits: 1-min bars, max 1 day per request with useRTH=False for futures
# Pacing: ~60 requests per 10 minutes, with 2000 bars max per request
# We request 1 day at a time to stay safe.

IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 4  # Use different clientId than the live trader


def get_quarterly_contracts(symbol: str, start_year: int, end_year: int) -> list[dict]:
    """Generate list of quarterly contract specs from start to end year."""
    contracts = []
    for year in range(start_year, end_year + 1):
        for month in [3, 6, 9, 12]:
            # Contract becomes front month ~3 months before expiry
            # We'll request data for the period when each contract was actively traded
            contracts.append({
                "symbol": symbol,
                "year": year,
                "month": month,
                "yyyymm": f"{year}{month:02d}",
                "code": QUARTERLY_CODES[month],
            })
    return contracts


def find_third_friday(year: int, month: int) -> datetime:
    """Find 3rd Friday of a month (futures expiry)."""
    import calendar
    cal = calendar.monthcalendar(year, month)
    fridays = [week[4] for week in cal if week[4] != 0]
    return datetime(year, month, fridays[2], 17, 0, tzinfo=ET)


def download_contract_bars(ib: IB, symbol: str, yyyymm: str,
                            start_date: datetime, end_date: datetime,
                            include_expired: bool = True) -> pd.DataFrame:
    """Download 1-min bars for a single contract over a date range.

    Requests one day at a time to respect IB limits.
    """
    contract = Future(
        symbol=symbol,
        exchange="CME",
        lastTradeDateOrContractMonth=yyyymm,
        includeExpired=include_expired,
    )

    qualified = ib.qualifyContracts(contract)
    if not qualified:
        log.warning("Cannot qualify %s %s (includeExpired=%s)", symbol, yyyymm, include_expired)
        return pd.DataFrame()

    log.info("Downloading %s (%s) from %s to %s",
             contract.localSymbol, yyyymm,
             start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    all_bars = []
    current_end = end_date
    request_count = 0
    consecutive_empty = 0

    while current_end > start_date:
        # Use UTC dash format: yyyymmdd-HH:mm:ss (IB preferred)
        end_utc = current_end.astimezone(UTC) if current_end.tzinfo else UTC.localize(current_end)
        end_str = end_utc.strftime("%Y%m%d-%H:%M:%S")

        for attempt in range(5):
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_str,
                    durationStr="1 D",
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,
                    timeout=30,
                )
                request_count += 1

                if bars:
                    df = util.df(bars)
                    all_bars.append(df)
                    earliest = bars[0].date
                    log.info("  %s -> %d bars (%s to %s)",
                             end_str[:8], len(bars),
                             str(bars[0].date)[:16], str(bars[-1].date)[:16])
                    consecutive_empty = 0
                    # Move end back to before the earliest bar we got
                    current_end = pd.Timestamp(earliest).to_pydatetime() - timedelta(minutes=1)
                    if current_end.tzinfo is None:
                        current_end = UTC.localize(current_end)
                else:
                    consecutive_empty += 1
                    current_end -= timedelta(days=1)

                # Pacing: IB allows ~60 requests per 10 min
                if request_count % 5 == 0:
                    time.sleep(3)
                else:
                    time.sleep(1)

                break  # Success, exit retry loop

            except Exception as e:
                err_str = str(e)
                if "pacing" in err_str.lower() or "162" in err_str:
                    wait = 15 * (attempt + 1)
                    log.warning("Pacing violation, waiting %ds (attempt %d/5)", wait, attempt + 1)
                    time.sleep(wait)
                elif "no data" in err_str.lower() or "HMDS query" in err_str:
                    log.debug("No data for %s at %s, moving back", symbol, end_str)
                    consecutive_empty += 1
                    current_end -= timedelta(days=1)
                    break
                else:
                    log.error("Error downloading %s at %s: %s", symbol, end_str, e)
                    if attempt < 4:
                        time.sleep(5)
                    else:
                        current_end -= timedelta(days=1)
                    break

        # If we hit many consecutive empty days, we've gone past the contract's data
        if consecutive_empty > 10:
            log.info("  10+ consecutive empty days, stopping for %s %s", symbol, yyyymm)
            break

    if not all_bars:
        return pd.DataFrame()

    result = pd.concat(all_bars, ignore_index=True)
    result = result.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    log.info("  %s %s: %d bars total, %d requests",
             symbol, yyyymm, len(result), request_count)
    return result


def build_continuous_series(ib: IB, symbol: str) -> pd.DataFrame:
    """Build a continuous 1-min bar series by chaining quarterly contracts.

    Goes back as far as IB has data (typically ~2 years for 1-min).
    Uses the front-month contract for each period.
    """
    now = datetime.now(ET)

    # Generate contracts from 2 years ago to now
    start_year = now.year - 2
    contracts = get_quarterly_contracts(symbol, start_year, now.year)

    # Filter to contracts that could have data (not too far in the future)
    contracts = [c for c in contracts
                 if datetime(c["year"], c["month"], 1, tzinfo=ET) <= now + timedelta(days=90)]

    log.info("Will try %d quarterly contracts for %s (%d-%d)",
             len(contracts), symbol, start_year, now.year)

    all_dfs = []

    for c in contracts:
        expiry = find_third_friday(c["year"], c["month"])
        is_expired = expiry < now

        # Each contract is front month for ~3 months before expiry
        # Request data from 3 months before expiry to expiry
        data_start = (expiry - timedelta(days=95)).replace(tzinfo=ET)
        data_end = min(expiry, now.replace(tzinfo=ET))

        if data_end.tzinfo is None:
            data_end = ET.localize(data_end)
        if data_start.tzinfo is None:
            data_start = ET.localize(data_start)

        # Convert to UTC for IB
        data_start_utc = data_start.astimezone(UTC)
        data_end_utc = data_end.astimezone(UTC)

        df = download_contract_bars(
            ib, symbol, c["yyyymm"],
            data_start_utc, data_end_utc,
            include_expired=is_expired,
        )

        if not df.empty:
            df["contract"] = f"{symbol}{c['code']}{c['year'] % 100}"
            all_dfs.append(df)

        # Extra pacing between contracts
        time.sleep(5)

    if not all_dfs:
        log.error("No data obtained for %s", symbol)
        return pd.DataFrame()

    # Concatenate and deduplicate (overlapping periods between contracts)
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    result = result.reset_index(drop=True)

    # Rename columns to match our cache format
    result = result.rename(columns={"date": "dt"})

    # Ensure timezone info
    if result["dt"].dtype == "object":
        result["dt"] = pd.to_datetime(result["dt"])
    if hasattr(result["dt"].dt, "tz") and result["dt"].dt.tz is not None:
        result["dt"] = result["dt"].dt.tz_convert(ET)
    else:
        result["dt"] = result["dt"].dt.tz_localize(UTC).dt.tz_convert(ET)

    log.info("%s continuous: %d total bars, %s to %s",
             symbol, len(result),
             result["dt"].iloc[0].strftime("%Y-%m-%d %H:%M"),
             result["dt"].iloc[-1].strftime("%Y-%m-%d %H:%M"))

    return result


def verify_data(df: pd.DataFrame, symbol: str):
    """Print verification summary of downloaded data."""
    if df.empty:
        log.error("No data to verify for %s", symbol)
        return

    print(f"\n{'='*60}")
    print(f"  DATA VERIFICATION: {symbol}")
    print(f"{'='*60}")

    # Date range
    first = df["dt"].iloc[0]
    last = df["dt"].iloc[-1]
    print(f"  Date range:   {first.strftime('%Y-%m-%d %H:%M')} to {last.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Total bars:   {len(df):,}")

    # Days covered
    dates = df["dt"].dt.date.unique()
    print(f"  Trading days: {len(dates)}")

    # Cash vs overnight split
    hours = df["dt"].dt.hour
    minutes = df["dt"].dt.minute
    time_str = df["dt"].dt.strftime("%H:%M")
    cash_mask = (time_str >= "09:30") & (time_str < "16:00")
    overnight_mask = ~cash_mask

    cash_bars = cash_mask.sum()
    overnight_bars = overnight_mask.sum()
    print(f"  Cash bars:    {cash_bars:,} ({cash_bars/len(df)*100:.1f}%)")
    print(f"  Overnight:    {overnight_bars:,} ({overnight_bars/len(df)*100:.1f}%)")

    # Verify overnight hours exist
    print(f"\n  Overnight hour verification:")
    for check_hour in [18, 20, 22, 0, 2, 4, 6, 7, 8]:
        mask = hours == check_hour
        count = mask.sum()
        print(f"    {check_hour:02d}:00 hour: {count:,} bars {'OK' if count > 0 else 'MISSING'}")

    # Check for gaps (days with no data)
    all_dates = pd.date_range(first.date(), last.date(), freq="B")  # Business days
    data_dates = set(df["dt"].dt.date.unique())
    missing = [d.date() for d in all_dates if d.date() not in data_dates]
    if missing:
        print(f"\n  Missing business days: {len(missing)}")
        if len(missing) <= 10:
            for d in missing:
                print(f"    {d}")
    else:
        print(f"\n  No missing business days")

    # Contracts used
    if "contract" in df.columns:
        print(f"\n  Contracts used:")
        for contract, count in df["contract"].value_counts().sort_index().items():
            print(f"    {contract}: {count:,} bars")

    print(f"{'='*60}\n")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    log.info("Connecting to IB at %s:%d (clientId=%d)", IB_HOST, IB_PORT, IB_CLIENT_ID)
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=30)
    ib.reqMarketDataType(1)
    log.info("IB connected")

    try:
        for symbol in ["MES", "MNQ"]:
            log.info("=" * 60)
            log.info("Downloading %s full-session data...", symbol)
            log.info("=" * 60)

            df = build_continuous_series(ib, symbol)

            if df.empty:
                log.error("No data for %s, skipping", symbol)
                continue

            # Save to cache
            cache_path = os.path.join(CACHE_DIR, f"{symbol}_futures_full_1min.csv")
            df.to_csv(cache_path, index=False)
            log.info("Saved %d bars to %s", len(df), cache_path)

            # Also save a cash-only version for Phase 2 comparison
            time_str = df["dt"].dt.strftime("%H:%M")
            cash_df = df[(time_str >= "09:30") & (time_str < "16:00")].copy()
            cash_path = os.path.join(CACHE_DIR, f"{symbol}_futures_cash_1min.csv")
            cash_df.to_csv(cash_path, index=False)
            log.info("Saved %d cash-hours bars to %s", len(cash_df), cash_path)

            verify_data(df, symbol)

    finally:
        ib.disconnect()
        log.info("IB disconnected")


if __name__ == "__main__":
    main()
