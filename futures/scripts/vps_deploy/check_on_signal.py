import sys
sys.path.insert(0, "/root/futures_trader")

from ib_insync import IB, Future
import pandas as pd
import pytz

from trading.strategies.overnight_reversion import OvernightReversion
from trading.config import OVERNIGHT_REVERSION_DEFAULTS

ET = pytz.timezone("America/New_York")
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=96, timeout=15)

# Get MNQ front-month contract
c = Future(symbol="MNQ", exchange="CME")
details = ib.reqContractDetails(c)
details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
fc = details[0].contract
ib.qualifyContracts(fc)

# Pull 2 days of bars
bars = ib.reqHistoricalData(fc, "", "2 D", "1 min", "TRADES", False, 1)
df = pd.DataFrame([{
    "dt": b.date, "open": b.open, "high": b.high,
    "low": b.low, "close": b.close, "volume": b.volume,
} for b in bars])
df["dt"] = pd.to_datetime(df["dt"])
if df["dt"].dt.tz is None:
    df["dt"] = df["dt"].dt.tz_localize("UTC").dt.tz_convert(ET)
else:
    df["dt"] = df["dt"].dt.tz_convert(ET)

# Run strategy
strat = OvernightReversion(**OVERNIGHT_REVERSION_DEFAULTS)
sig_df = strat.generate_signals(df)

last = sig_df.iloc[-1]
last_5 = sig_df.tail(5)

print(f"Latest bar: {last['dt']} close=${last['close']:.2f}")
print(f"Current signal: {int(last['signal'])}")
print()
print("Last 5 bars:")
for _, row in last_5.iterrows():
    print(f"  {row['dt'].strftime('%H:%M')} close=${row['close']:.2f} signal={int(row['signal'])}")

# Compute z-score manually
import numpy as np
recent = sig_df.tail(90)
vwap_ish = recent["close"].mean()
std = recent["close"].std()
cur = float(last["close"])
z = (cur - vwap_ish) / std if std > 0 else 0
print()
print(f"Current price: ${cur:.2f}")
print(f"Recent mean (90 bars): ${vwap_ish:.2f}")
print(f"Recent std: ${std:.2f}")
print(f"Approx z-score: {z:+.2f} (signal fires at |z| > 1.5)")

ib.disconnect()
