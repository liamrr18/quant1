"""Live audit: fetch bars for every instrument, print results."""
import sys
sys.path.insert(0, "/root/futures_trader")
import pandas as pd
import pytz
from ib_insync import IB, Future, Stock

ET = pytz.timezone("America/New_York")
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=94, timeout=15)


def test_instrument(name, contract, use_rth):
    try:
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="2 D",
            barSizeSetting="1 min", whatToShow="TRADES",
            useRTH=use_rth, formatDate=1,
        )
        if not bars:
            print(f"  {name}: NO BARS")
            return
        first = bars[0].date
        last = bars[-1].date
        print(f"  {name}: {len(bars)} bars | first={first} | last={last} | "
              f"close=${bars[-1].close:.2f} | RTH={use_rth}")
    except Exception as e:
        print(f"  {name}: ERROR {e}")


print("=== FUTURES (useRTH=True for cash strategies) ===")
for sym in ["MES", "MNQ"]:
    c = Future(symbol=sym, exchange="CME")
    d = ib.reqContractDetails(c); d.sort(key=lambda x: x.contract.lastTradeDateOrContractMonth)
    test_instrument(f"{sym} (CME)", d[0].contract, use_rth=True)

c = Future(symbol="MGC", exchange="COMEX")
d = ib.reqContractDetails(c); d.sort(key=lambda x: x.contract.lastTradeDateOrContractMonth)
test_instrument("MGC (COMEX)", d[0].contract, use_rth=True)

print()
print("=== OVERNIGHT (useRTH=False) ===")
c = Future(symbol="MNQ", exchange="CME")
d = ib.reqContractDetails(c); d.sort(key=lambda x: x.contract.lastTradeDateOrContractMonth)
test_instrument("MNQ overnight", d[0].contract, use_rth=False)

print()
print("=== STOCKS (useRTH=True) ===")
for sym in ["SPY", "QQQ", "SMH", "XLK", "GLD", "TLT"]:
    test_instrument(sym, Stock(sym, "SMART", "USD"), use_rth=True)

ib.disconnect()
