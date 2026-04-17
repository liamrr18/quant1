from ib_insync import IB, MarketOrder, Future, Stock
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=97, timeout=10)
for item in ib.accountSummary():
    if item.tag == "NetLiquidation":
        print(f"Equity: ${float(item.value):,.2f}")
    elif item.tag == "RealizedPnL":
        print(f"Realized today: ${float(item.value):+,.2f}")
    elif item.tag == "UnrealizedPnL":
        print(f"Unrealized: ${float(item.value):+,.2f}")

for p in ib.positions():
    if int(p.position) != 0:
        print(f"Position: {p.contract.localSymbol}: {int(p.position)} @ avgCost {p.avgCost:.2f}")

ib.reqGlobalCancel()
ib.sleep(2)

for p in ib.positions():
    if int(p.position) != 0:
        qty = abs(int(p.position))
        side = "SELL" if p.position > 0 else "BUY"
        c = p.contract
        if isinstance(c, Future):
            c.exchange = c.exchange or ("COMEX" if "MGC" in (c.symbol or "") else "CME")
        else:
            c.exchange = "SMART"
        ib.qualifyContracts(c)
        trade = ib.placeOrder(c, MarketOrder(side, qty))
        ib.sleep(5)
        print(f"Closed {side} {qty} {c.localSymbol}: {trade.orderStatus.status}")

ib.sleep(3)
rem = [p for p in ib.positions() if int(p.position) != 0]
print(f"Remaining: {len(rem)}")
ib.disconnect()
