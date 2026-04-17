from ib_insync import IB
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=95, timeout=10)
for p in ib.positions():
    if int(p.position) != 0:
        print(f"{p.contract.localSymbol}: {int(p.position)} @ avg {p.avgCost:.2f}")
for item in ib.accountSummary():
    if item.tag == "NetLiquidation":
        print(f"Equity: ${float(item.value):,.2f}")
ib.disconnect()
