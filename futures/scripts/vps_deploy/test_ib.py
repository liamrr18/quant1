from ib_insync import IB
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=99, timeout=15)
acct = ib.accountSummary()
for item in acct:
    if item.tag == "NetLiquidation":
        print(f"Equity: ${item.value}")
        break
positions = [p for p in ib.positions() if int(p.position) != 0]
print(f"Open positions: {len(positions)}")
for p in positions:
    print(f"  {p.contract.localSymbol or p.contract.symbol}: {p.position}")
ib.disconnect()
print("API OK")
