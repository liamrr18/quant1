"""Test Discord alerts from both repos."""
import sys

# Test futures discord
sys.path.insert(0, "/root/futures_trader")
from trading.discord_alerts import discord_trade, discord_eod
discord_trade("AUDIT_Futures_VWAP", "MNQ", "long", "entry", 1, 26465.25)
discord_trade("AUDIT_Futures_VWAP", "MNQ", "long", "exit", 1, 26500.00, pnl=69.38,
              reason="test", entry_price=26465.25)
discord_eod({"AUDIT_Futures": {"daily_pnl": 69.38, "trades": 1}})
print("Futures alerts sent")

# Test equity discord
sys.path.pop(0)
sys.path.insert(0, "/root/flamboyant-lewin")
import importlib
if "trading.discord_alerts" in sys.modules:
    del sys.modules["trading.discord_alerts"]
if "trading" in sys.modules:
    del sys.modules["trading"]
from trading.discord_alerts import discord_trade as eq_trade, discord_eod as eq_eod
eq_trade("AUDIT_Equity_ORB", "SPY", "long", "entry", 150, 700.00)
eq_trade("AUDIT_Equity_ORB", "SPY", "long", "exit", 150, 702.00, pnl=300.00,
         reason="test", entry_price=700.00)
eq_eod({"AUDIT_Equity_ORB": {"daily_pnl": 300.00, "trades": 1}})
print("Equity alerts sent")
