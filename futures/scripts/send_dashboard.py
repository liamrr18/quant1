#!/usr/bin/env python3
"""Send pinnable trading system dashboard to Discord."""

import os
import requests
import time

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_PORTFOLIO", "")

# Message 1: Summary + Futures strategies
msg1 = {
    "embeds": [
        {
            "title": "\U0001f4ca TRADING SYSTEM DASHBOARD",
            "description": (
                "**7 strategies** across **9 instruments** on Interactive Brokers\n"
                "All paper trading live \u2014 April 2026"
            ),
            "color": 0x2ECC71,
            "fields": [
                {"name": "Account Equity", "value": "$1,001,877", "inline": True},
                {"name": "Est. Trades/Day", "value": "5\u201310", "inline": True},
                {"name": "Active Hours", "value": "8 PM \u2192 4 PM ET\n(20 of 24 hours)", "inline": True},
                {"name": "Combined OOS Sharpe", "value": "~4.0", "inline": True},
                {"name": "Processes", "value": "5 (clientIds 1,3,10,11,12)", "inline": True},
                {"name": "\u26a0\ufe0f Panic Button", "value": "Double-click `EMERGENCY_STOP.bat` on Desktop", "inline": True},
            ],
        },
        {
            "title": "1\ufe0f\u20e3  Futures ORB Breakout",
            "description": "Buys/shorts when price breaks above/below the first 15-minute range on S&P and Nasdaq futures.",
            "color": 0x3498DB,
            "fields": [
                {"name": "Symbols", "value": "MES, MNQ", "inline": True},
                {"name": "Entry Window", "value": "9:45 AM \u2013 3:30 PM ET", "inline": True},
                {"name": "Flat By", "value": "3:50 PM ET", "inline": True},
                {"name": "OOS Sharpe", "value": "MES 0.98 | MNQ 1.56", "inline": True},
                {"name": "OOS Win Rate", "value": "~52%", "inline": True},
                {"name": "Trades/Day", "value": "0\u20131 per symbol", "inline": True},
                {"name": "Risk", "value": "1%/trade | 5% daily limit", "inline": True},
                {"name": "ClientId", "value": "1", "inline": True},
                {"name": "Days", "value": "Mon\u2013Fri", "inline": True},
                {"name": "Logs", "value": "`futures_trader/logs/futures/<date>/trader.log`", "inline": False},
                {"name": "Restart", "value": "`cd futures_trader && python run_futures.py`", "inline": False},
            ],
        },
        {
            "title": "2\ufe0f\u20e3  Futures VWAP Reversion",
            "description": "Fades extended moves from VWAP during mid-day chop. Enters when price deviates >1 std dev, exits on reversion.",
            "color": 0x3498DB,
            "fields": [
                {"name": "Symbols", "value": "MES, MNQ, **MGC (gold)**", "inline": True},
                {"name": "Entry Window", "value": "10:00 AM \u2013 3:00 PM ET", "inline": True},
                {"name": "Flat By", "value": "3:25 PM ET", "inline": True},
                {"name": "OOS Sharpe", "value": "MES 4.08 | MNQ 4.31 | MGC 3.46", "inline": True},
                {"name": "OOS Win Rate", "value": "MES 80% | MNQ 83% | MGC 67%", "inline": True},
                {"name": "Trades/Day", "value": "2\u20134 total", "inline": True},
                {"name": "Risk", "value": "1%/trade | 5% daily limit", "inline": True},
                {"name": "ClientId", "value": "1 (same process as ORB)", "inline": True},
                {"name": "Days", "value": "Mon\u2013Fri", "inline": True},
                {"name": "Logs", "value": "`futures_trader/logs/futures/<date>/trader.log`", "inline": False},
                {"name": "Restart", "value": "`cd futures_trader && python run_futures.py`", "inline": False},
            ],
        },
        {
            "title": "3\ufe0f\u20e3  Overnight Reversion",
            "description": "Mean reversion during dead overnight hours when liquidity is thin and prices tend to snap back.",
            "color": 0x3498DB,
            "fields": [
                {"name": "Symbols", "value": "MNQ", "inline": True},
                {"name": "Entry Window", "value": "8:00 PM \u2013 2:00 AM ET", "inline": True},
                {"name": "Flat By", "value": "9:25 AM ET", "inline": True},
                {"name": "OOS Sharpe", "value": "1.65", "inline": True},
                {"name": "OOS Win Rate", "value": "~58%", "inline": True},
                {"name": "Trades/Night", "value": "0\u20131", "inline": True},
                {"name": "Risk", "value": "1 contract max", "inline": True},
                {"name": "ClientId", "value": "3", "inline": True},
                {"name": "Days", "value": "Sun night \u2013 Fri morning", "inline": True},
                {"name": "Logs", "value": "`futures_trader/logs/futures/<date>/overnight.log`", "inline": False},
                {"name": "Restart", "value": "`cd futures_trader && python run_overnight.py`", "inline": False},
            ],
        },
    ]
}

# Message 2: Equity strategies + Operations
msg2 = {
    "embeds": [
        {
            "title": "4\ufe0f\u20e3  Equity ORB Breakout",
            "description": "Same ORB strategy as futures but on SPY and QQQ equity ETFs via IB stock orders.",
            "color": 0x3498DB,
            "fields": [
                {"name": "Symbols", "value": "SPY, QQQ", "inline": True},
                {"name": "Entry Window", "value": "9:45 AM \u2013 3:55 PM ET", "inline": True},
                {"name": "Flat By", "value": "3:50 PM ET", "inline": True},
                {"name": "OOS Sharpe", "value": "SPY 0.98 | QQQ 4.20", "inline": True},
                {"name": "OOS Win Rate", "value": "SPY ~50% | QQQ ~58%", "inline": True},
                {"name": "Trades/Day", "value": "0\u20131 per symbol", "inline": True},
                {"name": "Risk", "value": "30% max position | 2% daily limit", "inline": True},
                {"name": "ClientId", "value": "10", "inline": True},
                {"name": "Days", "value": "Mon\u2013Fri", "inline": True},
                {"name": "Logs", "value": "`spy-trader/.../logs/<date>/trader.log`", "inline": False},
                {"name": "Restart", "value": "`cd spy-trader/.../flamboyant-lewin && python run_live.py`", "inline": False},
            ],
        },
        {
            "title": "5\ufe0f\u20e3  Opening Drive",
            "description": "Rides strong first-5-minute momentum on semiconductor and tech sector ETFs.",
            "color": 0x3498DB,
            "fields": [
                {"name": "Symbols", "value": "SMH, XLK", "inline": True},
                {"name": "Entry Window", "value": "9:35 AM \u2013 12:00 PM ET", "inline": True},
                {"name": "Flat By", "value": "3:50 PM ET", "inline": True},
                {"name": "OOS Sharpe", "value": "SMH 3.87 | XLK 3.26", "inline": True},
                {"name": "OOS Win Rate", "value": "~55%", "inline": True},
                {"name": "Trades/Day", "value": "0\u20131 per symbol", "inline": True},
                {"name": "Risk", "value": "20% max position | 1.5% daily limit", "inline": True},
                {"name": "ClientId", "value": "11", "inline": True},
                {"name": "Days", "value": "Mon\u2013Fri", "inline": True},
                {"name": "Logs", "value": "`spy-trader/.../logs/opendrive/<date>/`", "inline": False},
                {"name": "Restart", "value": "`cd spy-trader/.../flamboyant-lewin && python run_opendrive.py`", "inline": False},
            ],
        },
        {
            "title": "6\ufe0f\u20e3  Pairs Spread",
            "description": "Trades the GLD/TLT spread \u2014 goes long gold vs treasuries when the z-score is extreme, exits on mean reversion.",
            "color": 0x3498DB,
            "fields": [
                {"name": "Symbols", "value": "GLD (traded) vs TLT (signal)", "inline": True},
                {"name": "Entry Window", "value": "9:45 AM \u2013 3:00 PM ET", "inline": True},
                {"name": "Flat By", "value": "3:50 PM ET", "inline": True},
                {"name": "OOS Sharpe", "value": "4.86", "inline": True},
                {"name": "OOS Win Rate", "value": "~60%", "inline": True},
                {"name": "Trades/Day", "value": "1\u20132", "inline": True},
                {"name": "Risk", "value": "20% max position | 1.5% daily limit", "inline": True},
                {"name": "ClientId", "value": "12", "inline": True},
                {"name": "Days", "value": "Mon\u2013Fri", "inline": True},
                {"name": "Logs", "value": "`spy-trader/.../logs/pairs/<date>/`", "inline": False},
                {"name": "Restart", "value": "`cd spy-trader/.../flamboyant-lewin && python run_pairs.py`", "inline": False},
            ],
        },
        {
            "title": "\U0001f6e0\ufe0f  Operations",
            "description": "How to manage the system",
            "color": 0x95A5A6,
            "fields": [
                {"name": "Restart All", "value": "Double-click `start_futures_trading.bat` on Desktop", "inline": False},
                {"name": "Auto-Start", "value": "Task Scheduler \u2192 `StartTradingSystems` fires on login", "inline": False},
                {"name": "Emergency Stop", "value": "Double-click `EMERGENCY_STOP.bat` on Desktop\nKills all processes + closes all IB positions", "inline": False},
                {"name": "Check Status", "value": "```tasklist | findstr python```", "inline": False},
                {"name": "Discord Alerts", "value": "#every-trade \u2014 real-time entries/exits with WIN/LOSS and P&L\n#summaries \u2014 end-of-day per-strategy recap", "inline": False},
            ],
        },
    ]
}

r1 = requests.post(WEBHOOK, json=msg1, timeout=15)
print(f"Message 1 (Summary + Futures): HTTP {r1.status_code}")
if r1.status_code != 204:
    print(r1.text[:300])

time.sleep(1.5)

r2 = requests.post(WEBHOOK, json=msg2, timeout=15)
print(f"Message 2 (Equities + Ops): HTTP {r2.status_code}")
if r2.status_code != 204:
    print(r2.text[:300])

if r1.status_code == 204 and r2.status_code == 204:
    print("\nBoth messages sent successfully. Pin them in Discord.")
