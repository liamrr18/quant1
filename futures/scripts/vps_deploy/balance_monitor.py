#!/usr/bin/env python3
"""Balance & health monitor — runs on VPS every 5 minutes.

Pulls account stats from IB Gateway, checks systemd services, posts to Discord.
"""

import os
import subprocess
import time
import traceback
from datetime import datetime

import requests
import pytz
from ib_insync import IB, Future, Stock

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_BALANCE", "")
ET = pytz.timezone("America/New_York")

SERVICES = [
    ("ibgateway", "IB Gateway"),
    ("trader-futures", "Futures ORB+VWAP"),
    ("trader-overnight", "Overnight"),
    ("trader-equity-orb", "Equity ORB"),
    ("trader-opendrive", "OpenDrive"),
    ("trader-pairs", "Pairs"),
]

TRADE_LOGS = [
    "/root/futures_trader/logs/futures",
    "/root/flamboyant-lewin/logs",
    "/root/flamboyant-lewin/logs/opendrive",
    "/root/flamboyant-lewin/logs/pairs",
]


def post(payload):
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=10)
        return r.status_code
    except Exception as e:
        print(f"Discord post failed: {e}")
        return None


def service_status(name):
    r = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True, timeout=5)
    return r.stdout.strip()


def count_trades_today():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    total = 0
    for log_root in TRADE_LOGS:
        path = f"{log_root}/{today}/trades.csv"
        try:
            with open(path) as f:
                # Count lines excluding header and empty
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
                if lines and "timestamp" in lines[0]:
                    lines = lines[1:]
                # Only count 'entry' or 'exit' rows
                total += sum(1 for ln in lines if ",entry," in ln or ",exit," in ln)
        except FileNotFoundError:
            pass
        except Exception:
            pass
    return total


def get_ib_snapshot():
    ib = IB()
    ib.connect("127.0.0.1", 4002, clientId=20, timeout=15)
    try:
        summary = ib.accountSummary()
        equity = 0.0
        unreal = 0.0
        real = 0.0
        margin = 0.0
        for item in summary:
            if item.tag == "NetLiquidation":
                equity = float(item.value)
            elif item.tag == "UnrealizedPnL":
                unreal = float(item.value)
            elif item.tag == "RealizedPnL":
                real = float(item.value)
            elif item.tag == "AvailableFunds":
                margin = float(item.value)

        # Live P&L per position
        positions = []
        for item in ib.portfolio():
            if int(item.position) == 0:
                continue
            sym = item.contract.localSymbol or item.contract.symbol
            qty = int(item.position)
            upl = item.unrealizedPNL
            positions.append({
                "symbol": sym,
                "qty": qty,
                "side": "LONG" if qty > 0 else "SHORT",
                "price": item.marketPrice,
                "unrealized": upl,
            })

        return {
            "equity": equity,
            "unrealized": unreal,
            "realized": real,
            "margin": margin,
            "positions": positions,
        }
    finally:
        ib.disconnect()


def build_embed(snap, services, trades_today):
    total_pnl = snap["unrealized"] + snap["realized"]
    color = 0x2ECC71 if total_pnl >= 0 else 0xE74C3C  # green if up, red if down
    now = datetime.now(ET).strftime("%H:%M ET")

    # Service status block
    svc_lines = []
    any_down = False
    for name, label in services:
        status = service_status(name)
        ok = status == "active"
        if not ok:
            any_down = True
        svc_lines.append(f"{'\u2705' if ok else '\u274c'} {label}")
    svc_str = "\n".join(svc_lines)

    # Positions block
    pos_str = "Flat"
    if snap["positions"]:
        lines = []
        for p in snap["positions"]:
            pnl_sign = "+" if p["unrealized"] >= 0 else ""
            lines.append(f"{p['side']} {abs(p['qty'])} {p['symbol']} @ ${p['price']:,.2f} ({pnl_sign}${p['unrealized']:,.0f})")
        pos_str = "\n".join(lines)

    embed = {
        "title": f"\U0001f4b0 ${snap['equity']:,.0f}",
        "color": color,
        "fields": [
            {"name": "Today's P&L", "value": f"**${total_pnl:+,.0f}**\nReal: ${snap['realized']:+,.0f} | Unreal: ${snap['unrealized']:+,.0f}", "inline": True},
            {"name": "Margin", "value": f"${snap['margin']:,.0f}", "inline": True},
            {"name": "Trades", "value": f"{trades_today}", "inline": True},
            {"name": f"Positions ({len(snap['positions'])})", "value": pos_str, "inline": False},
            {"name": "Services", "value": svc_str, "inline": False},
        ],
        "footer": {"text": now},
    }
    return embed, any_down


def run_once():
    services = SERVICES
    trades_today = count_trades_today()

    # Check services first (doesn't need IB)
    down_services = []
    for name, label in services:
        if service_status(name) != "active":
            down_services.append(label)

    try:
        snap = get_ib_snapshot()
        embed, any_down = build_embed(snap, services, trades_today)
        post({"embeds": [embed]})
    except Exception as e:
        # IB connection failed — send alert instead
        msg = f"\u26a0\ufe0f **IB CONNECTION LOST** — monitor can't reach IB Gateway\n```\n{str(e)[:300]}\n```"
        post({"content": msg})
        traceback.print_exc()
        any_down = True

    # Separate alert for any DOWN services
    if down_services:
        alert = "\U0001f6a8 **SERVICE DOWN:** " + ", ".join(down_services) + " \u2014 check immediately"
        post({"content": alert})


def main():
    print("Balance monitor starting...")
    while True:
        try:
            run_once()
        except Exception:
            traceback.print_exc()
        time.sleep(300)  # 5 minutes


if __name__ == "__main__":
    main()
