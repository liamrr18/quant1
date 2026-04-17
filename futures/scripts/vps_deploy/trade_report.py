#!/usr/bin/env python3
"""One-time trade report: all trades today, classified as bug or legit."""

import csv
import glob
import os
from datetime import datetime, timedelta, timezone

import pytz
import requests
from ib_insync import IB, Future, Stock

ET = pytz.timezone("America/New_York")
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_PORTFOLIO", "")

TODAY = datetime.now(ET).date()
YEST = TODAY - timedelta(days=1)

# ─── Pull IB executions for today ────────────────────────────────────────
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=20, timeout=15)

# Account equity
equity = 0.0
for item in ib.accountSummary():
    if item.tag == "NetLiquidation":
        equity = float(item.value)
        break

# Get all fills from today
from ib_insync import ExecutionFilter
ef = ExecutionFilter()
ef.time = TODAY.strftime("%Y%m%d") + "-00:00:00"
fills = ib.reqExecutions(ef)

ib_trades = []
for f in fills:
    try:
        t = f.time
        if isinstance(t, str):
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        t_et = t.astimezone(ET)
        if t_et.date() != TODAY:
            continue
        ib_trades.append({
            "time": t_et,
            "symbol": f.contract.localSymbol or f.contract.symbol,
            "side": f.execution.side,  # BOT or SLD
            "shares": int(f.execution.shares),
            "price": float(f.execution.price),
            "order_id": f.execution.orderId,
        })
    except Exception:
        pass
ib.disconnect()

ib_trades.sort(key=lambda x: x["time"])

# ─── Classify each trade ────────────────────────────────────────────────
# Group by symbol to detect churn patterns
by_sym = {}
for t in ib_trades:
    by_sym.setdefault(t["symbol"], []).append(t)

# Churn detection: >5 trades in a rolling 10-minute window on same symbol
def is_churn_trade(t, all_same_symbol):
    # If this trade is part of a cluster of 6+ on the same symbol within 30 min
    cluster = [x for x in all_same_symbol if abs((x["time"] - t["time"]).total_seconds()) < 1800]
    return len(cluster) >= 6

# Oversized MGC: any single fill of >10 contracts MGC = the 30-contract bug
def is_mgc_oversized(t):
    return "MGC" in t["symbol"] and t["shares"] >= 10

# Build entry/exit pairs per symbol per direction
def pair_trades(trades):
    """Pair BOT/SLD into round trips per symbol."""
    results = []
    open_positions = []  # list of (trade, side)
    for t in trades:
        # Simple FIFO pairing
        if t["side"] == "BOT":
            # Check if this closes a short
            closed = None
            for i, op in enumerate(open_positions):
                if op["side"] == "SLD" and op["shares"] == t["shares"]:
                    closed = open_positions.pop(i)
                    break
            if closed:
                pnl = (closed["price"] - t["price"]) * t["shares"]
                if _is_future(t["symbol"]):
                    pnl *= _multiplier(t["symbol"])
                results.append({
                    "entry_time": closed["time"], "exit_time": t["time"],
                    "symbol": t["symbol"], "direction": "short",
                    "shares": t["shares"], "entry_price": closed["price"],
                    "exit_price": t["price"], "pnl": pnl,
                    "entry_order": closed["order_id"], "exit_order": t["order_id"],
                })
            else:
                open_positions.append(t)
        else:  # SLD
            closed = None
            for i, op in enumerate(open_positions):
                if op["side"] == "BOT" and op["shares"] == t["shares"]:
                    closed = open_positions.pop(i)
                    break
            if closed:
                pnl = (t["price"] - closed["price"]) * t["shares"]
                if _is_future(t["symbol"]):
                    pnl *= _multiplier(t["symbol"])
                results.append({
                    "entry_time": closed["time"], "exit_time": t["time"],
                    "symbol": t["symbol"], "direction": "long",
                    "shares": t["shares"], "entry_price": closed["price"],
                    "exit_price": t["price"], "pnl": pnl,
                    "entry_order": closed["order_id"], "exit_order": t["order_id"],
                })
            else:
                open_positions.append(t)
    return results, open_positions


def _is_future(sym):
    return any(x in sym for x in ["MES", "MNQ", "MGC", "MCL", "M2K", "MBT"])


def _multiplier(sym):
    return {"MES": 5, "MNQ": 2, "MGC": 10}.get(sym[:3], 1)


# Pair trades per symbol
all_round_trips = []
still_open = []
for sym, trs in by_sym.items():
    rts, op = pair_trades(trs)
    all_round_trips.extend(rts)
    still_open.extend(op)

all_round_trips.sort(key=lambda x: x["entry_time"])

# Classify each round trip
def classify(rt, symbol_trades):
    reasons = []
    hold_sec = (rt["exit_time"] - rt["entry_time"]).total_seconds()

    # XLK churn
    if rt["symbol"] == "XLK":
        reasons.append("XLK churn (broken hard_tp bug)")

    # MGC oversized
    if "MGC" in rt["symbol"] and rt["shares"] >= 10:
        reasons.append(f"MGC oversized ({rt['shares']} contracts)")

    # Fast round trip with same-side pattern (<2min AND no strategy reason = churn)
    # If <2 min hold and there are many such rapid trades on this symbol, it's churn
    if hold_sec < 120:
        rapid_count = sum(1 for r in symbol_trades
                         if (r["exit_time"] - r["entry_time"]).total_seconds() < 120
                         and r != rt)
        if rapid_count >= 3:
            reasons.append("rapid-fire churn (<2min holds)")

    return reasons


sym_round_trips = {}
for rt in all_round_trips:
    sym_round_trips.setdefault(rt["symbol"], []).append(rt)

legit_trades = []
bug_trades = []
for rt in all_round_trips:
    bug_reasons = classify(rt, sym_round_trips[rt["symbol"]])
    rt["bug_reasons"] = bug_reasons
    if bug_reasons:
        bug_trades.append(rt)
    else:
        legit_trades.append(rt)

# ─── Build Discord embeds ────────────────────────────────────────────────
def fmt_row(rt):
    win_loss = "WIN" if rt["pnl"] > 0 else ("LOSS" if rt["pnl"] < 0 else "FLAT")
    emoji = "\u2705" if rt["pnl"] > 0 else "\u274c" if rt["pnl"] < 0 else "\u2796"
    if rt["bug_reasons"]:
        emoji = "\u26a0\ufe0f"
    t = rt["entry_time"].strftime("%H:%M")
    return f"{emoji} `{t}` {rt['direction'][:1].upper()} {rt['shares']} {rt['symbol']} ${rt['entry_price']:.2f}\u2192${rt['exit_price']:.2f} | **${rt['pnl']:+.0f}**"


def chunk_rows(rows, max_chars=800):
    chunks, cur, cur_len = [], [], 0
    for r in rows:
        if cur_len + len(r) + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(r)
        cur_len += len(r) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


legit_pnl = sum(rt["pnl"] for rt in legit_trades)
bug_pnl = sum(rt["pnl"] for rt in bug_trades)
total_pnl = legit_pnl + bug_pnl
legit_wins = sum(1 for rt in legit_trades if rt["pnl"] > 0)
legit_wr = legit_wins / len(legit_trades) * 100 if legit_trades else 0

embeds = []

# Header
embeds.append({
    "title": f"\U0001f4cb Trade Report \u2014 {TODAY.strftime('%b %d, %Y')}",
    "color": 0x2ECC71 if total_pnl >= 0 else 0xE74C3C,
    "fields": [
        {"name": "Total trades", "value": f"{len(all_round_trips)}", "inline": True},
        {"name": "Legit trades", "value": f"{len(legit_trades)} ({legit_wr:.0f}% WR)", "inline": True},
        {"name": "Bug trades", "value": f"{len(bug_trades)}", "inline": True},
        {"name": "Legit P&L", "value": f"${legit_pnl:+,.0f}", "inline": True},
        {"name": "Bug P&L", "value": f"${bug_pnl:+,.0f}", "inline": True},
        {"name": "Total P&L", "value": f"**${total_pnl:+,.0f}**", "inline": True},
        {"name": "Equity", "value": f"${equity:,.0f}", "inline": True},
        {"name": "Net if no bugs", "value": f"${equity - bug_pnl:,.0f}", "inline": True},
        {"name": "Still open", "value": f"{len(still_open)}", "inline": True},
    ],
})

# Legit trades embeds
if legit_trades:
    rows = [fmt_row(rt) for rt in legit_trades]
    chunks = chunk_rows(rows)
    for i, c in enumerate(chunks):
        embeds.append({
            "title": f"\u2705 Legit Trades ({len(legit_trades)})" + (f" - pt {i+1}" if len(chunks) > 1 else ""),
            "color": 0x2ECC71,
            "description": c,
        })

# Bug trades embeds
if bug_trades:
    rows = []
    for rt in bug_trades:
        base = fmt_row(rt)
        rows.append(f"{base}\n    \U0001f41b {', '.join(rt['bug_reasons'])}")
    chunks = chunk_rows(rows)
    for i, c in enumerate(chunks):
        embeds.append({
            "title": f"\u26a0\ufe0f Bug Trades ({len(bug_trades)})" + (f" - pt {i+1}" if len(chunks) > 1 else ""),
            "color": 0xF1C40F,
            "description": c,
        })

# Send one embed at a time to stay under 6000 char per message limit
import time as _t
for i, e in enumerate(embeds):
    r = requests.post(WEBHOOK, json={"embeds": [e]}, timeout=15)
    print(f"Sent embed {i+1}/{len(embeds)}: HTTP {r.status_code}")
    if r.status_code != 204:
        print(r.text[:300])
    _t.sleep(0.6)  # Discord rate limit
