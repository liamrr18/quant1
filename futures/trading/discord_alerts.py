"""Discord webhook alerts for futures trading.

Two channels:
  TRADE_WEBHOOK  — every entry/exit/halt (real-time)
  SUMMARY_WEBHOOK — end-of-day summary per strategy
"""

import logging
import os
from datetime import datetime

import pytz
import requests

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

TRADE_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_TRADE", "")
SUMMARY_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_SUMMARY", "")


def _send(content: str, webhook: str = None):
    """Send a message to a Discord webhook."""
    url = webhook or TRADE_WEBHOOK
    try:
        requests.post(url, json={"content": content}, timeout=5)
    except Exception as e:
        log.debug("Discord alert failed: %s", e)


def discord_trade(strategy: str, symbol: str, direction: str, action: str,
                  contracts: int, price: float, pnl: float = None,
                  reason: str = None, entry_price: float = None):
    """Send trade notification to the TRADE channel."""
    now = datetime.now(ET).strftime("%H:%M ET")

    if action == "entry":
        emoji = "\U0001f7e2" if direction == "long" else "\U0001f534"
        msg = (f"{emoji} **ENTRY** | **{strategy}** {direction.upper()} "
               f"{contracts} {symbol} @ **${price:,.2f}**\n"
               f"`{now}`")
    else:
        # Determine win/loss from actual P&L
        if pnl is not None and pnl > 0:
            emoji = "\u2705"  # green check
            result = "WIN"
        elif pnl is not None and pnl < 0:
            emoji = "\u274c"  # red X
            result = "LOSS"
        else:
            emoji = "\u2796"  # minus sign
            result = "FLAT"

        pnl_str = f"**${pnl:+,.2f}**" if pnl is not None else "N/A"
        entry_str = f" (entry ${entry_price:,.2f})" if entry_price is not None else ""
        reason_str = f" | {reason}" if reason else ""

        msg = (f"{emoji} **{result}** | **{strategy}** {direction.upper()} "
               f"{contracts} {symbol} @ ${price:,.2f}{entry_str}\n"
               f"P&L: {pnl_str}{reason_str}\n"
               f"`{now}`")

    _send(msg, TRADE_WEBHOOK)


def discord_halt(strategy: str, reason: str, daily_pnl: float):
    """Send halt notification to the TRADE channel."""
    msg = f"\u26a0\ufe0f **[FUTURES] {strategy} HALTED** {reason} | Daily P&L: ${daily_pnl:+,.2f}"
    _send(msg, TRADE_WEBHOOK)


def discord_eod(summaries: dict):
    """Send end-of-day summary to the SUMMARY channel."""
    now = datetime.now(ET).strftime("%Y-%m-%d")
    lines = [f"\U0001f4ca **EOD Summary {now}**"]

    total_pnl = 0
    total_trades = 0
    for name, s in summaries.items():
        pnl = s.get("daily_pnl", 0)
        trades = s.get("trades", 0)
        total_pnl += pnl
        total_trades += trades
        emoji = "\u2705" if pnl >= 0 else "\u274c"
        lines.append(f"{emoji} **{name}**: ${pnl:+,.2f} ({trades} trades)")

    lines.append(f"\n**Total: ${total_pnl:+,.2f}** ({total_trades} trades)")
    _send("\n".join(lines), SUMMARY_WEBHOOK)
