"""Discord webhook alerts for trade notifications.

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
    """Send a message to Discord via webhook."""
    url = webhook or TRADE_WEBHOOK
    try:
        requests.post(url, json={"content": content}, timeout=5)
    except Exception as e:
        log.debug("Discord alert failed: %s", e)


def discord_trade(strategy: str, symbol: str, direction: str,
                  action: str, shares: int, price: float,
                  pnl: float = None, reason: str = "",
                  entry_price: float = None):
    """Send trade notification to the TRADE channel."""
    now = datetime.now(ET).strftime("%H:%M ET")

    if action == "entry":
        emoji = "\U0001f7e2" if direction == "long" else "\U0001f534"
        msg = (f"{emoji} **ENTRY** | **{strategy}** {direction.upper()} "
               f"{shares} {symbol} @ **${price:,.2f}**\n"
               f"`{now}`")
    else:
        if pnl is not None and pnl > 0:
            emoji = "\u2705"
            result = "WIN"
        elif pnl is not None and pnl < 0:
            emoji = "\u274c"
            result = "LOSS"
        else:
            emoji = "\u2796"
            result = "FLAT"

        pnl_str = f"**${pnl:+,.2f}**" if pnl is not None else "N/A"
        entry_str = f" (entry ${entry_price:,.2f})" if entry_price is not None else ""
        reason_str = f" | {reason}" if reason else ""

        msg = (f"{emoji} **{result}** | **{strategy}** {direction.upper()} "
               f"{shares} {symbol} @ ${price:,.2f}{entry_str}\n"
               f"P&L: {pnl_str}{reason_str}\n"
               f"`{now}`")

    _send(msg, TRADE_WEBHOOK)


def discord_halt(strategy: str, reason: str, daily_pnl: float):
    """Send halt notification to the TRADE channel."""
    msg = (f"\U000026A0 **HALTED: {strategy}**\n"
           f"Reason: {reason}\n"
           f"Daily P&L: ${daily_pnl:+.2f}\n"
           f"No further trades today.")
    _send(msg, TRADE_WEBHOOK)


def discord_eod(summaries: dict):
    """Send end-of-day summary to the SUMMARY channel."""
    now = datetime.now(ET).strftime("%Y-%m-%d")
    total_pnl = sum(s.get("daily_pnl", 0) for s in summaries.values())
    total_trades = sum(s.get("trades", 0) for s in summaries.values())

    emoji = "\u2705" if total_pnl >= 0 else "\u274c"

    lines = [f"\U0001f4ca **EOD Summary — {now}**", ""]
    for name, s in summaries.items():
        pnl = s.get("daily_pnl", 0)
        trades = s.get("trades", 0)
        tag = "\u2705" if pnl >= 0 else "\u274c"
        lines.append(f"{tag} **{name}**: ${pnl:+,.2f} ({trades} trades)")

    lines.append(f"\n**Total: ${total_pnl:+,.2f}** ({total_trades} trades)")
    _send("\n".join(lines), SUMMARY_WEBHOOK)
