"""Discord webhook alerts for trade notifications.

Sends a message to Discord on every trade entry, exit, halt, and EOD summary.
"""

import logging
from datetime import datetime

import pytz
import requests

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

WEBHOOK_URL = "https://discord.com/api/webhooks/1493688011102359645/oU2HLGifmBBVsYu4CXBHII7DZg6I8uYgKoeF8VidhEbqKqiEs9Go79Bocp36PxCF_hKh"


def _send(content: str):
    """Send a message to Discord via webhook."""
    try:
        requests.post(WEBHOOK_URL, json={"content": content}, timeout=5)
    except Exception as e:
        log.debug("Discord alert failed: %s", e)


def discord_trade(strategy: str, symbol: str, direction: str,
                  action: str, shares: int, price: float,
                  pnl: float = None, reason: str = ""):
    """Send trade notification."""
    now = datetime.now(ET).strftime("%H:%M")

    if action == "entry":
        emoji = "\U0001f7e2" if direction == "long" else "\U0001f534"  # green/red circle
        msg = (f"{emoji} **{action.upper()}** {direction.upper()} "
               f"**{symbol}** {shares} shares @ ${price:.2f}\n"
               f"`{strategy} | {now} ET`")
    else:
        if pnl is not None and pnl > 0:
            emoji = "\U0001f4b0"  # money bag
        elif pnl is not None and pnl < 0:
            emoji = "\U0001f4c9"  # chart down
        else:
            emoji = "\u2B1C"  # white square
        pnl_str = f" | P&L: **${pnl:+.2f}**" if pnl is not None else ""
        msg = (f"{emoji} **{action.upper()}** {direction.upper()} "
               f"**{symbol}** {shares} shares @ ${price:.2f}{pnl_str}\n"
               f"`{strategy} | {reason} | {now} ET`")

    _send(msg)


def discord_halt(strategy: str, reason: str, daily_pnl: float):
    """Send halt notification."""
    msg = (f"\U000026A0 **HALTED: {strategy}**\n"
           f"Reason: {reason}\n"
           f"Daily P&L: ${daily_pnl:+.2f}\n"
           f"No further trades today.")
    _send(msg)


def discord_eod(summaries: dict):
    """Send end-of-day summary."""
    now = datetime.now(ET).strftime("%Y-%m-%d")
    total_pnl = sum(s.get("daily_pnl", 0) for s in summaries.values())
    total_trades = sum(s.get("trades", 0) for s in summaries.values())

    emoji = "\U0001f4c8" if total_pnl >= 0 else "\U0001f4c9"

    lines = [f"{emoji} **End of Day — {now}**", ""]
    for name, s in summaries.items():
        pnl = s.get("daily_pnl", 0)
        trades = s.get("trades", 0)
        tag = "\u2705" if pnl >= 0 else "\u274C"
        lines.append(f"{tag} **{name}**: ${pnl:+.2f} ({trades} trades)")

    lines.append(f"\n**Total: ${total_pnl:+.2f}** ({total_trades} trades)")
    _send("\n".join(lines))
