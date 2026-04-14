"""Email alert system for trading events.

Sends email notifications for:
- Trade entries and exits
- Daily loss limit hit
- Strategy halt
- End-of-day summary

Setup: Set GMAIL_APP_PASSWORD in .env file.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import pytz
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

EMAIL_TO = os.getenv("ALERT_EMAIL", "liamrr18@gmail.com")
EMAIL_FROM = os.getenv("ALERT_EMAIL", "liamrr18@gmail.com")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

_enabled = bool(GMAIL_APP_PASSWORD)


def _send(subject: str, body: str):
    """Send an email via Gmail SMTP."""
    if not _enabled:
        log.debug("Alerts disabled (no GMAIL_APP_PASSWORD)")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, GMAIL_APP_PASSWORD)
            server.send_message(msg)

        log.info("Alert sent: %s", subject)
    except Exception as e:
        log.error("Failed to send alert: %s", e)


def alert_trade(strategy_name: str, symbol: str, direction: str,
                action: str, shares: int, price: float,
                pnl: float = None, reason: str = ""):
    """Alert on trade entry or exit."""
    now = datetime.now(ET).strftime("%H:%M ET")
    pnl_str = f"  PnL: ${pnl:+.2f}" if pnl is not None else ""

    subject = f"Trade: {action.upper()} {direction} {symbol} ({strategy_name})"
    body = (
        f"Trading Alert — {now}\n\n"
        f"Strategy: {strategy_name}\n"
        f"Action:   {action.upper()}\n"
        f"Symbol:   {symbol}\n"
        f"Direction:{direction}\n"
        f"Shares:   {shares}\n"
        f"Price:    ${price:.2f}\n"
        f"{pnl_str}\n"
        f"Reason:   {reason}\n"
    )
    _send(subject, body)


def alert_halt(strategy_name: str, reason: str, daily_pnl: float):
    """Alert when a strategy gets halted by risk manager."""
    now = datetime.now(ET).strftime("%H:%M ET")
    subject = f"HALT: {strategy_name} stopped trading"
    body = (
        f"Risk Alert — {now}\n\n"
        f"Strategy {strategy_name} has been HALTED.\n\n"
        f"Reason:    {reason}\n"
        f"Daily PnL: ${daily_pnl:+.2f}\n\n"
        f"No further trades will be placed today.\n"
        f"The strategy will resume tomorrow if conditions allow.\n"
    )
    _send(subject, body)


def alert_eod_summary(summaries: dict):
    """Alert with end-of-day summary across all strategies.

    Args:
        summaries: dict of {strategy_name: {pnl, trades, equity, ...}}
    """
    now = datetime.now(ET).strftime("%Y-%m-%d")
    total_pnl = sum(s.get("daily_pnl", 0) for s in summaries.values())
    total_trades = sum(s.get("trades", 0) for s in summaries.values())

    lines = [f"End of Day Report — {now}\n"]
    for name, s in summaries.items():
        pnl = s.get("daily_pnl", 0)
        trades = s.get("trades", 0)
        equity = s.get("final_equity", 0)
        lines.append(f"  {name}:")
        lines.append(f"    PnL: ${pnl:+.2f}  |  Trades: {trades}  |  Equity: ${equity:,.2f}")
        if s.get("halted"):
            lines.append(f"    STATUS: HALTED ({s.get('halt_reason', '')})")
        lines.append("")

    lines.append(f"  TOTAL PnL:    ${total_pnl:+.2f}")
    lines.append(f"  TOTAL TRADES: {total_trades}")

    emoji = "+" if total_pnl >= 0 else ""
    subject = f"EOD: ${total_pnl:+.2f} ({total_trades} trades) — {now}"
    _send(subject, "\n".join(lines))


def is_enabled() -> bool:
    """Check if email alerts are configured."""
    return _enabled
