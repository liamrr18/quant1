"""Email alerts for futures trading."""

import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import pytz

from trading.config import EMAIL_TO, EMAIL_FROM, GMAIL_APP_PASSWORD

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

_enabled = bool(GMAIL_APP_PASSWORD and EMAIL_TO and EMAIL_FROM)


def is_enabled() -> bool:
    return _enabled


def _send_email(subject: str, body: str):
    if not _enabled:
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_FROM, GMAIL_APP_PASSWORD)
            s.send_message(msg)
    except Exception as e:
        log.debug("Email alert failed: %s", e)


def alert_trade(strategy_name: str, symbol: str, direction: str, action: str,
                contracts: int, price: float, pnl: float = None, reason: str = None):
    now = datetime.now(ET).strftime("%H:%M ET")
    if action == "entry":
        subject = f"[FUTURES] {direction.upper()} {contracts} {symbol} @ ${price:.2f}"
        body = f"{strategy_name}: {direction} {contracts} {symbol} @ ${price:.2f} ({now})"
    else:
        pnl_str = f" P&L: ${pnl:+,.2f}" if pnl is not None else ""
        subject = f"[FUTURES] EXIT {symbol}{pnl_str}"
        body = f"{strategy_name}: exit {direction} {contracts} {symbol} @ ${price:.2f}{pnl_str} [{reason}] ({now})"
    _send_email(subject, body)


def alert_halt(strategy_name: str, reason: str, daily_pnl: float):
    subject = f"[FUTURES] {strategy_name} HALTED"
    body = f"Reason: {reason}\nDaily P&L: ${daily_pnl:+,.2f}"
    _send_email(subject, body)


def alert_eod_summary(summaries: dict):
    now = datetime.now(ET).strftime("%Y-%m-%d")
    lines = [f"Futures EOD Summary {now}", "=" * 40]
    total_pnl = 0
    for name, s in summaries.items():
        pnl = s.get("daily_pnl", 0)
        total_pnl += pnl
        lines.append(f"{name}: ${pnl:+,.2f} ({s.get('trades', 0)} trades)")
    lines.append(f"\nTotal: ${total_pnl:+,.2f}")
    _send_email(f"[FUTURES] EOD ${total_pnl:+,.0f}", "\n".join(lines))
