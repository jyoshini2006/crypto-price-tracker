"""Stage 8 - filtering and alerts.

Three rules are evaluated on every batch:

* **threshold** - a coin closes above the configured level (fires on the
  crossing, and on the very first observation, so it does not spam every run),
* **movers** - the top ``alerts.top_movers`` gainers and losers by 24h change,
* **spike** - the price moved more than ``alerts.spike_pct`` since the previous
  stored run.

Delivery goes to Telegram and/or email. ``alerts.dry_run: true`` (the default)
logs everything and sends nothing, which is the safe way to run a demo.
Every alert is also written to ``logs/alerts.log`` and the SQLite ``alerts``
table so the dashboard has an alert log to render.
"""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Iterable, Mapping

import pandas as pd

from cleaner import utc_timestamp

LOGGER = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


@dataclass
class Alert:
    kind: str
    message: str
    symbol: str | None = None
    severity: str = "info"
    delivered: bool = False
    channel: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class AlertEngine:
    """Evaluate alert rules and deliver the results."""

    def __init__(
        self,
        cfg: Any,
        storage: Any | None = None,
        logger: logging.Logger | None = None,
        session: Any | None = None,
        smtp_factory: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.storage = storage
        self.log = logger or LOGGER
        self.session = session
        self.smtp_factory = smtp_factory or smtplib.SMTP

        self.enabled: bool = bool(cfg.get("alerts.enabled", True))
        self.dry_run: bool = bool(cfg.get("alerts.dry_run", True))
        self.thresholds: dict[str, float] = {
            str(symbol).upper(): float(level)
            for symbol, level in (cfg.get("alerts.price_thresholds", {}) or {}).items()
        }
        self.top_movers: int = max(0, int(cfg.get("alerts.top_movers", 3)))
        self.spike_pct: float = float(cfg.get("alerts.spike_pct", 5.0))
        self.channels: dict[str, Any] = cfg.section("alerts").get("channels", {}) or {}
        self.alert_file = None
        if cfg.get("logging.alert_file"):
            self.alert_file = cfg.path("logging.dir") / str(cfg.get("logging.alert_file"))

    # -- rule evaluation --------------------------------------------------
    def evaluate(
        self,
        frame: pd.DataFrame,
        previous_prices: Mapping[str, float] | None = None,
    ) -> list[Alert]:
        if frame is None or frame.empty:
            return []
        previous_prices = {str(k).upper(): float(v) for k, v in (previous_prices or {}).items()}
        alerts: list[Alert] = []
        alerts.extend(self._threshold_alerts(frame, previous_prices))
        alerts.extend(self._spike_alerts(frame, previous_prices))
        alerts.extend(self._mover_alerts(frame))
        return alerts

    def _threshold_alerts(self, frame: pd.DataFrame, previous_prices: Mapping[str, float]) -> list[Alert]:
        alerts: list[Alert] = []
        for row in frame.itertuples(index=False):
            level = self.thresholds.get(str(row.symbol).upper())
            if level is None:
                continue
            price = float(row.price)
            previous = previous_prices.get(str(row.symbol).upper())
            crossed = previous is None or previous < level
            if price >= level and crossed:
                alerts.append(
                    Alert(
                        kind="threshold",
                        symbol=str(row.symbol),
                        severity="warning",
                        message=(
                            f"{row.symbol} is at ${price:,.2f}, at or above the "
                            f"${level:,.2f} threshold"
                            + ("" if previous is None else f" (previous ${previous:,.2f})")
                        ),
                        meta={"price": price, "threshold": level, "previous": previous},
                    )
                )
        return alerts

    def _spike_alerts(self, frame: pd.DataFrame, previous_prices: Mapping[str, float]) -> list[Alert]:
        if self.spike_pct <= 0:
            return []
        alerts: list[Alert] = []
        for row in frame.itertuples(index=False):
            symbol = str(row.symbol).upper()
            previous = previous_prices.get(symbol)
            if not previous or previous <= 0:
                continue
            price = float(row.price)
            move = (price - previous) / previous * 100.0
            if abs(move) >= self.spike_pct:
                direction = "up" if move > 0 else "down"
                alerts.append(
                    Alert(
                        kind="spike",
                        symbol=symbol,
                        severity="warning",
                        message=(
                            f"{symbol} spiked {direction} {move:+.2f}% since the previous run "
                            f"(${previous:,.4f} -> ${price:,.4f}, threshold {self.spike_pct}%)"
                        ),
                        meta={"price": price, "previous": previous, "move_pct": move},
                    )
                )
        return alerts

    def _mover_alerts(self, frame: pd.DataFrame) -> list[Alert]:
        if self.top_movers <= 0:
            return []
        changes = pd.to_numeric(frame["change_24h"], errors="coerce")
        ranked = frame.assign(_change=changes).dropna(subset=["_change"])
        if ranked.empty:
            return []

        alerts: list[Alert] = []
        gainers = ranked.nlargest(self.top_movers, "_change")
        losers = ranked.nsmallest(self.top_movers, "_change")

        def describe(subset: pd.DataFrame) -> str:
            return ", ".join(
                f"{symbol} {float(change):+.2f}%"
                for symbol, change in zip(subset["symbol"], subset["_change"])
            )

        alerts.append(
            Alert(
                kind="movers",
                severity="info",
                message="Top gainers (24h): " + describe(gainers),
                meta={"side": "gainers", "symbols": gainers["symbol"].astype(str).tolist()},
            )
        )
        alerts.append(
            Alert(
                kind="movers",
                severity="info",
                message="Top losers (24h): " + describe(losers),
                meta={"side": "losers", "symbols": losers["symbol"].astype(str).tolist()},
            )
        )
        return alerts

    # -- delivery ---------------------------------------------------------
    def format_message(self, alerts: Iterable[Alert]) -> str:
        stamp = utc_timestamp()
        lines = [f"Crypto tracker alerts - {stamp}"]
        lines.extend(f"[{alert.severity.upper()}] {alert.message}" for alert in alerts)
        return "\n".join(lines)

    def _send_telegram(self, text: str) -> tuple[bool, str]:
        settings = self.channels.get("telegram", {}) or {}
        token = os.environ.get(str(settings.get("bot_token_env", "TELEGRAM_BOT_TOKEN")), "").strip()
        chat_id = os.environ.get(str(settings.get("chat_id_env", "TELEGRAM_CHAT_ID")), "").strip()
        if not token or not chat_id:
            return False, "telegram credentials missing"
        if self.session is None:  # pragma: no cover - imported lazily for tests
            import requests

            self.session = requests.Session()
        response = self.session.post(
            f"{TELEGRAM_API}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=15,
        )
        response.raise_for_status()
        return True, "telegram"

    def _send_email(self, text: str) -> tuple[bool, str]:
        settings = self.channels.get("email", {}) or {}
        username = os.environ.get(str(settings.get("username_env", "SMTP_USERNAME")), "").strip()
        password = os.environ.get(str(settings.get("password_env", "SMTP_PASSWORD")), "").strip()
        sender = os.environ.get(str(settings.get("from_env", "SMTP_FROM")), "").strip() or username
        recipient = os.environ.get(str(settings.get("to_env", "SMTP_TO")), "").strip()
        if not username or not password or not recipient:
            return False, "smtp credentials missing"

        message = EmailMessage()
        message["Subject"] = "Crypto tracker alert"
        message["From"] = sender
        message["To"] = recipient
        message.set_content(text)

        host = str(settings.get("smtp_host", "smtp.gmail.com"))
        port = int(settings.get("smtp_port", 587))
        with self.smtp_factory(host, port) as server:
            if bool(settings.get("use_tls", True)):
                server.starttls()
            server.login(username, password)
            server.send_message(message)
        return True, "email"

    def dispatch(self, alerts: list[Alert]) -> list[Alert]:
        """Send alerts through the enabled channels and record every one."""
        if not alerts:
            return []
        text = self.format_message(alerts)
        channel_results: list[str] = []

        if self.dry_run:
            self.log.info("dry-run alerts (nothing sent):\n%s", text)
            channel_results.append("dry-run")
        else:
            for name, sender in (("telegram", self._send_telegram), ("email", self._send_email)):
                settings = self.channels.get(name, {}) or {}
                if not bool(settings.get("enabled", False)):
                    continue
                try:
                    ok, channel = sender(text)
                    channel_results.append(channel if ok else f"{name}:failed")
                    if not ok:
                        self.log.warning("%s delivery skipped: %s", name, channel)
                except Exception as exc:
                    self.log.error("%s delivery failed: %s", name, exc)
                    channel_results.append(f"{name}:error")

        delivered = not self.dry_run and any(
            result in {"telegram", "email"} for result in channel_results
        )
        for alert in alerts:
            alert.delivered = delivered
            alert.channel = ",".join(channel_results) or None
        return alerts

    # -- persistence ------------------------------------------------------
    def _write_alert_log(self, alerts: list[Alert]) -> None:
        if self.alert_file is None:
            return
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)
        with self.alert_file.open("a", encoding="utf-8") as handle:
            for alert in alerts:
                handle.write(
                    f"{utc_timestamp()}\t{alert.severity}\t{alert.kind}\t"
                    f"{alert.symbol or '-'}\t{alert.message}\t"
                    f"channel={alert.channel or 'none'}\tdelivered={alert.delivered}\n"
                )

    def run(
        self,
        frame: pd.DataFrame,
        previous_prices: Mapping[str, float] | None = None,
    ) -> list[Alert]:
        """Evaluate, deliver and persist alerts for one batch."""
        if not self.enabled:
            self.log.info("alerts disabled in config")
            return []
        alerts = self.dispatch(self.evaluate(frame, previous_prices))
        if not alerts:
            self.log.info("no alerts triggered")
            return []
        self._write_alert_log(alerts)
        if self.storage is not None:
            for alert in alerts:
                self.storage.log_alert(
                    kind=alert.kind,
                    message=alert.message,
                    symbol=alert.symbol,
                    severity=alert.severity,
                    delivered=alert.delivered,
                    channel=alert.channel,
                )
        self.log.info("generated %d alert(s) via %s", len(alerts), alerts[0].channel or "none")
        return alerts
