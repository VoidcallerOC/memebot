"""Trade alerts via Telegram and/or Discord.

Both channels are optional and fully off unless configured. Alerts are
best-effort: a failed notification is logged and swallowed — it must never
crash the trading loop or block a trade/exit.
"""

from __future__ import annotations

import logging

try:
    import requests
except ImportError:  # alerts are best-effort; never block on a missing dep
    requests = None  # type: ignore[assignment]

from .config import Config

log = logging.getLogger(__name__)

TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"
HTTP_TIMEOUT = 6


class Notifier:
    def __init__(self, cfg: Config, session=None):
        self.cfg = cfg
        self.session = session or (requests.Session() if requests else None)
        self._telegram = bool(cfg.telegram_bot_token and cfg.telegram_chat_id)
        self._discord = bool(cfg.discord_webhook_url)

    @property
    def enabled(self) -> bool:
        return (self._telegram or self._discord) and self.session is not None

    def send(self, text: str) -> None:
        """Fire-and-forget to every configured channel."""
        if self._telegram:
            self._send_telegram(text)
        if self._discord:
            self._send_discord(text)

    # -- convenience wrappers for the loop ---------------------------------

    def startup(self, mode: str) -> None:
        self.send(f"🤖 Bot started — mode: <b>{mode}</b>" if self._telegram
                  else f"🤖 Bot started — mode: {mode}")

    def buy(self, symbol: str, size_usd: float, price: float, simulated: bool) -> None:
        tag = "PAPER" if simulated else "LIVE"
        self.send(f"🟢 [{tag}] BUY {symbol} — ${size_usd:.2f} @ {price:.8g}")

    def sell(self, symbol: str, fraction: float, price: float,
             reason: str, pnl: float) -> None:
        emoji = "🔴" if reason == "stop_loss" else "💰"
        self.send(
            f"{emoji} SELL {symbol} {fraction*100:.0f}% ({reason}) "
            f"@ {price:.8g} — PnL ${pnl:+.2f}"
        )

    def halt(self, lost_usd: float) -> None:
        self.send(f"⛔ DAILY LOSS LIMIT hit (−${lost_usd:.2f}). Halting new entries.")

    # -- channels ----------------------------------------------------------

    def _send_telegram(self, text: str) -> None:
        try:
            self.session.post(
                TELEGRAM_URL.format(token=self.cfg.telegram_bot_token),
                json={
                    "chat_id": self.cfg.telegram_chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=HTTP_TIMEOUT,
            )
        except Exception as exc:
            log.debug("telegram alert failed: %s", exc)

    def _send_discord(self, text: str) -> None:
        try:
            self.session.post(
                self.cfg.discord_webhook_url,
                json={"content": text},
                timeout=HTTP_TIMEOUT,
            )
        except Exception as exc:
            log.debug("discord alert failed: %s", exc)
