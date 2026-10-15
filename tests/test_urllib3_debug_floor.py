"""urllib3 DEBUG floor (observability only).

``LOG_LEVEL=DEBUG`` must keep Memebot diagnostics and must drop routine
urllib3 DEBUG, including quote URLs that contain mints. urllib3 WARNING and
ERROR still emit. Logger levels do not perform I/O.
"""
from __future__ import annotations

import logging

import bot.main as main_module

QUOTE_URL = (
    "https://quote.example/v6/quote?inputMint="
    "Mint111111111111111111111111111111111111111"
)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_debug_keeps_memebot_diagnostics_and_drops_urllib3_debug():
    root = logging.getLogger()
    bot = logging.getLogger("bot")
    urllib3_log = logging.getLogger("urllib3")
    pool = logging.getLogger("urllib3.connectionpool")
    before = (root.level, bot.level, urllib3_log.level, pool.level)
    handler = _ListHandler()
    root.addHandler(handler)
    try:
        bot.setLevel(logging.NOTSET)
        urllib3_log.setLevel(logging.NOTSET)
        pool.setLevel(logging.NOTSET)
        main_module._apply_log_level("DEBUG")

        bot.debug("tick start positions=%s", 1)
        pool.debug('%s %s', "GET", QUOTE_URL)
        pool.warning("urllib3 dependency warning")
        pool.error("urllib3 dependency error")

        messages = [r.getMessage() for r in handler.records]
        debug = [r.getMessage() for r in handler.records if r.levelno == logging.DEBUG]
        assert any(m.startswith("tick start positions=") for m in debug)
        assert not any(QUOTE_URL in m for m in messages)
        assert not any("inputMint=" in m for m in debug)
        assert "urllib3 dependency warning" in messages
        assert "urllib3 dependency error" in messages
        assert urllib3_log.level == logging.WARNING
        assert root.level == logging.DEBUG
    finally:
        root.removeHandler(handler)
        root.setLevel(before[0])
        bot.setLevel(before[1])
        urllib3_log.setLevel(before[2])
        pool.setLevel(before[3])
