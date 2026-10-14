"""LOG_LEVEL config and DEBUG tick diagnostics (observability only).

Offline fakes and the deterministic clock from the stuck-sell suite. Checks
that LOG_LEVEL is parsed strictly, that DEBUG emits the per-tick diagnostics,
and that DEBUG never changes decisions, network/RPC call counts or state, and
never logs secrets.
"""
from __future__ import annotations

import dataclasses
import logging

import pytest

import bot.main as main_module
from bot.config import LOG_LEVELS, Config, _validate, load_config
from tests.test_p1_stuck_sell_intents import (  # noqa: F401  (env is a fixture)
    MINT, MINT2, OK_MARK, SENTINEL, SL_MARK, env, make_bot, open_pos,
)
from tests.test_stop_loss_reentry_cooldown import eligible, stop_out

REAL_PRICE_USD = main_module._price_usd  # captured before the env fixture patches it
SECRETS = {
    "wallet_private_key": SENTINEL,
    "jupiter_api_key": "jup-key-SECRET-5f1e",
    "telegram_bot_token": "123456:TG-SECRET-TOKEN",
    "discord_webhook_url": "https://discord.example/api/webhooks/1/DISCORD-SECRET",
}


@pytest.fixture
def debug_logs(caplog):
    caplog.set_level(logging.DEBUG, logger="bot")
    return caplog


def debug_text(caplog):
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


# --- config -----------------------------------------------------------------

def test_default_info_when_absent(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert load_config().log_level == "INFO"
    assert Config().log_level == "INFO"


def test_blank_is_default(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "   ")
    assert load_config().log_level == "INFO"


@pytest.mark.parametrize("level", LOG_LEVELS)
def test_each_supported_level(monkeypatch, level):
    monkeypatch.setenv("LOG_LEVEL", level)
    assert load_config().log_level == level


@pytest.mark.parametrize("raw,want", [("debug", "DEBUG"), ("  Warning\n", "WARNING"),
                                      ("\tcritical ", "CRITICAL"), ("eRRoR", "ERROR")])
def test_case_and_whitespace_normalized(monkeypatch, raw, want):
    monkeypatch.setenv("LOG_LEVEL", raw)
    assert load_config().log_level == want


@pytest.mark.parametrize("raw", ["verbose", "WARN", "10", "INFO,DEBUG", "NOTSET"])
def test_invalid_level_is_a_clear_config_error(monkeypatch, raw):
    monkeypatch.setenv("LOG_LEVEL", raw)
    with pytest.raises(ValueError, match="LOG_LEVEL must be one of DEBUG, INFO, WARNING"):
        load_config()
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        _validate(Config(log_level=raw.upper()))


def test_main_exits_2_on_invalid_level_before_anything_starts(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "loud")
    monkeypatch.setenv("LIVE_TRADING", "false")
    monkeypatch.setattr(main_module, "ProcessLock",
                        lambda *a: pytest.fail("must not reach the process lock"))
    assert main_module.main() == 2


@pytest.mark.parametrize("level", LOG_LEVELS)
def test_apply_log_level_sets_root(level):
    root = logging.getLogger()
    before = root.level
    try:
        main_module._apply_log_level(level)
        assert root.level == getattr(logging, level)
    finally:
        root.setLevel(before)


# --- DEBUG diagnostics ------------------------------------------------------

def test_tick_start_end_and_normal_price_read(env, debug_logs):
    bot = make_bot(env, armed=False)
    open_pos(bot)
    env.marks[MINT] = 1.2
    bot._tick()
    text = debug_text(debug_logs)
    assert "tick start: 1 open position(s), 0 pending intent(s)" in text
    assert "tick end: " in text and "s elapsed, 1 open position(s)" in text
    assert (f"position {MINT[:8]} (A): price ok 1.2 | entry 1 change +20.00% SL<=0.85 (15%) "
            "next TP +50% -> no threshold crossed") in text
    assert MINT not in text  # only the short form


@pytest.mark.parametrize("mark,word", [(None, "missing (None)"),
                                       (float("nan"), "invalid/rejected"),
                                       (-1.0, "invalid/rejected"),
                                       ("1.0", "invalid/rejected")])
def test_missing_or_invalid_price_debug(env, debug_logs, mark, word):
    bot = make_bot(env, armed=False)
    open_pos(bot)
    env.marks[MINT] = mark
    bot._tick()
    assert f"position {MINT[:8]} (A): price {word} -> hold" in debug_text(debug_logs)
    assert MINT in bot.portfolio.positions


@pytest.mark.parametrize("armed", [True, False], ids=["live", "paper"])
def test_stop_loss_decision_debug(env, debug_logs, armed):
    stop_out(env, armed=armed)
    assert (f"position {MINT[:8]} (A): price ok 0.5 | entry 1 change -50.00% SL<=0.85 "
            "(15%) next TP +50% -> exit requested: stop_loss") in debug_text(debug_logs)


def test_take_profit_decision_debug(env, debug_logs):
    bot = make_bot(env, armed=False)
    open_pos(bot)
    env.marks[MINT] = 2.5
    bot._tick()
    text = debug_text(debug_logs)
    assert "change +150.00%" in text
    assert "-> exit requested: take_profit:50, take_profit:100" in text


def test_exit_backoff_skip_debug(env, debug_logs):
    bot = make_bot(env)
    open_pos(bot)
    pos = bot.portfolio.positions[MINT]
    pos.exit_failures, pos.exit_retry_at = 2, env.clock.now + 60
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert MINT in bot.portfolio.positions and bot.executor.calls == []
    assert (f"position {MINT[:8]} (A): exit backoff active -> skip stop_loss this tick"
            in debug_text(debug_logs))


def test_entry_skip_diagnostics(env, debug_logs):
    bot = stop_out(env)  # MINT now in cooldown
    open_pos(bot, mint=MINT2, symbol="B")
    debug_logs.clear()
    assert eligible(bot, MINT, MINT2, "FreshMint9999") == ["FreshMint9999"]
    info = [r.getMessage() for r in debug_logs.records if r.levelno == logging.INFO]
    assert f"entry skipped for {MINT[:8]}: stop-loss re-entry cooldown active" in info
    text = debug_text(debug_logs)
    assert f"entry skipped for {MINT2[:8]}: already held" in text
    assert "entry rejected for FreshMin: safety screen (passed=False, price>0=False" in text


def test_buy_frozen_skip_debug(env, debug_logs):
    bot = make_bot(env)
    bot._buy_frozen = lambda m: True
    assert eligible(bot, MINT) == []
    assert f"entry skipped for {MINT[:8]}: buy frozen" in debug_text(debug_logs)


def test_risk_gate_debug(env, debug_logs):
    bot = make_bot(env, armed=False)
    for i in range(bot.cfg.max_open_positions):
        open_pos(bot, mint=f"M{i}" + "1" * 40, symbol=f"S{i}")
        env.marks[f"M{i}" + "1" * 40] = OK_MARK
    bot._tick()
    assert "entry search skipped: risk gate (open=3, max=3" in debug_text(debug_logs)
    assert bot.entry_scans == []


def test_real_price_usd_debug_attempt_and_missing(debug_logs):
    class Jup:
        def quote(self, *a, **k):
            return None
    assert REAL_PRICE_USD(Jup(), MINT, 6) is None
    text = debug_text(debug_logs)
    assert f"price read {MINT[:8]}: quoting 1 whole token (decimals=6) to USDC" in text
    assert f"price read {MINT[:8]}: no quote/outAmount (missing)" in text


def test_nothing_emitted_at_info(env, caplog):
    caplog.set_level(logging.INFO, logger="bot")
    stop_out(env)
    assert not [r for r in caplog.records if r.levelno == logging.DEBUG]


# --- DEBUG never changes behaviour -------------------------------------------

class CountingJup:
    """Real _price_usd path: counts every quote call (the only price I/O)."""

    def __init__(self, marks):
        self.marks, self.calls = marks, []

    def quote(self, inp, out, amount, **kw):
        self.calls.append((inp, out, amount, tuple(sorted(kw))))
        usd = self.marks.get(inp)
        return None if usd is None else {"outAmount": str(int(usd * 10**6))}


def run_scenario(env, monkeypatch, level, armed):
    logging.getLogger("bot").setLevel(level)
    monkeypatch.setattr(main_module, "_price_usd", REAL_PRICE_USD)
    env.chain.calls.clear()
    env.clock.now = 1_800_000_000.0
    bot = make_bot(env, armed=armed)
    bot.jup = CountingJup({MINT: 0.5, MINT2: 2.5})
    open_pos(bot)
    open_pos(bot, mint=MINT2, symbol="B")
    bot._tick()
    env.clock.now += 30
    bot._tick()
    state = (sorted(bot.portfolio.positions), bot.portfolio.realized_pnl,
             dict(bot.portfolio.reentry_cooldowns),
             [(i["mint"], i["status"]) for i in bot.portfolio.pending_intents],
             {m: (p.raw_tokens, sorted(p.ladder_filled)) for m, p in bot.portfolio.positions.items()})
    return (list(bot.jup.calls), list(env.chain.calls), list(bot.executor.calls),
            [n[3] for n in bot.notifier.sells], state)


@pytest.mark.parametrize("armed", [True, False], ids=["live", "paper"])
def test_debug_makes_no_extra_calls_and_same_decisions(env, monkeypatch, armed):
    logger = logging.getLogger("bot")
    before = logger.level
    try:
        info = run_scenario(env, monkeypatch, logging.INFO, armed)
        debug = run_scenario(env, monkeypatch, logging.DEBUG, armed)
    finally:
        logger.setLevel(before)
    assert info == debug
    assert info[0]  # quotes really went through the real _price_usd


def test_no_secrets_or_payloads_at_debug(env, debug_logs, monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k.upper(), v)
    bot = make_bot(env)
    bot.cfg = dataclasses.replace(bot.cfg, **SECRETS)
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    eligible(bot, MINT, MINT2)
    everything = "\n".join(r.getMessage() for r in debug_logs.records)
    assert debug_text(debug_logs)
    for v in SECRETS.values():
        assert v not in everything
    for word in ("swapTransaction", "signedTransaction", "outAmount", "Authorization", "{"):
        assert word not in debug_text(debug_logs)
    assert max(len(r.getMessage()) for r in debug_logs.records) < 400


class RaisingDebugHandler(logging.Handler):
    def emit(self, record):
        if record.levelno == logging.DEBUG:
            raise RuntimeError("broken log handler")


def test_logging_failure_does_not_change_outcome(env):
    logger = logging.getLogger("bot")
    handler, before = RaisingDebugHandler(level=logging.DEBUG), logger.level
    control = stop_out(env, armed=False)
    want = (sorted(control.portfolio.positions), control.portfolio.realized_pnl,
            dict(control.portfolio.reentry_cooldowns))
    env.clock.now += 1
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        bot = make_bot(env, armed=False)
        bot.portfolio.reentry_cooldowns.clear()
        open_pos(bot)
        bot._tick()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(before)
    got = (sorted(bot.portfolio.positions), bot.portfolio.realized_pnl,
           {k: v - 1 for k, v in bot.portfolio.reentry_cooldowns.items()})
    assert got == want


def test_bad_repr_symbol_does_not_raise(env, debug_logs):
    class Evil(str):
        def __str__(self):
            raise RuntimeError("bad str")
    bot = make_bot(env, armed=False)
    open_pos(bot)
    bot.portfolio.positions[MINT].symbol = Evil("A")  # only DEBUG lines format it at OK_MARK
    env.marks[MINT] = OK_MARK
    bot._tick()  # the diagnostic's formatting error is swallowed by _debug
    assert MINT in bot.portfolio.positions and bot.executor.calls == []
    assert any(r.msg.startswith("tick end") for r in debug_logs.records)
