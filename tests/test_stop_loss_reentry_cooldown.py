"""Stop-loss re-entry cooldown (STOP_LOSS_REENTRY_COOLDOWN_SECONDS).

Paper trial regression: a full stop-loss exit of PICK was followed by a
re-buy of the same mint two seconds later, repeatedly. Once a stop-loss sell
is BOOKED and closes the position (paper booking, or a live confirmed fill
settled in one atomic save) new buys of that mint are refused until the
cooldown expires. Failed / refused / unresolved sells, partial sells and
take-profit exits never start it. Persisted in state; offline fakes and a
deterministic clock only (nothing is signed or sent).
"""
from __future__ import annotations

import dataclasses
import json
import types

import pytest

from bot.config import Config, _validate, load_config
from bot.jupiter import TransactionStatus
from tests.test_p1_stuck_sell_intents import (  # noqa: F401  (env is a fixture)
    MINT, MINT2, SL_MARK, T0, env, failure, intents, make_bot, open_pos, sells_of,
)

COOLDOWN = 900


def eligible(bot, *mints):
    """Run the real entry search over ``mints``; return the mints that got
    past eligibility (reached the safety screen, which then rejects them)."""
    screened = []
    bot.strategy = types.SimpleNamespace(
        find_candidates=lambda: [types.SimpleNamespace(mint=m) for m in mints])
    bot.screener = types.SimpleNamespace(
        screen=lambda m, snapshot=None: screened.append(m)
        or types.SimpleNamespace(passed=False, price_usd=0.0, symbol="X"))
    bot._candidate_market_snapshot = lambda m: None
    bot.decision_shadow = None
    bot._seek_entry()
    return screened


def stop_out(env, armed=True, results=()):
    bot = make_bot(env, results, armed=armed)
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    return bot


@pytest.mark.parametrize("armed", [True, False], ids=["live", "paper"])
def test_confirmed_full_stop_loss_blocks_same_mint_only(env, armed):
    bot = stop_out(env, armed=armed)
    assert MINT not in bot.portfolio.positions
    if armed:
        assert intents(bot) == [(MINT, "settled")]
    assert bot.portfolio.reentry_cooldowns == {MINT: T0 + COOLDOWN}
    assert eligible(bot, MINT, MINT2) == [MINT2]  # other mints unaffected


def test_cooldown_expires_deterministically(env):
    bot = stop_out(env)
    env.clock.now = T0 + COOLDOWN - 1
    assert eligible(bot, MINT) == []
    env.clock.now = T0 + COOLDOWN
    assert eligible(bot, MINT) == [MINT]
    assert bot.portfolio.reentry_cooldowns == {}  # pruned


def test_cooldown_survives_restart(env):
    stop_out(env)
    fresh = make_bot(env, load=True)
    assert fresh.portfolio.reentry_cooldowns == {MINT: T0 + COOLDOWN}
    assert fresh.portfolio.corrupt == {}
    assert eligible(fresh, MINT, MINT2) == [MINT2]
    env.clock.now = T0 + COOLDOWN
    assert eligible(fresh, MINT) == [MINT]
    fresh._save()
    assert json.load(open(env.state))["portfolio"]["reentry_cooldowns"] == {}


def test_partial_sell_does_not_start_cooldown_until_stop_loss_closes(env):
    bot = make_bot(env)
    open_pos(bot)
    env.marks[MINT] = 1.6  # +60%: first rung, 50% of the position
    bot._tick()
    pos = bot.portfolio.positions[MINT]
    assert pos.raw_tokens == 1_000_000 and pos.ladder_filled == {50.0}
    assert bot.portfolio.reentry_cooldowns == {}
    env.clock.now += 60
    env.marks[MINT] = SL_MARK  # stop-loss closes the remainder
    bot._tick()
    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.reentry_cooldowns == {MINT: T0 + 60 + COOLDOWN}


@pytest.mark.parametrize("result,status", [
    (failure(TransactionStatus.CONFIRMED_FAILURE, error="route policy: refused"), "failed"),
    (failure(TransactionStatus.TIMEOUT, sig="sig-maybe"), "unresolved"),
], ids=["failed_or_rejected", "ambiguous_unresolved"])
def test_unsettled_stop_loss_does_not_start_cooldown(env, result, status):
    bot = stop_out(env, results=[result])
    assert MINT in bot.portfolio.positions
    assert bot.portfolio.reentry_cooldowns == {}
    assert (MINT, "settled") not in intents(bot)


def test_booking_save_failure_rolls_back_cooldown(env):
    bot = make_bot(env)
    open_pos(bot)
    real_save = bot._save
    bot._save = lambda: (False if any(i.get("status") == "settled"
                                      for i in bot.portfolio.pending_intents) else real_save())
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert MINT in bot.portfolio.positions
    assert intents(bot) == [(MINT, "confirmed_unbooked")]
    assert bot.portfolio.reentry_cooldowns == {}
    bot._save = real_save
    bot._tick()  # booked from stored evidence next tick: cooldown starts now
    assert intents(bot) == [(MINT, "settled")] and MINT not in bot.portfolio.positions
    assert MINT in bot.portfolio.reentry_cooldowns
    assert len(sells_of(bot, MINT)) == 1


@pytest.mark.parametrize("armed", [True, False], ids=["live", "paper"])
def test_take_profit_full_exit_unchanged_and_no_cooldown(env, armed):
    bot = make_bot(env, armed=armed)
    open_pos(bot)
    env.marks[MINT] = 4.5  # every rung (50/100/300%) at once: sells 50%+25%+25%
    bot._tick()
    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.realized_pnl == pytest.approx(2.0 * (4.5 - 1.0))
    assert [n[3] for n in bot.notifier.sells] == ["take_profit:50", "take_profit:100",
                                                   "take_profit:300"]
    assert bot.portfolio.reentry_cooldowns == {}
    assert eligible(bot, MINT) == [MINT]


def test_zero_disables(env):
    bot = make_bot(env)
    bot.cfg = dataclasses.replace(bot.cfg, stop_loss_reentry_cooldown_seconds=0)
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.reentry_cooldowns == {}
    assert eligible(bot, MINT) == [MINT]


def _persist(env, cooldowns):
    bot = make_bot(env)
    open_pos(bot, MINT2, "B")
    bot._save()
    data = json.load(open(env.state))
    if cooldowns is _ABSENT:
        data["portfolio"].pop("reentry_cooldowns")
    else:
        data["portfolio"]["reentry_cooldowns"] = cooldowns
    json.dump(data, open(env.state, "w"))
    return make_bot(env, load=True)


_ABSENT = object()


@pytest.mark.parametrize("raw", [_ABSENT, None, [], "x", 5, {"": T0 + 100}],
                         ids=["legacy_absent", "null", "list", "string", "number", "empty_mint"])
def test_legacy_or_malformed_field_loads_empty_without_corruption(env, raw):
    fresh = _persist(env, raw)
    assert fresh.portfolio.reentry_cooldowns == {}
    assert fresh.portfolio.corrupt == {}
    assert MINT2 in fresh.portfolio.positions  # positions untouched
    assert eligible(fresh, MINT) == [MINT]
    fresh._tick()  # no crash; position still managed
    assert fresh._running is True


def test_expired_malformed_and_far_future_entries_handled(env):
    fresh = _persist(env, {
        MINT: T0 + 100,              # valid, active
        "ExpiredMint": T0 - 1,        # expired: pruned
        "BadStr": "soon", "BadBool": True, "BadNull": None, "BadList": [1],
        "FarFuture": T0 + 10**9,      # clamped to now + COOLDOWN
    })
    assert fresh.portfolio.corrupt == {}
    assert set(fresh.portfolio.reentry_cooldowns) == {MINT, "ExpiredMint", "FarFuture"}
    assert eligible(fresh, MINT, "ExpiredMint", "FarFuture", MINT2) == ["ExpiredMint"]
    assert fresh.portfolio.reentry_cooldowns == {MINT: T0 + 100, "FarFuture": T0 + COOLDOWN}
    env.clock.now = T0 + COOLDOWN
    assert eligible(fresh, "FarFuture") == ["FarFuture"]


def test_config_default_and_bounds(monkeypatch):
    assert Config().stop_loss_reentry_cooldown_seconds == 900
    for ok in (0, 1, 900, 604800):
        _validate(dataclasses.replace(Config(), stop_loss_reentry_cooldown_seconds=ok))
    for bad in (-1, 604801, 10**9):
        with pytest.raises(ValueError, match="STOP_LOSS_REENTRY_COOLDOWN_SECONDS"):
            _validate(dataclasses.replace(Config(), stop_loss_reentry_cooldown_seconds=bad))
    monkeypatch.setenv("LIVE_TRADING", "false")
    monkeypatch.setenv("WALLET_PRIVATE_KEY", "")
    monkeypatch.delenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", raising=False)
    assert load_config().stop_loss_reentry_cooldown_seconds == 900
    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "1800")
    assert load_config().stop_loss_reentry_cooldown_seconds == 1800
    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "-5")
    with pytest.raises(ValueError, match="STOP_LOSS_REENTRY_COOLDOWN_SECONDS"):
        load_config()
    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "not-a-number")
    assert load_config().stop_loss_reentry_cooldown_seconds == 900  # repo convention
