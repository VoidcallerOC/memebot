"""P1: armed sells use exact integer raw token amounts.

On main ``_settle_live_sell`` sent ``int(min(original*frac, tokens) * 10**d)``
computed from float token counts (``tokens = raw / 10**d`` at entry), which
truncates one raw unit for many amounts (8 dp: 3->2, 6->5, 123_456_789 ->
123_456_788; 6 dp: 249->248; 9 dp: 15->14; 18 dp: 447->446 and 10**18+1 ->
10**18). Live positions now carry ``raw_tokens`` / ``raw_tokens_initial``
seeded from the verified entry fill; sells are
``min(floor(raw_tokens_initial * Fraction(repr(frac))), raw_tokens)`` and
booking decrements the raw holding by the exact confirmed amount.

Raw fields are attached with setattr (not constructor kwargs) so these tests
fail on main by the off-by-one value, not by a TypeError. Offline fakes only.
"""
from __future__ import annotations

import math
import types
from fractions import Fraction

import pytest

from tests.test_p1_stuck_sell_intents import sell_tx

import bot.main as main_module
from bot.jupiter import SOL_MINT, SwapResult, TransactionStatus
from bot.portfolio import Portfolio, Position
from bot.state import load_state
from tests.test_p1_stuck_sell_intents import (  # noqa: F401  (env is a fixture)
    MINT, T0, confirmed, env, failure, intents, make_bot, sells_of,
)

TRUNCATING = [(3, 8), (6, 8), (12, 8), (123_456_789, 8), (249, 6), (15, 9),
              (447, 18), (10**18 + 1, 18)]
exact_raw_sell_amount = getattr(main_module, "exact_raw_sell_amount", None)


def live_pos(bot, raw, dec, mint=MINT, entry=1.0):
    tokens = raw / 10**dec
    pos = bot.portfolio.open(mint, "R", entry, tokens * entry, tokens,
                             token_decimals=dec, actual_out_amount=raw)
    pos.raw_tokens = raw
    pos.raw_tokens_initial = raw
    return pos


def float_rule(raw, dec, frac=1.0):  # what main computed
    t = raw / 10**dec
    return int(min(t * frac, t) * 10**dec)


# -- root cause, reproduced -----------------------------------------------------

@pytest.mark.parametrize("raw, dec", TRUNCATING)
def test_float_rule_truncates_these_amounts(raw, dec):
    assert float_rule(raw, dec) < raw          # the bug: one or more raw units short


@pytest.mark.parametrize("raw, dec", TRUNCATING)
def test_stop_loss_sells_exact_raw_holding(env, raw, dec):
    bot = make_bot(env)
    live_pos(bot, raw, dec)
    env.marks[MINT] = 0.5
    bot._tick()
    assert sells_of(bot, MINT) == [(MINT, SOL_MINT, raw)]   # main: raw - 1 (or less)
    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.realized_pnl == pytest.approx((raw / 10**dec) * (0.5 - 1.0))


def test_entry_seeds_raw_holding_from_verified_fill():
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    from tests.test_phase3_execution import cfg as live_cfg
    bot.cfg = live_cfg()
    bot.portfolio = Portfolio()
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=MINT)])
    bot.screener = types.SimpleNamespace(
        screen=lambda _m, **k: types.SimpleNamespace(passed=True, price_usd=1.0, symbol="E"))
    bot.risk = types.SimpleNamespace(position_size_usd=lambda: 5.0)
    bot.executor = types.SimpleNamespace(swap=lambda *a, **k: SwapResult(
        ok=True, simulated=False, in_amount=1, out_amount=1, tx_signature="sig-e",
        status=TransactionStatus.CONFIRMED_SUCCESS, actual_in_amount=10**8,
        actual_out_amount=123_456_789, fee_lamports=5000, in_decimals=9, out_decimals=8))
    bot.notifier = types.SimpleNamespace(buy=lambda *a: None)
    bot._usd_to_lamports = lambda _usd: 10**8   # the fill's SOL in may not exceed this
    bot._sell_route_ok = lambda _m, _l: True
    bot._reconciliation_required = False
    bot._save = lambda: True                     # armed entries need durable saves
    bot._seek_entry()
    pos = bot.portfolio.positions[MINT]
    assert getattr(pos, "raw_tokens", None) == 123_456_789
    assert getattr(pos, "raw_tokens_initial", None) == 123_456_789
    assert pos.tokens == 123_456_789 / 10**8


# -- the pure amount rule -------------------------------------------------------

def P(raw, initial=None, dec=6):
    pos = Position(mint=MINT, symbol="R", entry_price=1.0, size_usd=1.0, tokens=1.0, token_decimals=dec)
    pos.raw_tokens = raw
    pos.raw_tokens_initial = raw if initial is None else initial
    return pos


@pytest.mark.parametrize("dec", [0, 6, 9, 18])
def test_decimals_0_6_9_18(dec):
    raw = 10**dec * 7 + 3
    assert exact_raw_sell_amount(P(raw, dec=dec), 1.0) == (raw, "")
    assert exact_raw_sell_amount(P(raw, dec=dec), 0.5) == (raw // 2, "")
    assert exact_raw_sell_amount(P(raw, dec=dec), 0.25) == (raw // 4, "")


def test_decimal_fraction_policy_is_exact():
    # float: 100 * 0.29 = 28.999999999999996 -> int 28; exact policy gives 29
    assert int(100 * 0.29) == 28
    assert exact_raw_sell_amount(P(100), 0.29) == (29, "")
    assert exact_raw_sell_amount(P(10), 0.3) == (3, "")
    assert main_module._exact_fraction(0.1) == Fraction(1, 10)
    assert main_module._exact_fraction(1) == 1


def test_full_sell_never_exceeds_current_holding():
    assert exact_raw_sell_amount(P(4, initial=7), 1.0) == (4, "")
    assert exact_raw_sell_amount(P(4, initial=7), 0.75) == (4, "")   # floor(5.25)=5, capped at 4


def test_repeated_partials_consistent_never_negative(env):
    bot = make_bot(env)
    pos = live_pos(bot, 7, 6)
    sold = []
    for frac in (0.5, 0.25, 0.25):
        raw, _ = exact_raw_sell_amount(pos, frac)
        bot.portfolio.sell_fraction(MINT, frac, 2.0, "take_profit:x", raw_sold=raw)
        sold.append(raw)
        assert pos.raw_tokens >= 0 and pos.tokens == pos.raw_tokens / 10**6
    assert sold == [3, 1, 1] and pos.raw_tokens == 2 and sum(sold) + pos.raw_tokens == 7
    raw, _ = exact_raw_sell_amount(pos, 1.0)
    bot.portfolio.sell_fraction(MINT, 1.0, 2.0, "stop_loss", raw_sold=raw)
    assert raw == 2 and MINT not in bot.portfolio.positions


@pytest.mark.parametrize("frac", [0, 0.0, -0.1, 1.0000001, 2, math.nan, math.inf, -math.inf,
                                  True, False, "0.5", None, Fraction(1, 2)])
def test_invalid_fractions_refused(frac):
    amount, reason = exact_raw_sell_amount(P(1000), frac)
    assert amount is None and "invalid sell fraction" in reason


@pytest.mark.parametrize("dec", [None, -1, 19, True, 6.0, "6"])
def test_invalid_decimals_refused(dec):
    amount, reason = exact_raw_sell_amount(P(1000, dec=dec), 1.0)
    assert amount is None and reason == "token decimals unverified"


@pytest.mark.parametrize("raw, initial", [(0, 10), (-1, 10), (11, 10), (True, 10), (5.0, 10),
                                          ("5", 10), (5, None), (5, 5.0), (None, None)])
def test_malformed_or_inconsistent_raw_refused(raw, initial):
    pos = P(raw, initial=initial)
    if initial is None:
        pos.raw_tokens_initial = None
    amount, reason = exact_raw_sell_amount(pos, 1.0)
    assert amount is None and reason


def test_zero_result_refused():
    assert exact_raw_sell_amount(P(3), 0.25) == (None, "sell amount rounds to zero raw units")


def test_huge_raw_amount_stays_exact():
    raw = 2**64 - 1
    assert exact_raw_sell_amount(P(raw, dec=18), 1.0) == (raw, "")
    assert exact_raw_sell_amount(P(raw, dec=18), 0.5) == (raw // 2, "")


# -- intents, settlement, exactly once -----------------------------------------

def test_intent_persists_exact_raw_and_failure_does_not_decrement(env):
    bot = make_bot(env, [failure(TransactionStatus.CONFIRMED_FAILURE)])
    pos = live_pos(bot, 123_456_789, 8)
    env.marks[MINT] = 0.5
    bot._tick()
    bot._save()
    assert bot.portfolio.pending_intents[0]["raw_amount"] == 123_456_789
    assert pos.raw_tokens == 123_456_789
    bot2 = make_bot(env, load=True)
    assert bot2.portfolio.pending_intents[0]["raw_amount"] == 123_456_789
    assert bot2.portfolio.positions[MINT].raw_tokens == 123_456_789
    assert bot2.portfolio.positions[MINT].raw_tokens_initial == 123_456_789


def test_on_time_success_decrements_exactly_once(env):
    bot = make_bot(env)
    pos = live_pos(bot, 7, 6)
    env.marks[MINT] = 1.6                       # +60% -> 0.5 rung
    bot._tick()
    assert sells_of(bot, MINT) == [(MINT, SOL_MINT, 3)]
    assert pos.raw_tokens == 4 and pos.tokens == 4 / 10**6
    assert bot.portfolio.realized_pnl == pytest.approx(3 / 10**6 * 0.6)
    bot._tick()                                  # rung already filled: nothing more
    assert pos.raw_tokens == 4 and len(bot.notifier.sells) == 1


def test_delayed_settlement_decrements_exactly_once(env):
    bot = make_bot(env, [failure(TransactionStatus.TIMEOUT, sig="sig-1")])
    pos = live_pos(bot, 123_456_789, 8)
    env.marks[MINT] = 1.6
    bot._tick()
    assert pos.raw_tokens == 123_456_789          # unknown: no decrement
    env.chain.status["sig-1"] = {"err": None, "confirmationStatus": "confirmed"}
    # strictly provable persistent-wSOL sell (see test_p1_stuck_sell_intents.sell_tx)
    env.chain.txs["sig-1"] = sell_tx(mint=MINT, token_pre=123_456_789, token_post=61_728_395,
                                     wsol_out=5_000_000, owner="WalletPubkey1111111111111111111111111111111")
    for _ in range(3):
        bot._tick()
    assert intents(bot) == [(MINT, "settled")]
    assert pos.raw_tokens == 123_456_789 - 61_728_394   # floor(123456789 * 1/2) = 61728394
    assert len(bot.notifier.sells) == 1


def test_actual_in_mismatch_not_booked(env):
    bad = confirmed(1)
    bad.actual_in_amount = 2_999_999
    bot = make_bot(env, [bad])
    pos = live_pos(bot, 3_000_000, 6)
    env.marks[MINT] = 0.5
    bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]
    assert pos.raw_tokens == 3_000_000 and bot.portfolio.realized_pnl == 0.0
    assert bot._reconciliation_required is True
    assert bot.notifier.sells == []


# -- persistence / legacy --------------------------------------------------------

def test_serialization_round_trip_and_legacy_dict():
    pos = P(123, initial=456)
    d = pos.to_dict()
    assert (d["raw_tokens"], d["raw_tokens_initial"]) == (123, 456)
    back = Position.from_dict(d)
    assert (back.raw_tokens, back.raw_tokens_initial) == (123, 456)
    legacy = {k: v for k, v in d.items() if not k.startswith("raw_tokens")}
    old = Position.from_dict(legacy)
    assert old.raw_tokens is None and old.raw_tokens_initial is None


def test_legacy_live_position_fails_closed_not_rounded(env, caplog):
    bot = make_bot(env)
    bot.portfolio.open(MINT, "L", 1.0, 2.0, 2.0, token_decimals=6, actual_out_amount=2_000_000)
    env.marks[MINT] = 0.5
    bot._tick()
    assert bot.executor.calls == []                       # nothing submitted
    intent = bot.portfolio.pending_intents[0]
    assert intent["status"] == "blocked" and intent["raw_amount"] is None
    assert "operator action required" in intent["error"]
    assert bot._reconciliation_required is False
    pos = bot.portfolio.positions[MINT]
    assert (pos.tokens, pos.realized_pnl, bot.portfolio.realized_pnl) == (2.0, 0.0, 0.0)
    assert any("refused before submit" in r.message for r in caplog.records)
    bot._tick()                                            # resolver clears; refused again
    assert [i["status"] for i in bot.portfolio.pending_intents] == ["failed", "blocked"]
    assert bot.executor.calls == []


def test_legacy_state_file_loads_and_stays_fail_closed(env):
    bot = make_bot(env)
    bot.portfolio.open(MINT, "L", 1.0, 2.0, 2.0, token_decimals=6)
    d = bot.portfolio.to_dict()
    for p in d["positions"]:
        p.pop("raw_tokens"), p.pop("raw_tokens_initial")
    bot.portfolio.load_dict(d)
    bot._save()
    bot2 = make_bot(env, load=True)
    assert bot2.portfolio.positions[MINT].raw_tokens is None
    env.marks[MINT] = 0.5
    bot2._tick()
    assert bot2.executor.calls == []


def test_paper_sell_unchanged_float_path(env):
    bot = make_bot(env, armed=False)
    bot.portfolio.open(MINT, "P", 1.0, 2.0, 2.0, token_decimals=6)
    bot._sell(MINT, 0.5, 2.0, "take_profit:100")
    pos = bot.portfolio.positions[MINT]
    assert pos.tokens == 1.0 and pos.raw_tokens is None
    assert bot.portfolio.realized_pnl == pytest.approx(1.0)
    assert bot.executor.calls == []


# Written before the per-position exit backoff: these tests re-run an exit on
# the next tick, so the delay is zeroed (counter/alert still real).
pytestmark = pytest.mark.usefixtures("no_exit_backoff")
