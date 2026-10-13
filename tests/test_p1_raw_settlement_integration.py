"""P1: exact raw sells (#35) integrated with crash-safe settlement (#36).

Covers what only the combination needs:

* raw units are u64 (``1 .. 2**64-1``, the bound ``jupiter.parse_raw_amount``
  enforces) everywhere: holdings, computed sell amounts, persisted raw amounts
  accepted by settlement/recovery, and the entry fill. Bools, non-ints,
  negatives, zero, inconsistent or oversized values fail closed BEFORE any
  quote, signature or send, and are never clamped, truncated or repaired;
* a failed state save is never treated as durable: no send without a durable
  intent; a booking whose save failed is rolled back and retried;
* the delayed/recovery path decrements by the intent's persisted raw_amount.

Same offline world as ``test_p1_native_sol_settlement_e2e`` (real executor
confirm path, real get_confirmed_transaction + extract_fill on the pinned
fixture; signing spy, send stub, fake HTTP). No network.
"""
from __future__ import annotations

import json
import types

import pytest

import bot.jupiter as jupiter_module
import bot.main as main_module
from bot.jupiter import SOL_MINT, SwapResult, TransactionStatus
from bot.portfolio import MAX_RAW_UNITS, Portfolio, Position, is_raw_units
from bot.state import save_state
from tests.test_p1_native_sol_settlement_e2e import (  # noqa: F401  (world is a fixture)
    ENTRY, FULL, IN_RAW, MINT, OUT_LAMPORTS, RAW_DOUBLE, SIG, SL_MARK, SL_PNL, TP_MARK,
    assert_booked_once, assert_never_terminal_unbooked, boot, intents_of, lands,
    open_position, persisted, position, run, show, world,
)

U64 = 2**64 - 1


# -- the bound ----------------------------------------------------------------------

def test_raw_bound_is_the_parsers_u64_bound():
    assert MAX_RAW_UNITS == U64 == jupiter_module._U64_MAX
    assert jupiter_module.parse_raw_amount(str(U64)) == U64
    assert jupiter_module.parse_raw_amount(str(U64 + 1)) is None
    assert [is_raw_units(v) for v in (1, U64)] == [True, True]
    assert not any(is_raw_units(v) for v in (0, -1, U64 + 1, True, 1.0, "1", None))


def P(raw, initial, dec=8):
    pos = Position(mint=MINT, symbol="R", entry_price=1.0, size_usd=1.0, tokens=1.0, token_decimals=dec)
    pos.raw_tokens, pos.raw_tokens_initial = raw, initial
    return pos


@pytest.mark.parametrize("raw, initial", [
    (U64 + 1, U64 + 1), (U64, U64 + 1), (2**70, 2**70), (IN_RAW + 1, IN_RAW), (0, IN_RAW),
    (-1, IN_RAW), (True, IN_RAW), (IN_RAW, True), (float(IN_RAW), IN_RAW), (str(IN_RAW), IN_RAW),
    (IN_RAW, None), (None, IN_RAW),
])
def test_exact_amount_refuses_out_of_range_or_inconsistent_holding(raw, initial):
    amount, reason = main_module.exact_raw_sell_amount(P(raw, initial), 1.0)
    assert amount is None and reason


def test_exact_amount_at_the_u64_edge_is_exact():
    assert main_module.exact_raw_sell_amount(P(U64, U64, dec=18), 1.0) == (U64, "")
    assert main_module.exact_raw_sell_amount(P(U64, U64, dec=18), 0.29) == (U64 * 29 // 100, "")


@pytest.mark.parametrize("raw_sold, held, initial", [
    (U64 + 1, U64, U64), (IN_RAW, U64 + 1, U64 + 1), (IN_RAW, IN_RAW, IN_RAW - 1),
    (IN_RAW + 1, IN_RAW, IN_RAW), (0, IN_RAW, IN_RAW), (True, IN_RAW, IN_RAW), (1.0, IN_RAW, IN_RAW),
])
def test_portfolio_raw_sell_rejects_without_mutation(raw_sold, held, initial):
    pf = Portfolio()
    pos = pf.open(MINT, "R", 1.0, 1.0, 1.0, token_decimals=8,
                  raw_tokens=held, raw_tokens_initial=initial)
    before = pos.to_dict()
    with pytest.raises(ValueError):
        pf.sell_fraction(MINT, 1.0, 1.0, "stop_loss", raw_sold=raw_sold)
    assert pf.positions[MINT].to_dict() == before and pf.realized_pnl == 0.0


@pytest.mark.parametrize("out_raw", [U64 + 1, 0, True])
def test_entry_with_out_of_range_fill_raw_records_no_position(out_raw):
    from tests.test_phase3_execution import cfg as live_cfg
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = live_cfg()
    bot.portfolio = Portfolio()
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=MINT)])
    bot.screener = types.SimpleNamespace(
        screen=lambda _m, **k: types.SimpleNamespace(passed=True, price_usd=1.0, symbol="E"))
    bot.risk = types.SimpleNamespace(position_size_usd=lambda: 5.0)
    bot.executor = types.SimpleNamespace(swap=lambda *a, **k: SwapResult(
        ok=True, simulated=False, in_amount=1, out_amount=1, tx_signature="sig-e",
        status=TransactionStatus.CONFIRMED_SUCCESS, actual_in_amount=10**8,
        actual_out_amount=out_raw, fee_lamports=5000, in_decimals=9, out_decimals=8))
    bot.notifier = types.SimpleNamespace(buy=lambda *a: (_ for _ in ()).throw(AssertionError("no buy")))
    bot._usd_to_lamports = lambda _usd: 10**8   # SOL in is valid; only the raw out is bad
    bot._sell_route_ok = lambda _m, _l: True
    bot._reconciliation_required = False
    bot._save = lambda: True                     # armed entries need durable saves
    bot._seek_entry()
    assert bot.portfolio.positions == {}
    (intent,) = bot.portfolio.buy_intents        # unresolved: entries blocked, nothing recorded
    assert intent["status"] == "unresolved" and "token output" in intent["resolution"]


# -- corrupted persisted holdings fail closed before quote / sign / send ----------

@pytest.mark.parametrize("raw, initial, why", [
    (U64 + 1, U64 + 1, "out of range"),
    (2**70, 2**70, "out of range"),
    (IN_RAW + 1, IN_RAW, "inconsistent"),
    (0, IN_RAW, "inconsistent"),
    (-IN_RAW, IN_RAW, "inconsistent"),
    (True, IN_RAW, "no exact raw holding"),
    (float(IN_RAW), float(IN_RAW), "no exact raw holding"),
    (str(IN_RAW), IN_RAW, "no exact raw holding"),
    (IN_RAW, None, "no exact raw holding"),
    (None, None, "operator action required"),        # legacy: no raw fields at all
])
def test_corrupted_persisted_holding_fails_closed_before_any_execution(world, raw, initial, why):
    bot = boot(world)
    open_position(bot, world)
    state = persisted(world)
    pos = position(state)
    pos["raw_tokens"], pos["raw_tokens_initial"] = raw, initial
    with open(world.state, "w") as f:                  # tmp test state only
        json.dump(state, f)

    bot2 = boot(world)                                 # production load, no repair
    world.marks[MINT] = SL_MARK
    run(bot2, 3)
    assert bot2.jup.quotes == [] and world.signs == [] and world.sends == []
    after = persisted(world)
    kept = position(after)
    assert (kept["raw_tokens"], kept["raw_tokens_initial"]) == (raw, initial)  # never repaired
    assert kept["tokens"] == FULL and after["portfolio"]["realized_pnl"] == 0.0
    mine = intents_of(after)
    assert mine and all(i["status"] in {"blocked", "failed"} for i in mine)
    assert all(why in (i.get("error") or "") for i in mine)
    assert bot2.notifier.sells == []


# -- persisted raw amounts accepted by settlement / recovery ----------------------

def _unbooked(**over):
    intent = {"mint": MINT, "fraction": 1.0, "reason": "stop_loss", "requested_price": SL_MARK,
              "status": "confirmed_unbooked", "tx_signature": SIG, "created_at": 1.0,
              "raw_amount": IN_RAW, "actual_in_amount": IN_RAW,
              "actual_out_amount": OUT_LAMPORTS, "fee_lamports": 10_143}
    intent.update(over)
    return intent


@pytest.mark.parametrize("intent_over, pos_raw, why", [
    ({"raw_amount": U64 + 1, "actual_in_amount": U64 + 1}, (IN_RAW, IN_RAW), "raw_amount"),
    ({"raw_amount": 0, "actual_in_amount": 0}, (IN_RAW, IN_RAW), "raw_amount"),
    ({"raw_amount": -1, "actual_in_amount": -1}, (IN_RAW, IN_RAW), "raw_amount"),
    ({"raw_amount": float(IN_RAW), "actual_in_amount": IN_RAW}, (IN_RAW, IN_RAW), "raw_amount"),
    ({"actual_in_amount": U64 + 1}, (IN_RAW, IN_RAW), "input amount"),
    ({"actual_out_amount": U64 + 1}, (IN_RAW, IN_RAW), "output amount"),
    ({"actual_out_amount": None}, (IN_RAW, IN_RAW), "output amount"),
    ({}, (IN_RAW - 1, IN_RAW - 1), "exceeds the current raw holding"),
    ({}, (IN_RAW + 1, IN_RAW), "raw holding inconsistent"),
    ({}, (U64 + 1, U64 + 1), "raw holding inconsistent"),
    ({}, (IN_RAW, None), "raw holding inconsistent"),
])
def test_recovery_rejects_out_of_range_or_inconsistent_persisted_amounts(world, intent_over, pos_raw, why):
    bot = boot(world)
    open_position(bot, world)
    pos = bot.portfolio.positions[MINT]
    pos.raw_tokens, pos.raw_tokens_initial = pos_raw
    bot.portfolio.pending_intents.append(_unbooked(**intent_over))
    bot._save()
    bot2 = boot(world)
    bot2._resolve_sell_intents()
    state = persisted(world)
    (stored,) = intents_of(state)
    assert stored["status"] == "confirmed_unbooked"           # open: mint frozen
    assert why in stored["resolution"] and "operator action required" in stored["resolution"]
    assert (position(state)["raw_tokens"], position(state)["raw_tokens_initial"]) == pos_raw
    assert state["portfolio"]["realized_pnl"] == 0.0 and state["risk"]["realized_pnl_today"] == 0.0
    assert bot2._sell_frozen(MINT) and bot2.notifier.sells == []


def test_delayed_recovery_decrements_by_the_persisted_raw_amount(world):
    """The intent's persisted raw_amount is what is decremented: the resolver
    never recomputes it from the fraction (the fraction is tampered here)."""
    bot = boot(world)
    open_position(bot, world, raw=RAW_DOUBLE)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL, visible=False)
    world.marks[MINT] = TP_MARK
    bot._tick()                                         # TIMEOUT: unresolved, no decrement
    state = persisted(world)
    (intent,) = intents_of(state)
    assert (intent["status"], intent["raw_amount"]) == ("unresolved", IN_RAW)
    assert position(state)["raw_tokens"] == RAW_DOUBLE and state["portfolio"]["realized_pnl"] == 0.0
    intent["fraction"] = 1.0                            # tmp state: fraction no longer matches raw
    with open(world.state, "w") as f:
        json.dump(state, f)

    show(world)
    bot2 = boot(world)
    assert run(bot2, 3) == 3
    after = assert_booked_once(world, pnl=FULL * (TP_MARK - ENTRY), remaining_raw=IN_RAW)
    assert world.rpc.count("getTransaction") >= 1      # real getTransaction + extract_fill
    bot3 = boot(world)
    run(bot3, 3)
    assert bot3.notifier.sells == []
    assert intents_of(persisted(world)) == intents_of(after)
    assert_booked_once(world, pnl=FULL * (TP_MARK - ENTRY), remaining_raw=IN_RAW)


# -- a failed save is never treated as durable --------------------------------------

def test_save_state_reports_success_and_failure(tmp_path):
    pf, risk = Portfolio(), types.SimpleNamespace(to_dict=lambda: {})
    good = tmp_path / "state.json"
    assert save_state(str(good), pf, risk) is True
    before = good.read_text()
    blocker = tmp_path / "file"
    blocker.write_text("x")                             # a file where a directory is needed
    assert save_state(str(blocker / "state.json"), pf, risk) is False
    pf.realized_pnl = 1.0
    assert good.read_text() == before


def _fail_saves(monkeypatch, failing):
    """save_state returns False (writes nothing) on the given 1-based calls."""
    real = main_module.save_state
    count = [0]

    def save(*a, **kw):
        count[0] += 1
        if count[0] in failing:
            return False
        return real(*a, **kw)

    monkeypatch.setattr(main_module, "save_state", save)
    return real


def test_presend_save_failure_sends_nothing_then_retries_and_books_once(world, monkeypatch):
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL)
    world.marks[MINT] = SL_MARK
    _fail_saves(monkeypatch, {1})                       # the pre-send intent save fails
    bot._tick()
    assert bot.jup.quotes == [] and world.signs == [] and world.sends == []
    (intent,) = bot.portfolio.pending_intents
    assert intent["status"] == "blocked" and "nothing sent" in intent["error"]
    assert bot.portfolio.positions[MINT].raw_tokens == IN_RAW
    bot._tick()                                         # cleared, stop-loss retried, saves work
    assert [i["status"] for i in intents_of(persisted(world))] == ["failed", "settled"]
    assert_booked_once(world)
    assert len(bot.notifier.sells) == 1


@pytest.mark.parametrize("path", ["on_time", "delayed"])
def test_booking_save_failure_rolls_back_and_books_once_later(world, monkeypatch, path):
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL, visible=path == "on_time")
    world.marks[MINT] = SL_MARK
    if path == "delayed":
        bot._tick()
        show(world)
    # saves in the settling tick: [pending, signature,] confirmed_unbooked, booking
    _fail_saves(monkeypatch, {4} if path == "on_time" else {2})
    bot._tick()
    (intent,) = bot.portfolio.pending_intents
    assert intent["status"] == "confirmed_unbooked" and "not durable" in intent["resolution"]
    assert "booked_at" not in intent
    pos = bot.portfolio.positions[MINT]                 # memory rolled back exactly
    assert (pos.raw_tokens, pos.tokens) == (IN_RAW, FULL)
    assert bot.portfolio.realized_pnl == 0.0 and bot.risk.realized_pnl_today == 0.0
    assert bot.notifier.sells == []
    state = persisted(world)                            # disk: evidence, no booking
    assert [i["status"] for i in intents_of(state)] == ["confirmed_unbooked"]
    assert position(state)["raw_tokens"] == IN_RAW and state["portfolio"]["realized_pnl"] == 0.0
    assert_never_terminal_unbooked(world)

    for _ in range(3):
        bot._tick()                                     # saves work again: booked once
    assert len(bot.notifier.sells) == 1
    assert_booked_once(world)
    run(boot(world), 2)
    assert_booked_once(world)


def test_evidence_and_booking_saves_both_fail_then_crash_never_double(world, monkeypatch):
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG, SIG]
    world.on_send[SIG] = lands(MINT, FULL)
    world.marks[MINT] = SL_MARK
    # saves: pending, signature (pre-broadcast), evidence, booking
    real_save = _fail_saves(monkeypatch, {3, 4})
    bot._tick()
    del bot                     # process dies; disk has the intent with its pre-broadcast signature
    (intent,) = intents_of(persisted(world))
    assert (intent["status"], intent["raw_amount"], intent["tx_signature"]) == ("pending", IN_RAW, SIG)
    monkeypatch.setattr(main_module, "save_state", real_save)   # saves work again
    world.now += main_module.SELL_INTENT_EXPIRY_SECONDS + 1
    bot2 = boot(world)                                  # production restart from disk
    run(bot2, 3)                                        # resolves by the saved signature
    assert world.sends == [SIG]                         # never sent twice
    assert_never_terminal_unbooked(world)
    assert_booked_once(world)                           # booked once, from on-chain evidence


@pytest.mark.parametrize("out", [None, 0, U64 + 1, True])
def test_on_time_confirmed_without_valid_proceeds_never_books(world, out):
    """The executor reports a confirmed sell whose input matches but whose
    proceeds are missing / zero / out of range: unresolved, frozen, no P&L,
    raw holding untouched."""
    bot = boot(world)
    open_position(bot, world)
    calls = []

    def swap(i, o, amount, *, allocation_usd=None, before_broadcast=None):
        calls.append((i, o, amount))
        return SwapResult(ok=True, simulated=False, in_amount=amount, out_amount=1,
                          tx_signature=SIG, status=TransactionStatus.CONFIRMED_SUCCESS,
                          actual_in_amount=amount, actual_out_amount=out, fee_lamports=1)

    bot.executor = types.SimpleNamespace(swap=swap)
    world.marks[MINT] = SL_MARK
    bot._tick()
    assert calls == [(MINT, SOL_MINT, IN_RAW)]
    state = persisted(world)
    (intent,) = intents_of(state)
    assert intent["status"] == "unresolved" and bot._sell_frozen(MINT)
    assert position(state)["raw_tokens"] == IN_RAW and state["portfolio"]["realized_pnl"] == 0.0
    assert state["risk"]["realized_pnl_today"] == 0.0 and bot.notifier.sells == []


# Written before the per-position exit backoff: these tests re-run an exit on
# the next tick, so the delay is zeroed (counter/alert still real).
pytestmark = pytest.mark.usefixtures("no_exit_backoff")
