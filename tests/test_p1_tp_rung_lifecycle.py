"""P1: take-profit rungs are consumed only with the booked sale.

Before: ``_manage_positions`` added the rung to ``ladder_filled`` when the
threshold was hit, before the sell; the pre-send intent save persisted it. A
crash, a non-durable intent save, a pre-submit refusal, a failed or expired
sell, or a sell refused by the per-mint freeze therefore skipped that rung
forever.

Now: paper marks the rung after the sell's accounting; live intents carry
``rung`` and the rung is marked in ``_book_confirmed_sell`` inside the same
accounting snapshot and the same atomic save as the settlement (on-time,
delayed resolver and restart recovery alike). Failed / refused sells leave the
rung eligible; unresolved ones freeze the mint (no second sell).

Real paths: ``TradingBot`` constructor (``load_state``), ``run``/``_tick``,
``RiskManager.evaluate_exit`` (default ladder 50:0.5, 100:0.25, 300:0.25),
``_manage_positions``, ``_sell``, ``_settle_live_sell``, the resolver,
``SwapExecutor`` -> ``extract_fill`` on the pinned native-SOL fixture (and
copies whose seller token amount is rescaled for other rung sizes) and
``save_state``. Fakes: RPC session, quote, signing spy, send stub.
"""
from __future__ import annotations

import copy
import json
import logging

import pytest

import bot.main as main_module
from bot.config import Config
from bot.jupiter import SOL_MINT
from tests.test_p1_native_sol_settlement_e2e import (  # noqa: F401  (world is a fixture)
    ENTRY, FULL, IN_RAW, MINT, OWNER, RAW_DOUBLE, SIG, SL_MARK, TP_MARK, boot,
    fixture_result, intents_of, open_position, persisted, position, run, show, world,
)
from tests.test_p1_state_durability import install as install_faulty_os

TP50_PNL = FULL * (TP_MARK - ENTRY)
SIG_A, SIG_B, SIG_C = ("RungSaleA" + "1" * 79, "RungSaleB" + "1" * 79, "RungSaleC" + "1" * 79)
RAW_QUAD = 4 * IN_RAW               # 1,097,700 raw: rung 50 sells 548,850; 100 and 300 sell 274,425
MARK_110 = ENTRY * 2.1              # +110%: rungs 50 and 100
MARK_310 = ENTRY * 4.1              # +310%: all rungs


def tx_selling(raw):
    """The fixture transaction with the seller's token amount set to ``raw``
    (sold entirely); everything else (native SOL out, fee) unchanged."""
    tx = copy.deepcopy(fixture_result())
    for key, amount in (("preTokenBalances", raw), ("postTokenBalances", 0)):
        for bal in tx["meta"][key]:
            if bal["mint"] == MINT and bal["owner"] == OWNER:
                bal["uiTokenAmount"].update(amount=str(amount), uiAmount=amount / 10**8 or None,
                                            uiAmountString=str(amount / 10**8))
    return tx


def sells(w, sig, raw, visible=True, lands=True):
    """The next send returns ``sig``; on chain it sells ``raw`` (or not)."""
    w.sig_queue.append(sig)
    w.txs.setdefault(sig, tx_selling(raw))

    def effect(world):
        if lands:
            world.wallet[MINT] = round(world.wallet[MINT] - raw / 10**8, 12)
        if visible:
            show(world, sig)
    w.on_send[sig] = effect


def ladder(state=None, w=None):
    return position(state if state is not None else persisted(w))["ladder_filled"]


def settled_rungs(state):
    return sorted(i["rung"] for i in intents_of(state) if i["status"] == "settled")


def assert_rung_invariant(w):
    """On disk: a rung is filled iff a settled intent for it is on record, and
    the raw holding / P&L agree with the settled intents."""
    state = persisted(w)
    settled = [i for i in intents_of(state) if i["status"] == "settled"]
    pos = position(state)
    if pos is not None:
        assert sorted(pos["ladder_filled"]) == sorted(
            i["rung"] for i in settled if i.get("rung") is not None), state
        assert pos["raw_tokens"] == pos["raw_tokens_initial"] - sum(i["booked_raw"] for i in settled)
    assert state["portfolio"]["realized_pnl"] == pytest.approx(sum(i["booked_pnl"] for i in settled))
    assert state["risk"]["realized_pnl_today"] == pytest.approx(state["portfolio"]["realized_pnl"])
    return state


def assert_tp50_once(w, *, sends=None, raw=RAW_DOUBLE):
    state = assert_rung_invariant(w)
    settled = [i for i in intents_of(state) if i["status"] == "settled"]
    assert len(settled) == 1
    (s,) = settled
    sold = raw // 2
    assert (s["rung"], s["raw_amount"], s["booked_raw"]) == (50.0, sold, sold)
    assert s["booked_pnl"] == pytest.approx(sold / 10**8 * (TP_MARK - ENTRY))
    assert ladder(state) == [50.0]
    assert position(state)["raw_tokens"] == raw - sold
    assert not [i for i in intents_of(state) if i["status"] in main_module.OPEN_SELL_INTENT_STATUSES]
    if sends is not None:
        assert w.sends == sends
    return state


def fail_saves(monkeypatch, failing, crash=False):
    """save_state on the given 1-based calls: returns False without writing,
    or (crash=True) the process dies before the write."""
    real = main_module.save_state
    count = [0]

    def save(*a, **kw):
        count[0] += 1
        if count[0] in failing:
            if crash:
                raise SystemExit(f"crash before save {count[0]}")
            return False
        return real(*a, **kw)

    monkeypatch.setattr(main_module, "save_state", save)
    return real


def crash_after_save(monkeypatch, k):
    real = main_module.save_state
    count = [0]

    def save(*a, **kw):
        ok = real(*a, **kw)
        count[0] += 1
        if count[0] == k:
            raise SystemExit(f"crash after save {k}")
        return ok

    monkeypatch.setattr(main_module, "save_state", save)
    return real


def live_tp(world, raw=RAW_DOUBLE, mark=TP_MARK):
    bot = boot(world)
    open_position(bot, world, raw=raw)
    world.marks[MINT] = mark
    return bot


# -- the rung is not consumed by the threshold, nor by the pre-send save --------------

def test_presend_intent_save_carries_rung_but_does_not_consume_it(world, monkeypatch):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    crash_after_save(monkeypatch, 1)                 # dies right after the durable intent
    with pytest.raises(SystemExit):
        bot._tick()
    state = persisted(world)
    (intent,) = intents_of(state)
    assert (intent["status"], intent["rung"], intent["raw_amount"]) == ("pending", 50.0, IN_RAW)
    assert ladder(state) == []                       # NOT consumed
    assert world.sends == [] and world.signs == []


def test_crash_after_threshold_before_intent_saved_retries_rung_once(world, monkeypatch):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    real = fail_saves(monkeypatch, {1}, crash=True)  # dies before anything is written
    with pytest.raises(SystemExit):
        bot._tick()
    monkeypatch.setattr(main_module, "save_state", real)
    assert intents_of(persisted(world)) == [] and ladder(w=world) == []
    bot2 = boot(world)
    assert run(bot2, 3) == 3
    assert bot2.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]
    assert_tp50_once(world, sends=[SIG])


def test_crash_after_intent_saved_resolves_failed_and_retries_rung_once(world, monkeypatch):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    real = crash_after_save(monkeypatch, 1)
    with pytest.raises(SystemExit):
        bot._tick()
    monkeypatch.setattr(main_module, "save_state", real)
    world.now += main_module.SELL_INTENT_EXPIRY_SECONDS + 1
    run(boot(world), 4)
    state = assert_tp50_once(world, sends=[SIG])
    assert [i["status"] for i in intents_of(state)] == ["failed", "settled"]


# -- pre-submit refusal: zero quote/sign/send, rung stays eligible ---------------------

def test_presend_durability_failure_sends_nothing_and_rung_is_retried(world, monkeypatch):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    real = fail_saves(monkeypatch, {1})
    bot._tick()
    assert bot.jup.quotes == [] and world.signs == [] and world.sends == []
    assert bot.portfolio.positions[MINT].ladder_filled == set()
    (intent,) = bot.portfolio.pending_intents
    assert intent["status"] == "blocked" and intent["rung"] == 50.0
    assert ladder(w=world) == []
    monkeypatch.setattr(main_module, "save_state", real)
    bot._tick()                                       # cleared -> rung re-evaluated -> sold
    assert bot.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]
    state = assert_tp50_once(world, sends=[SIG])
    assert [i["status"] for i in intents_of(state)] == ["failed", "settled"]
    for _ in range(3):
        bot._tick()
    assert world.sends == [SIG] and len(bot.notifier.sells) == 1


def test_presend_post_replace_fsync_failure_sends_nothing_rung_retried(world, monkeypatch):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    install_faulty_os(monkeypatch, {1: "dir_fsync"}, path=world.state)  # the pre-send save
    bot._tick()
    assert world.sends == [] and bot.jup.quotes == []
    assert bot.portfolio.positions[MINT].ladder_filled == set()
    bot._tick()
    assert_tp50_once(world, sends=[SIG])


def test_presubmit_refusal_keeps_rung_eligible(world, monkeypatch):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    bot.portfolio.positions[MINT].raw_tokens = None    # no exact raw holding: refused
    bot._tick()
    assert bot.jup.quotes == [] and world.sends == []
    assert bot.portfolio.positions[MINT].ladder_filled == set()
    bot.portfolio.positions[MINT].raw_tokens = RAW_DOUBLE   # operator re-verified it
    bot._tick()
    assert_tp50_once(world, sends=[SIG])


# -- definitive failure vs uncertain outcome ------------------------------------------

def test_definitive_no_sale_leaves_rung_eligible_then_sells_once(world):
    """Sent but expired unconfirmed with the wallet unchanged: proven NOT
    executed -> failed, rung eligible, sold once by a new transaction."""
    bot = live_tp(world)
    sells(world, SIG_A, IN_RAW, visible=False, lands=False)
    bot._tick()
    (intent,) = intents_of(persisted(world))
    assert (intent["status"], intent["rung"]) == ("unresolved", 50.0)
    assert ladder(w=world) == []
    world.now += main_module.SELL_INTENT_EXPIRY_SECONDS + 1
    sells(world, SIG, IN_RAW)
    for _ in range(4):
        bot._tick()
    assert world.sends == [SIG_A, SIG]
    state = assert_tp50_once(world)
    assert [i["status"] for i in intents_of(state)] == ["failed", "settled"]
    assert state["portfolio"]["realized_pnl"] == pytest.approx(TP50_PNL)


def test_uncertain_outcome_never_resubmits_until_resolved(world):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW, visible=False)          # landed, status not yet visible
    bot._tick()
    for _ in range(4):                                # inside the window: frozen
        bot._tick()
    assert world.sends == [SIG] and bot.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]
    assert bot.portfolio.positions[MINT].ladder_filled == set()
    assert [i["status"] for i in intents_of(persisted(world))] == ["unresolved"]
    show(world)
    for _ in range(3):
        bot._tick()
    assert_tp50_once(world, sends=[SIG])
    assert len(bot.notifier.sells) == 1
    bot2 = boot(world)                                # restart: no resell
    run(bot2, 3)
    assert bot2.jup.quotes == [] and bot2.notifier.sells == []
    assert_tp50_once(world, sends=[SIG])


# -- booking not durable: rung rolled back with the accounting -----------------------

@pytest.mark.parametrize("fault", ["save_false", "dir_fsync_after_replace"])
def test_booking_save_failure_rolls_rung_back_and_recovers_once(world, monkeypatch, fault):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    if fault == "save_false":
        real = fail_saves(monkeypatch, {4})           # pending, signature, confirmed_unbooked, booking
    else:
        install_faulty_os(monkeypatch, {4: "dir_fsync"}, path=world.state)  # the booking save
    bot._tick()
    pos = bot.portfolio.positions[MINT]
    assert pos.ladder_filled == set() and pos.raw_tokens == RAW_DOUBLE
    assert bot.portfolio.realized_pnl == 0.0 and bot.risk.realized_pnl_today == 0.0
    (intent,) = bot.portfolio.pending_intents
    assert intent["status"] == "confirmed_unbooked" and intent["rung"] == 50.0
    if fault == "save_false":
        assert ladder(w=world) == []                  # disk: evidence only
        monkeypatch.setattr(main_module, "save_state", real)
    else:
        assert ladder(w=world) == [50.0]              # new version visible, not durable
        install_faulty_os(monkeypatch)
    assert_rung_invariant(world)
    for _ in range(3):
        bot._tick()
    assert_tp50_once(world, sends=[SIG])
    assert bot.portfolio.positions[MINT].ladder_filled == {50.0}
    assert bot.portfolio.realized_pnl == pytest.approx(TP50_PNL)
    assert len(bot.notifier.sells) == 1
    run(boot(world), 3)
    assert_tp50_once(world, sends=[SIG])


# -- crash after every save: settlement and rung completion are one write ------------

@pytest.mark.parametrize("path", ["on_time", "delayed"])
def test_crash_after_every_save_converges_to_one_sale_one_rung(world, monkeypatch, tmp_path, path):
    k = 1
    while True:
        d = tmp_path / f"{path}-{k}"
        d.mkdir()
        world.__init__(d)
        bot = live_tp(world)
        sells(world, SIG, IN_RAW, visible=path == "on_time")
        world.sig_queue.append(SIG)                   # a resend would be visible as a 2nd SIG
        if path == "delayed":
            bot._tick()
            assert ladder(w=world) == []
            show(world)
        real = crash_after_save(monkeypatch, k)
        try:
            bot._tick()
            crashed = False
        except SystemExit:
            crashed = True
        monkeypatch.setattr(main_module, "save_state", real)
        if not crashed:
            break
        assert_rung_invariant(world)                  # never rung-without-sale or vice versa
        world.now += main_module.SELL_INTENT_EXPIRY_SECONDS + 1
        run(boot(world), 4)
        assert_tp50_once(world, sends=[SIG])
        run(boot(world), 2)                           # durable completion: no resell
        assert_tp50_once(world, sends=[SIG])
        k += 1
    assert k - 1 == {"on_time": 4, "delayed": 2}[path]
    assert_tp50_once(world, sends=[SIG])


def test_booked_rung_is_in_the_same_write_as_the_settlement(world, monkeypatch):
    """Every state write of the settling tick is captured: none has the rung
    without the settled booking, or the settled booking without the rung."""
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    writes = []
    real = main_module.save_state

    def spy(path, portfolio, risk):
        ok = real(path, portfolio, risk)
        writes.append(persisted(world))
        return ok

    monkeypatch.setattr(main_module, "save_state", spy)
    bot._tick()
    assert [[i["status"] for i in intents_of(s)] for s in writes] == [
        ["pending"], ["pending"], ["confirmed_unbooked"], ["settled"]]
    assert [intents_of(s)[0]["tx_signature"] for s in writes] == [None, SIG, SIG, SIG]
    assert [ladder(s) for s in writes] == [[], [], [], [50.0]]
    for s in writes:
        assert sorted(ladder(s)) == settled_rungs(s)


# -- multiple rungs ------------------------------------------------------------------

def test_two_rungs_in_one_tick_each_sold_and_consumed_once(world):
    bot = live_tp(world, raw=RAW_QUAD, mark=MARK_110)
    sells(world, SIG_A, 2 * IN_RAW)                   # rung 50: 0.5 * 1,097,700
    sells(world, SIG_B, IN_RAW)                       # rung 100: 0.25 * 1,097,700
    bot._tick()
    assert bot.jup.quotes == [(MINT, SOL_MINT, 2 * IN_RAW), (MINT, SOL_MINT, IN_RAW)]
    state = assert_rung_invariant(world)
    assert world.sends == [SIG_A, SIG_B]
    assert ladder(state) == [50.0, 100.0]
    assert position(state)["raw_tokens"] == RAW_QUAD - 3 * IN_RAW
    pnl = 3 * IN_RAW / 10**8 * (MARK_110 - ENTRY)
    assert state["portfolio"]["realized_pnl"] == pytest.approx(pnl)
    assert state["risk"]["realized_pnl_today"] == pytest.approx(pnl)
    for _ in range(3):
        bot._tick()
    assert world.sends == [SIG_A, SIG_B]              # no double consume
    world.marks[MINT] = MARK_310
    sells(world, SIG_C, IN_RAW)
    bot._tick()                                       # rung 300: the remainder
    state = assert_rung_invariant(world)
    assert world.sends == [SIG_A, SIG_B, SIG_C]
    assert position(state) is None                    # fully sold through the ladder
    assert [i["rung"] for i in intents_of(state)] == [50.0, 100.0, 300.0]


def test_second_rung_frozen_by_unresolved_first_is_not_skipped(world):
    bot = live_tp(world, raw=RAW_QUAD, mark=MARK_110)
    sells(world, SIG_A, 2 * IN_RAW, visible=False)    # rung 50 outcome unknown
    bot._tick()
    assert world.sends == [SIG_A]                     # rung 100 refused by the freeze
    assert bot.portfolio.positions[MINT].ladder_filled == set()
    assert [i["rung"] for i in intents_of(persisted(world))] == [50.0]
    show(world, SIG_A)
    sells(world, SIG_B, IN_RAW)
    for _ in range(4):
        bot._tick()
    state = assert_rung_invariant(world)
    assert world.sends == [SIG_A, SIG_B]
    assert ladder(state) == [50.0, 100.0]
    assert settled_rungs(state) == [50.0, 100.0]
    assert position(state)["raw_tokens"] == IN_RAW


def test_stop_loss_after_a_rung_sells_exact_remaining_raw(world):
    bot = live_tp(world)
    sells(world, SIG, IN_RAW)
    bot._tick()
    assert_tp50_once(world, sends=[SIG])
    world.marks[MINT] = SL_MARK
    sells(world, SIG_A, IN_RAW)
    bot._tick()
    state = persisted(world)
    assert position(state) is None and world.sends == [SIG, SIG_A]
    assert [(i["reason"], i.get("rung"), i["booked_raw"]) for i in intents_of(state)] == [
        ("take_profit:50", 50.0, IN_RAW), ("stop_loss", None, IN_RAW)]
    pnl = TP50_PNL + FULL * (SL_MARK - ENTRY)
    assert state["portfolio"]["realized_pnl"] == pytest.approx(pnl)
    assert state["risk"]["realized_pnl_today"] == pytest.approx(pnl)


# -- paper -------------------------------------------------------------------------

def paper_bot(w):
    cfg = Config(live_trading=False, bankroll_usd=20.0, max_position_pct=10.0,
                 poll_interval_seconds=0, state_file=w.state, process_lock_file=w.lock)
    bot = main_module.TradingBot(cfg)
    bot.jup = None
    bot.notifier = type("N", (), {"enabled": False, "sell": lambda *a: None,
                                  "halt": lambda *a: None})()
    bot.portfolio.open(MINT, "XSCP", ENTRY, 100.0 * ENTRY, 100.0, token_decimals=8)
    return bot


def test_paper_rung_marked_only_after_accounting_and_retried_after_failure(world, monkeypatch):
    bot = paper_bot(world)
    world.marks[MINT] = TP_MARK
    real = bot.portfolio.sell_fraction
    monkeypatch.setattr(bot.portfolio, "sell_fraction",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("paper fail")))
    with pytest.raises(RuntimeError):
        bot._manage_positions()
    assert bot.portfolio.positions[MINT].ladder_filled == set()
    assert bot.portfolio.realized_pnl == 0.0
    monkeypatch.setattr(bot.portfolio, "sell_fraction", real)
    bot._manage_positions()
    pos = bot.portfolio.positions[MINT]
    assert pos.ladder_filled == {50.0} and pos.tokens == pytest.approx(50.0)
    assert bot.portfolio.realized_pnl == pytest.approx(50.0 * (TP_MARK - ENTRY))
    bot._manage_positions()                           # not sold twice
    assert pos.tokens == pytest.approx(50.0)
    assert ladder(w=world) == [50.0]
    assert bot.risk.realized_pnl_today == pytest.approx(bot.portfolio.realized_pnl)


def test_paper_multiple_rungs_and_stop_loss_unchanged(world):
    bot = paper_bot(world)
    world.marks[MINT] = MARK_110
    bot._manage_positions()
    pos = bot.portfolio.positions[MINT]
    assert pos.ladder_filled == {50.0, 100.0} and pos.tokens == pytest.approx(25.0)
    world.marks[MINT] = SL_MARK
    bot._manage_positions()
    assert MINT not in bot.portfolio.positions


# -- legacy state ---------------------------------------------------------------------

def _legacy_state(world, intents, filled):
    bot = live_tp(world)
    bot.portfolio.positions[MINT].ladder_filled = set(filled)
    bot.portfolio.pending_intents = intents
    bot._save()
    raw = json.load(open(world.state))
    for i in raw["portfolio"]["pending_intents"]:
        i.pop("rung", None)                           # pre-upgrade intents have no rung
    json.dump(raw, open(world.state, "w"))


def test_legacy_filled_rung_without_sale_is_kept_and_flagged_not_resold(world, caplog):
    _legacy_state(world, [], {50.0})
    caplog.set_level(logging.WARNING)
    bot = boot(world)
    assert "rung 50 is marked filled without a settled sale" in caplog.text
    world.marks[MINT] = TP_MARK
    run(bot, 3)
    assert world.sends == [] and bot.jup.quotes == []  # never re-sold automatically
    state = persisted(world)
    assert ladder(state) == [50.0] and position(state)["raw_tokens"] == RAW_DOUBLE
    assert state["portfolio"]["realized_pnl"] == 0.0 and intents_of(state) == []


def test_legacy_confirmed_unbooked_intent_books_once_rung_stays_filled(world, caplog):
    intent = {"mint": MINT, "fraction": 0.5, "reason": "take_profit:50", "requested_price": TP_MARK,
              "status": "confirmed_unbooked", "tx_signature": SIG, "created_at": 1.0,
              "raw_amount": IN_RAW, "actual_in_amount": IN_RAW, "actual_out_amount": 8_656_067,
              "fee_lamports": 10_143}
    _legacy_state(world, [intent], {50.0})            # old code consumed it at the threshold
    world.wallet[MINT] = FULL                         # the sale landed
    caplog.set_level(logging.WARNING)
    run(boot(world), 3)
    state = persisted(world)
    (s,) = intents_of(state)
    assert s["status"] == "settled" and "rung" not in s and s["booked_raw"] == IN_RAW
    assert ladder(state) == [50.0] and position(state)["raw_tokens"] == IN_RAW
    assert state["portfolio"]["realized_pnl"] == pytest.approx(TP50_PNL)
    assert world.sends == []
    caplog.clear()
    boot(world)                                       # backed by a settled sale: no flag
    assert "marked filled without a settled sale" not in caplog.text


def test_legacy_load_never_fabricates_rungs_or_sales(world):
    _legacy_state(world, [], set())
    bot = boot(world)
    assert bot.portfolio.positions[MINT].ladder_filled == set()
    assert bot.portfolio.pending_intents == [] and bot.portfolio.realized_pnl == 0.0
