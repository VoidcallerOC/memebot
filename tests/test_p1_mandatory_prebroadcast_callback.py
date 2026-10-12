"""P1: the pre-broadcast persistence callback is MANDATORY for armed sells.

``SwapExecutor.requires_pre_broadcast_callback(input_mint)``: armed and the
input is not native SOL (a sell; an unknown input fails closed as a sell).
Without a callable ``before_broadcast`` such a swap is refused:

* via ``swap``: before the quote (after the allocation / SOL-exposure gates):
  zero quotes, zero builds, zero signatures, zero broadcasts;
* via ``_execute_live`` directly: after the signer check, before the Jupiter
  build: zero builds, zero signatures, zero broadcasts.

With a callback, only a return value that ``is True`` lets the transaction be
broadcast; False / None / truthy non-True / an exception -> zero broadcasts,
a definitive ``CONFIRMED_FAILURE`` that the sell path turns into a blocked /
failed intent (rung retryable). Paper mode and armed BUYS (SOL input) are
unchanged here; making the callback mandatory for buys is PR B (the guard's
one-line change: ``return self.cfg.is_armed``).

Real signing with a throwaway keypair and a fake RPC that verifies every
broadcast, as in ``test_p1_prebroadcast_signature``.
"""
from __future__ import annotations

import pytest

from bot.config import Config
from bot.jupiter import SOL_MINT, SwapExecutor, TransactionStatus
from tests.test_p1_native_sol_settlement_e2e import (
    IN_RAW, MINT, RAW_DOUBLE, TP_MARK, Jup, intents_of, persisted, position,
)
from tests.test_p1_prebroadcast_signature import (  # noqa: F401  (rw is a fixture)
    TP50_PNL, boot, booked_once, rw, start,
)
from tests.test_p1_state_durability import install as install_faulty_os


def executor(w):
    ex = boot(w).executor
    ex.jup = Jup()
    return ex


def sell(ex, **kw):
    return ex.swap(MINT, SOL_MINT, IN_RAW, allocation_usd=2.0, **kw)


def assert_refused_before_quote(w, ex, result):
    assert result.ok is False and result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature is None and "pre-broadcast" in result.error
    assert ex.jup.quotes == [] and w.builds == 0 and w.sends == []


# -- missing callback: refused before quote / build / sign / broadcast ----------------

@pytest.mark.parametrize("cb", [None, True, "persist", 1], ids=["missing", "True", "str", "int"])
def test_armed_sell_without_callable_callback_refused_before_quote(rw, cb):
    ex = executor(rw)
    assert ex.requires_pre_broadcast_callback(MINT) is True
    assert_refused_before_quote(rw, ex, sell(ex, before_broadcast=cb))


def test_execute_live_direct_without_callback_refused_before_build(rw):
    ex = executor(rw)
    result = ex._execute_live({"outAmount": "1"}, IN_RAW, 1, input_mint=MINT, output_mint=SOL_MINT)
    assert result.status == TransactionStatus.CONFIRMED_FAILURE and result.tx_signature is None
    assert rw.builds == 0 and rw.sends == []          # nothing built, so nothing signed


def test_unknown_input_fails_closed_as_a_sell(rw):
    ex = executor(rw)
    assert ex.requires_pre_broadcast_callback(None) is True
    result = ex._execute_live({"outAmount": "1"}, IN_RAW, 1)
    assert result.status == TransactionStatus.CONFIRMED_FAILURE and rw.builds == 0 and rw.sends == []


# -- callback present: only `is True` broadcasts --------------------------------------

class Boom(RuntimeError):
    pass


def _raise(sig):
    raise Boom("disk on fire")


@pytest.mark.parametrize("ret", [False, None, 1, "yes", [1], _raise],
                         ids=["False", "None", "one", "yes", "list", "raises"])
def test_callback_not_exactly_true_means_zero_broadcasts(rw, ret):
    ex = executor(rw)
    seen = []

    def cb(sig):
        seen.append((sig, len(rw.sends)))
        return ret(sig) if callable(ret) else ret

    result = sell(ex, before_broadcast=cb)
    assert rw.builds == 1 and ex.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]
    (sig, sends_at_call), = seen                      # signed, signature handed over once
    assert sends_at_call == 0 and rw.sends == []      # ... and never broadcast
    assert result.ok is False and result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature == sig and "nothing sent" in result.error


def test_callback_true_broadcasts_exactly_the_signature_it_was_given(rw):
    ex = executor(rw)
    rw.wallet[MINT] = IN_RAW / 10**8                  # the sale lands on the fake chain
    seen = []
    result = sell(ex, before_broadcast=lambda sig: seen.append((sig, len(rw.sends))) or True)
    (sig, sends_at_call), = seen
    assert sends_at_call == 0                         # callback strictly before the broadcast
    assert rw.sends == [sig] and rw.durable_at_send == [False]   # (no bot: nothing on disk)
    assert result.tx_signature == sig and result.status == TransactionStatus.CONFIRMED_SUCCESS


# -- production callback: non-durable save -> zero broadcasts, retried once ----------

@pytest.mark.parametrize("step", ["dir_fsync", "file_fsync"])
def test_production_callback_not_durable_zero_broadcasts_then_retry_once(rw, monkeypatch, step):
    bot = start(rw)
    install_faulty_os(monkeypatch, {2: step}, path=rw.state)   # save 2 = the signature save
    bot._tick()
    assert rw.sends == [] and rw.builds == 1
    (intent,) = bot.portfolio.pending_intents
    assert intent["status"] == "blocked" and intent["tx_signature"]
    install_faulty_os(monkeypatch)
    bot._tick()
    (sig,) = rw.sends
    assert sig != intent["tx_signature"]
    state = booked_once(rw, sig)
    assert [i["status"] for i in intents_of(state)] == ["failed", "settled"]


def test_sell_path_with_callback_stripped_sends_nothing_rung_stays_eligible(rw):
    """If a caller ever dropped the callback, the executor refuses; the sell
    path records a definitive no-send failure and the rung is retried."""
    bot = start(rw, raw=RAW_DOUBLE, mark=TP_MARK)
    real_swap = bot.executor.swap
    bot.executor.swap = lambda *a, before_broadcast=None, **k: real_swap(*a, **k)
    bot._tick()
    assert rw.sends == [] and bot.jup.quotes == [] and rw.builds == 0
    (intent,) = intents_of(persisted(rw))
    assert intent["tx_signature"] is None and intent["outcome"] == "confirmed_failure"
    assert position(persisted(rw))["ladder_filled"] == []
    bot.executor.swap = real_swap
    bot._tick()                                       # resolver: not executed -> rung retried
    (sig,) = rw.sends
    state = booked_once(rw, sig, pnl=TP50_PNL, remaining_raw=IN_RAW, rung=50.0)
    assert [i["status"] for i in intents_of(state)] == ["failed", "settled"]


def test_sell_e2e_with_production_callback_unchanged(rw):
    bot = start(rw)
    bot._tick()
    (sig,) = rw.sends
    booked_once(rw, sig)


# -- unchanged: paper, and armed buys (PR B scope) ------------------------------------

def test_paper_sell_without_callback_unchanged(rw):
    cfg = Config(live_trading=False, bankroll_usd=20.0, max_position_pct=10.0)
    ex = SwapExecutor(cfg, jup=Jup(), session=rw)
    assert ex.requires_pre_broadcast_callback(MINT) is False
    result = ex.swap(MINT, SOL_MINT, IN_RAW, allocation_usd=2.0)
    assert result.ok and result.simulated and result.tx_signature == "DRYRUN"
    assert rw.builds == 0 and rw.sends == []


def test_armed_buy_without_callback_still_broadcasts_pr_b_scope(rw):
    """Documented scope boundary: armed buys (SOL input) do not require the
    callback in this PR. PR B makes it mandatory for every armed submission."""
    ex = executor(rw)
    assert ex.requires_pre_broadcast_callback(SOL_MINT) is False
    rw.plan = ["drop"]
    result = ex.swap(SOL_MINT, MINT, 1_000_000, allocation_usd=2.0)
    (sig,) = rw.sends
    assert result.tx_signature == sig and rw.builds == 1
    assert result.status == TransactionStatus.TIMEOUT       # not visible within timeout 0
