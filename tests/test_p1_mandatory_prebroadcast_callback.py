"""P1: the pre-broadcast persistence callback is MANDATORY for every armed swap.

``SwapExecutor.requires_pre_broadcast_callback(input_mint)``: armed (sells
since #39; armed BUYS too since the buy-path PR: ``return self.cfg.is_armed``).
Without a callable ``before_broadcast`` such a swap is refused:

* via ``swap``: before the quote (after the allocation / SOL-exposure gates):
  zero quotes, zero builds, zero signatures, zero broadcasts;
* via ``_execute_live`` directly: after the signer check, before the Jupiter
  build: zero builds, zero signatures, zero broadcasts.

With a callback, only a return value that ``is True`` lets the transaction be
broadcast; False / None / truthy non-True / an exception -> zero broadcasts,
a definitive ``CONFIRMED_FAILURE`` that the sell path turns into a blocked /
failed intent (rung retryable). Paper mode is unchanged. The buy path's own
lifecycle tests live in ``test_p1_buy_prebroadcast_signature``.

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


SELL_QUOTE = {"inputMint": MINT, "outputMint": SOL_MINT, "outAmount": "1"}


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
    result = ex._execute_live(SELL_QUOTE, IN_RAW, 1, input_mint=MINT, output_mint=SOL_MINT)
    assert result.status == TransactionStatus.CONFIRMED_FAILURE and result.tx_signature is None
    assert rw.builds == 0 and rw.sends == []          # nothing built, so nothing signed


def test_unknown_input_fails_closed_as_a_sell(rw):
    ex = executor(rw)
    assert ex.requires_pre_broadcast_callback(None) is True
    result = ex._execute_live({"outAmount": "1"}, IN_RAW, 1)      # no direction at all
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


# -- unchanged: paper; flipped by the buy-path PR: armed buys need the callback -------

def test_paper_sell_without_callback_unchanged(rw):
    cfg = Config(live_trading=False, bankroll_usd=20.0, max_position_pct=10.0)
    ex = SwapExecutor(cfg, jup=Jup(), session=rw)
    assert ex.requires_pre_broadcast_callback(MINT) is False
    result = ex.swap(MINT, SOL_MINT, IN_RAW, allocation_usd=2.0)
    assert result.ok and result.simulated and result.tx_signature == "DRYRUN"
    assert rw.builds == 0 and rw.sends == []


def test_armed_buy_without_callback_now_refused_before_quote_build_sign_broadcast(rw):
    """Flipped from #39's documented scope boundary: an armed buy (SOL input)
    without the callback is refused like a sell (after the SOL-exposure
    gate, whose SOL/USD price quote is the only quote made)."""
    ex = executor(rw)
    assert ex.requires_pre_broadcast_callback(SOL_MINT) is True
    loads = counting(ex)
    result = ex.swap(SOL_MINT, MINT, 1_000_000, allocation_usd=2.0)
    assert result.ok is False and result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature is None and "armed buy without a pre-broadcast" in result.error
    assert (SOL_MINT, MINT, 1_000_000) not in ex.jup.quotes        # no swap quote
    assert loads == [] and rw.builds == 0 and rw.sends == []


# -- the executed direction is the quote's, validated against the request ------------
# (review of #39: a direct caller must not be able to pass a token->SOL quote
# while claiming input_mint=SOL and so skip the mandatory sell callback)

OTHER = "Other11111111111111111111111111111111111111"
BUY_QUOTE = {"inputMint": SOL_MINT, "outputMint": MINT, "outAmount": "1"}


def counting(ex):
    loads = []
    real = ex._load_keypair
    ex._load_keypair = lambda K: loads.append(1) or real(K)
    return loads


def assert_refused_before_anything(w, result, loads):
    assert result.ok is False and result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature is None and "refused before signing" in result.error
    assert loads == [] and w.builds == 0 and w.sends == []      # no key, build, sign, send


@pytest.mark.parametrize("quote, args", [
    pytest.param({"inputMint": MINT, "outputMint": SOL_MINT, "outAmount": "1"}, (SOL_MINT, MINT),
                 id="reviewer-sell-quote-claimed-as-buy"),
    pytest.param({"inputMint": MINT, "outputMint": SOL_MINT, "outAmount": "1"}, (SOL_MINT, SOL_MINT),
                 id="sell-quote-claimed-sol-to-sol"),
    pytest.param({"inputMint": SOL_MINT, "outputMint": OTHER, "outAmount": "1"}, (SOL_MINT, MINT),
                 id="output-mismatch"),
    pytest.param({"inputMint": OTHER, "outputMint": MINT, "outAmount": "1"}, (SOL_MINT, MINT),
                 id="input-mismatch"),
    pytest.param({"outputMint": MINT, "outAmount": "1"}, (SOL_MINT, MINT), id="missing-inputMint"),
    pytest.param({"inputMint": SOL_MINT, "outAmount": "1"}, (SOL_MINT, MINT), id="missing-outputMint"),
    pytest.param({"inputMint": 1, "outputMint": MINT, "outAmount": "1"}, (SOL_MINT, MINT),
                 id="non-string-inputMint"),
    pytest.param({"inputMint": SOL_MINT, "outputMint": b"m", "outAmount": "1"}, (SOL_MINT, MINT),
                 id="non-string-outputMint"),
    pytest.param({"inputMint": "", "outputMint": MINT, "outAmount": "1"}, ("", MINT), id="empty"),
    pytest.param(BUY_QUOTE, (None, None), id="request-mints-missing"),
    pytest.param({"inputMint": SOL_MINT, "outputMint": SOL_MINT, "outAmount": "1"}, (SOL_MINT, SOL_MINT),
                 id="same-mint"),
    pytest.param("not-a-dict", (SOL_MINT, MINT), id="quote-not-object"),
])
def test_direction_mismatch_refused_before_key_build_sign_broadcast(rw, quote, args):
    ex = executor(rw)
    loads = counting(ex)
    result = ex._execute_live(quote, IN_RAW, 1, input_mint=args[0], output_mint=args[1])
    assert_refused_before_anything(rw, result, loads)


def test_swapped_mints_refused_even_with_a_true_callback(rw):
    ex = executor(rw)
    loads = counting(ex)
    calls = []
    result = ex._execute_live(BUY_QUOTE, IN_RAW, 1, input_mint=MINT, output_mint=SOL_MINT,
                              before_broadcast=lambda sig: calls.append(sig) or True)
    assert_refused_before_anything(rw, result, loads)
    assert calls == []


def test_consistent_sell_quote_with_true_callback_broadcasts_once(rw):
    ex = executor(rw)
    rw.wallet[MINT] = IN_RAW / 10**8
    seen = []
    result = ex._execute_live(SELL_QUOTE, IN_RAW, 1, input_mint=MINT, output_mint=SOL_MINT,
                              before_broadcast=lambda sig: seen.append(sig) or True)
    assert rw.sends == seen and len(seen) == 1 and result.tx_signature == seen[0]


def test_consistent_sell_quote_without_callback_refused_before_build(rw):
    ex = executor(rw)
    result = ex._execute_live(SELL_QUOTE, IN_RAW, 1, input_mint=MINT, output_mint=SOL_MINT)
    assert result.status == TransactionStatus.CONFIRMED_FAILURE and result.tx_signature is None
    assert rw.builds == 0 and rw.sends == [] and "pre-broadcast" in result.error


def test_consistent_buy_quote_without_callback_now_refused_before_build(rw):
    """Flipped by the buy-path PR: direct ``_execute_live`` with a consistent
    buy quote and no callback is refused after the signer check, before the
    build: zero builds, signatures, broadcasts."""
    ex = executor(rw)
    rw.plan = ["drop"]
    result = ex._execute_live(BUY_QUOTE, 1_000_000, 1, input_mint=SOL_MINT, output_mint=MINT)
    assert result.ok is False and result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature is None and "pre-broadcast" in result.error
    assert rw.builds == 0 and rw.sends == []


def test_swap_path_never_executes_a_quote_with_another_direction(rw):
    """swap() validates the fetched quote (swap_quote_error, which now uses the
    same direction check) before execution: a quote for the other direction
    is a clean pre-execution failure."""
    ex = executor(rw)

    class FlippedJup(Jup):
        def quote(self, i, o, amount, **_kw):
            return super().quote(o, i, amount)       # answers for the other direction
    ex.jup = FlippedJup()
    result = sell(ex, before_broadcast=lambda sig: True)
    assert result.status == TransactionStatus.CONFIRMED_FAILURE and "invalid quote" in result.error
    assert rw.builds == 0 and rw.sends == []


# Written before the per-position exit backoff: these tests re-run an exit on
# the next tick, so the delay is zeroed (counter/alert still real).
pytestmark = pytest.mark.usefixtures("no_exit_backoff")
