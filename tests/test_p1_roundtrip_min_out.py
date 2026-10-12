"""P1: the pre-buy round trip is judged on Jupiter's slippage minimums.

Jupiter Swap API v1 ``QuoteResponse``: ``outAmount`` is the best output and
"does not account for slippage"; ``otherAmountThreshold`` is the "minimum
acceptable output amount - outAmount after applying slippageBps tolerance"
(ExactIn). The gate must request the sell-back for exactly the buy's
threshold and decide on the sell's threshold, never falling back to
``outAmount``. Offline fakes only; ``_usd_to_lamports`` runs for real.
"""
from __future__ import annotations

from fractions import Fraction

import pytest
import requests

from bot.jupiter import SOL_MINT
from tests.test_p1_sellability import (
    MINT, PLANNED, TOKENS_PER_LAMPORT, MODES, RouteJup, _DROP, assert_rejected, gate,
    make_bot, run_entry, with_fields,
)

SLIP = 150  # Config.max_slippage_bps default


def test_expected_passes_but_slippage_minimum_fails_is_rejected():
    # 8% expected loss passes on outAmount; with 1.5% min-out on each leg the
    # guaranteed return is 0.92 * 0.985**2 ~ 0.893 < 0.9 -> reject.
    jup = RouteJup(sell_ratio=Fraction(92, 100), slippage_bps=SLIP)
    assert gate(make_bot(jup)) is False


@pytest.mark.parametrize("mode", MODES)
def test_expected_vs_minimum_discrepancy_blocks_entry(mode, monkeypatch):
    jup = RouteJup(sell_ratio=Fraction(92, 100), slippage_bps=SLIP)
    bot, swaps, spy = run_entry(jup, mode, monkeypatch)
    assert_rejected(bot, swaps, spy)


@pytest.mark.parametrize("mode", MODES)
def test_realistic_slippage_and_two_percent_loss_still_passes(mode, monkeypatch):
    # 0.98 * 0.985**2 ~ 0.951 >= 0.9 -> entry proceeds with the planned lamports.
    jup = RouteJup(sell_ratio=Fraction(98, 100), slippage_bps=SLIP)
    if mode == "paper":
        bot, swaps, _ = run_entry(jup, mode, monkeypatch)
        assert [s[0] for s in swaps] == [(SOL_MINT, MINT, PLANNED)]
        assert MINT in bot.portfolio.positions
    else:
        from bot.jupiter import SwapExecutor, SwapResult, TransactionStatus
        from tests.test_p1_sellability import armed_cfg
        cfg = armed_cfg()
        bot = make_bot(jup, cfg)
        bot._save = lambda: True               # armed entries need a durable buy intent
        executor = SwapExecutor(cfg, jup=jup, session=object())
        reached = []
        executor._execute_live = lambda q, amount, out, **kw: reached.append(amount) or SwapResult(
            ok=False, simulated=False, in_amount=amount, out_amount=out,
            status=TransactionStatus.CONFIRMED_FAILURE, error="test stop")
        bot.executor = executor
        bot._seek_entry()
        assert reached == [PLANNED]
    buy_min = PLANNED * TOKENS_PER_LAMPORT * (10_000 - SLIP) // 10_000
    assert jup.route_calls()[:2] == [(SOL_MINT, MINT, PLANNED), (MINT, SOL_MINT, buy_min)]


def test_sell_quote_requested_for_exactly_the_buy_threshold():
    jup = RouteJup(
        buy=with_fields(outAmount="20000000000", otherAmountThreshold="19700000123"),
        sell=lambda q, a: dict(q, outAmount=str(PLANNED), otherAmountThreshold=str(PLANNED)),
    )
    assert gate(make_bot(jup)) is True
    assert jup.route_calls() == [(SOL_MINT, MINT, PLANNED), (MINT, SOL_MINT, 19_700_000_123)]


@pytest.mark.parametrize(("lamports", "sell_min", "ok"), [
    (10_000_000, 9_000_000, True),      # exactly 10% on the minimum passes
    (10_000_000, 8_999_999, False),     # 10% + 1 lamport fails
    (PLANNED, 12_000_000, True),
    (PLANNED, 11_999_999, False),
])
def test_ten_percent_boundary_is_on_the_threshold_not_out_amount(lamports, sell_min, ok):
    # outAmount alone would always pass (no loss); only the threshold decides.
    jup = RouteJup(sell=with_fields(outAmount=str(lamports), otherAmountThreshold=str(sell_min)))
    assert gate(make_bot(jup), lamports=lamports) is ok


BAD_THRESHOLDS = [
    pytest.param(_DROP, id="missing"), pytest.param(None, id="None"), pytest.param("", id="empty"),
    pytest.param("0", id="zero"), pytest.param(0, id="int-zero"), pytest.param("-1", id="negative"),
    pytest.param(-1, id="neg-int"), pytest.param("1.5", id="decimal"), pytest.param(1.5, id="float"),
    pytest.param("1e6", id="exponent"), pytest.param(True, id="bool"), pytest.param("nan", id="nan"),
    pytest.param(float("inf"), id="inf"), pytest.param(" 5", id="space"),
    pytest.param(str(2 ** 64), id="over-u64"), pytest.param("above-out", id="greater-than-outAmount"),
]


@pytest.mark.parametrize("leg", ["buy", "sell"])
@pytest.mark.parametrize("value", BAD_THRESHOLDS)
def test_bad_threshold_rejects_without_out_amount_fallback(leg, value):
    def apply(q, _amount):
        q = dict(q)
        if value is _DROP:
            q.pop("otherAmountThreshold")
        elif value == "above-out":
            q["otherAmountThreshold"] = str(int(q["outAmount"]) + 1)
        else:
            q["otherAmountThreshold"] = value
        return q

    jup = RouteJup(sell_ratio=Fraction(1), **{leg: apply})  # outAmount itself is lossless
    assert gate(make_bot(jup)) is False


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("leg", ["buy", "sell"])
def test_missing_threshold_blocks_entry_in_both_modes(mode, leg, monkeypatch):
    jup = RouteJup(**{leg: with_fields(otherAmountThreshold=_DROP)})
    bot, swaps, spy = run_entry(jup, mode, monkeypatch)
    assert_rejected(bot, swaps, spy)


def test_threshold_equal_to_out_amount_is_valid():
    jup = RouteJup(slippage_bps=0)
    assert gate(make_bot(jup)) is True


@pytest.mark.parametrize("leg", ["buy", "sell"])
@pytest.mark.parametrize("failure", [None, RuntimeError("down"), requests.Timeout("t")])
def test_quote_failures_still_reject(leg, failure):
    override = failure if isinstance(failure, Exception) else (lambda q, a: None)
    assert gate(make_bot(RouteJup(slippage_bps=SLIP, **{leg: override}))) is False
