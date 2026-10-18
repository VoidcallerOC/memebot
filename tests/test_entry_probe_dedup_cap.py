"""Entry-probing request budget in ``_seek_entry``.

(1) A mint listed more than once by the candidate source is evaluated at most
once per cycle. (2) PAPER mode only: at most ``ENTRY_JUPITER_CANDIDATES_PER_CYCLE`` (= 1)
candidate per cycle begins Jupiter-dependent evaluation (SOL/USD sizing,
sell-route check, buy); the allowance is consumed once every local screen has
passed, even if that candidate then fails or is rate-limited. Local-screen
rejections never consume it; deferred candidates are eligible next cycle.
Armed live entries are not capped (ae80ee6 behavior); dedup applies in both.
Offline fakes only: no network, signing or sends.
"""
from __future__ import annotations

import logging
import types
from fractions import Fraction

import pytest
import requests

import bot.jupiter as J
import bot.main as main_module
from bot.jupiter import SOL_MINT, USDC_MINT, JupiterClient, SwapExecutor
from bot.risk import RiskManager
from bot.safety import TokenSafety
from tests.test_p1_sellability import PLANNED, RouteJup, armed_cfg, make_bot, paper_cfg

A = "AAAAMint1111111111111111111111111111111111111"
B = "BBBBMint1111111111111111111111111111111111111"
C = "CCCCMint1111111111111111111111111111111111111"
HELD = "HeldMint1111111111111111111111111111111111111"
COOL = "CoolMint1111111111111111111111111111111111111"
UNSAFE = "UnsafeMint11111111111111111111111111111111111"

FAIL_SELL = dict(sell=lambda q, a: None)          # route rejected after both legs
FAIL_BUY = dict(buy=lambda q, a: None)            # same value the client returns on a 429


def entry_bot(jup, mints, cfg=None, unsafe=()):
    bot = make_bot(jup, cfg)
    bot.decision_shadow = None
    bot.executor = SwapExecutor(bot.cfg, jup=jup, session=object())
    bot.screened, bot.route_checked = [], []
    bot.set_feed = lambda ms: setattr(bot, "strategy", types.SimpleNamespace(
        find_candidates=lambda: [types.SimpleNamespace(mint=m) for m in ms]))
    bot.set_feed(mints)

    def screen(mint, **_k):
        bot.screened.append(mint)
        return TokenSafety(mint=mint, passed=mint not in unsafe, liquidity_usd=50_000,
                           price_usd=0.002, symbol=mint[:4], decimals=6)
    bot.screener = types.SimpleNamespace(screen=screen)
    real = bot._sell_route_ok

    def route_ok(mint, lamports):
        bot.route_checked.append(mint)
        return real(mint, lamports)
    bot._sell_route_ok = route_ok
    return bot


def legs(jup, mint):
    return [(SOL_MINT, mint, PLANNED), (mint, SOL_MINT, PLANNED * 1000)]


# -- (1) duplicate suppression --------------------------------------------

def test_duplicate_mint_is_evaluated_once_per_cycle(monkeypatch, caplog):
    # Raise the cap so only the dedup can stop the repeat.
    monkeypatch.setattr(main_module, "ENTRY_JUPITER_CANDIDATES_PER_CYCLE", 5)
    jup = RouteJup(**FAIL_SELL)
    bot = entry_bot(jup, [A, A, A])
    with caplog.at_level(logging.DEBUG, logger="bot"):
        bot._seek_entry()
    assert bot.screened == [A]
    assert bot.route_checked == [A]
    assert jup.route_calls() == legs(jup, A)          # no repeated Jupiter request
    assert sum("skipped: duplicate in this cycle" in r.getMessage() for r in caplog.records) == 2


def test_duplicate_of_locally_rejected_mint_is_not_rescreened():
    bot = entry_bot(RouteJup(), [UNSAFE, UNSAFE, B], unsafe={UNSAFE})
    bot._seek_entry()
    assert bot.screened == [UNSAFE, B]
    assert B in bot.portfolio.positions


def test_duplicate_set_resets_every_cycle(monkeypatch):
    monkeypatch.setattr(main_module, "ENTRY_JUPITER_CANDIDATES_PER_CYCLE", 5)
    # Zero per-mint backoff so only the dedup reset is under test here.
    monkeypatch.setattr(main_module, "ENTRY_BACKOFF_BASE_S", 0.0)
    jup = RouteJup(**FAIL_SELL)
    bot = entry_bot(jup, [A, A])
    bot._seek_entry()
    bot._seek_entry()
    assert bot.route_checked == [A, A]
    assert jup.route_calls() == legs(jup, A) * 2


# -- (2) per-cycle Jupiter candidate cap ------------------------------------

def test_cap_constant_is_one():
    assert main_module.ENTRY_JUPITER_CANDIDATES_PER_CYCLE == 1


def test_only_first_candidate_begins_jupiter_evaluation(caplog):
    jup = RouteJup(**FAIL_SELL)
    bot = entry_bot(jup, [A, B, C])
    with caplog.at_level(logging.INFO, logger="bot"):
        bot._seek_entry()
    assert bot.screened == [A, B, C]                  # local screens still run
    assert bot.route_checked == [A]
    assert jup.route_calls() == legs(jup, A)
    deferred = [r.getMessage() for r in caplog.records
                if "deferred: per-cycle Jupiter candidate cap reached" in r.getMessage()]
    assert deferred == [f"entry candidate {B[:8]} deferred: per-cycle Jupiter candidate cap reached",
                        f"entry candidate {C[:8]} deferred: per-cycle Jupiter candidate cap reached"]
    assert bot.portfolio.positions == {}


@pytest.mark.parametrize("failure", [FAIL_BUY, FAIL_SELL, dict(sell_ratio=Fraction(1, 2))],
                         ids=["buy-leg-none-429", "sell-leg-none", "uneconomic"])
def test_failed_first_candidate_does_not_release_the_allowance(failure):
    jup = RouteJup(**failure)
    bot = entry_bot(jup, [A, B])
    bot._seek_entry()
    assert bot.route_checked == [A]
    assert all(B not in c for c in jup.calls)
    assert bot.portfolio.positions == {}


def test_failed_paper_buy_does_not_release_the_allowance():
    jup = RouteJup()
    bot = entry_bot(jup, [A, B])
    bot.executor.swap = lambda *a, **k: J.SwapResult(ok=False, simulated=True, in_amount=PLANNED, out_amount=0,
                                                           error="rate-limited (HTTP 429 / cooldown)")
    bot._seek_entry()
    assert bot.route_checked == [A]
    assert bot.portfolio.positions == {}


def test_sizing_failure_consumes_allowance_and_still_ends_search():
    jup = RouteJup()
    bot = entry_bot(jup, [A, B])
    sized = []
    bot._usd_to_lamports = lambda usd: sized.append(usd) or None
    bot._seek_entry()
    assert sized == [2.0] and bot.route_checked == []
    assert bot.portfolio.positions == {}


def test_cached_sol_usd_still_consumes_allowance():
    jup = RouteJup(**FAIL_SELL)
    bot = entry_bot(jup, [A, B])
    bot._seek_entry()                                 # fills the SOL/USD cache
    sol_usd = [c for c in jup.calls if c[1] == USDC_MINT]
    bot._seek_entry()                                 # cache hit this time
    assert [c for c in jup.calls if c[1] == USDC_MINT] == sol_usd == [(SOL_MINT, USDC_MINT, 10**9)]
    # One candidate per cycle even on the cache hit; cycle 2's slot goes to B
    # (A is backed off after its failure, B was retained when deferred).
    assert bot.route_checked == [A, B]


class Resp:
    def __init__(self, status=200, data=None):
        self.status_code, self._data, self.headers = status, data, {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._data


class Sess:
    def __init__(self, *resps):
        self.resps, self.params = list(resps), []

    def get(self, url, params=None, **_k):
        self.params.append(dict(params or {}))
        return self.resps.pop(0)


def test_real_client_http_429_on_first_candidate_defers_the_rest(monkeypatch):
    monkeypatch.setattr(J.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(J.random, "uniform", lambda a, b: 1.0)
    sol_usd = {"inputMint": SOL_MINT, "outputMint": USDC_MINT, "inAmount": str(10**9),
               "outAmount": str(150 * 10**6)}
    sess = Sess(Resp(200, sol_usd), Resp(429))
    jup = JupiterClient(paper_cfg(), sess)
    bot = entry_bot(jup, [A, B, C])
    bot._seek_entry()
    assert [p["outputMint"] for p in sess.params] == [USDC_MINT, A]   # exactly 2 HTTP requests
    assert bot.route_checked == [A]
    assert jup.rate_limit_remaining() > 0
    assert bot.portfolio.positions == {}


def test_local_screen_rejections_do_not_consume_allowance(monkeypatch):
    jup = RouteJup()
    bot = entry_bot(jup, [HELD, COOL, UNSAFE, B, C], unsafe={UNSAFE})
    bot.portfolio.open(HELD, "HELD", 0.002, 2.0, 1000.0, token_decimals=6)
    bot.portfolio.reentry_cooldowns[COOL] = 10**12
    monkeypatch.setattr(main_module.time, "time", lambda: 1.0)
    bot._seek_entry()
    assert bot.screened == [UNSAFE, B]                # HELD/COOL rejected before screening
    assert bot.route_checked == [B]
    assert B in bot.portfolio.positions and C not in bot.portfolio.positions


def test_decision_shadow_skip_does_not_consume_allowance():
    jup = RouteJup()
    bot = entry_bot(jup, [A, B])
    bot.decision_shadow = types.SimpleNamespace(
        evaluate_candidate=lambda safety, snapshot=None: types.SimpleNamespace(
            mint=safety.mint, decision=types.SimpleNamespace(action="skip"),
            risk=types.SimpleNamespace(status="test")),
        allows_paper_entry=lambda eng: eng.mint != A)
    bot._seek_entry()
    assert bot.route_checked == [B]
    assert B in bot.portfolio.positions


def test_deferred_candidate_is_evaluated_in_a_later_cycle():
    jup = RouteJup(sell=lambda q, a: None if q["inputMint"] == A else q)
    bot = entry_bot(jup, [A, B])
    bot._seek_entry()
    assert bot.route_checked == [A] and bot.portfolio.positions == {}
    bot.set_feed([B])                                 # A dropped out of the feed
    bot._seek_entry()
    assert bot.route_checked == [A, B]
    assert B in bot.portfolio.positions


def test_successful_first_candidate_opens_and_ends_cycle():
    jup = RouteJup()
    bot = entry_bot(jup, [A, B])
    bot._seek_entry()
    assert list(bot.portfolio.positions) == [A]
    assert bot.route_checked == [A] and B not in bot.screened


def test_armed_live_entry_next_moves_on_to_next_candidate_uncapped(caplog):
    # Armed live entries keep the ae80ee6 behavior: no per-cycle cap.
    jup = RouteJup()
    bot = entry_bot(jup, [A, B, C], cfg=armed_cfg())
    attempts = []
    bot._live_entry = lambda mint, *a: attempts.append(mint) or "next"
    with caplog.at_level(logging.DEBUG, logger="bot"):
        bot._seek_entry()
    assert attempts == [A, B, C] and bot.route_checked == [A, B, C]
    assert not any("deferred" in r.getMessage() for r in caplog.records)


def test_armed_route_failure_moves_on_to_next_candidate_uncapped():
    jup = RouteJup(sell=lambda q, a: None if q["inputMint"] == A else q)
    bot = entry_bot(jup, [A, B], cfg=armed_cfg())
    attempts = []
    bot._live_entry = lambda mint, *a: attempts.append(mint) or "stop"
    bot._seek_entry()
    assert bot.route_checked == [A, B] and attempts == [B]


def test_armed_duplicate_skipping_still_applies(caplog):
    jup = RouteJup(**FAIL_SELL)
    bot = entry_bot(jup, [A, A, B, B], cfg=armed_cfg())
    bot._live_entry = lambda *a: pytest.fail("route fails; no live entry")
    with caplog.at_level(logging.DEBUG, logger="bot"):
        bot._seek_entry()
    assert bot.screened == [A, B] and bot.route_checked == [A, B]
    assert jup.route_calls() == legs(jup, A) + legs(jup, B)
    assert sum("skipped: duplicate in this cycle" in r.getMessage() for r in caplog.records) == 2


# -- held-position monitoring unchanged -------------------------------------

def test_manage_positions_requests_unchanged_by_entry_cap():
    calls = []

    class PriceJup:
        last_error_kind = None

        def quote(self, i, o, amount, **kw):
            calls.append((i, o, amount, kw))
            return {"inputMint": i, "outputMint": o, "inAmount": str(amount), "outAmount": "2000"}

    bot = entry_bot(PriceJup(), [A, B, C])
    bot.risk = RiskManager(paper_cfg())
    for m in (HELD, COOL):
        bot.portfolio.open(m, m[:4], 0.002, 2.0, 1000.0, token_decimals=6)
    bot._manage_positions()
    expected = [(m, USDC_MINT, 10**6, {}) for m in (HELD, COOL)]   # paper: no bypass kwarg
    assert calls == expected
    calls.clear()
    bot._manage_positions()
    assert calls == expected
