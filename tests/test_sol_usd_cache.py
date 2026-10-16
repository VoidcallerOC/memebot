"""Paper-only short-TTL SOL/USD cache (bot.main._SolUsdCache)."""
from __future__ import annotations

import math

import pytest

import bot.main as main_module
from bot.jupiter import SOL_MINT, USDC_MINT
from tests.test_paper_exit_full_quote import MINT, FakeJup, _bot, _cfg, sell_quote

pytestmark = pytest.mark.real_swap_policy
TTL = main_module.SOL_USD_CACHE_TTL_S


def _sol(out):
    return {"inputMint": SOL_MINT, "outputMint": USDC_MINT,
            "inAmount": str(10**9), "outAmount": out}


class SolJup:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def quote(self, i, o, amount, *, direct_only=False):
        assert (i, o) == (SOL_MINT, USDC_MINT)
        self.calls += 1
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


@pytest.fixture
def clock(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(main_module, "monotonic", lambda: t[0])
    return t


def test_ttl_is_in_range():
    assert 30 <= TTL <= 60


def test_hit_within_ttl_one_call(clock):
    c, j = main_module._SolUsdCache(), SolJup(_sol("200000000"))
    assert c.get(j) == 200_000_000
    clock[0] += TTL - 0.01
    assert c.get(j) == 200_000_000 and j.calls == 1


def test_miss_after_ttl_refetches(clock):
    c, j = main_module._SolUsdCache(), SolJup(_sol("200000000"), _sol("210000000"))
    c.get(j)
    clock[0] += TTL
    assert c.get(j) == 210_000_000 and j.calls == 2


@pytest.mark.parametrize("bad", [None, RuntimeError("429"), _sol("0"), _sol("-5"),
                                 _sol("nan"), _sol("1.5"), {"outAmount": "200000000"}])
def test_failure_or_invalid_not_cached(clock, bad):
    c, j = main_module._SolUsdCache(), SolJup(bad, _sol("200000000"))
    assert c.get(j) is None
    assert c.get(j) == 200_000_000 and j.calls == 2


def test_failed_refetch_after_expiry_is_none_not_stale(clock):
    c, j = main_module._SolUsdCache(), SolJup(_sol("200000000"), None, _sol("190000000"))
    c.get(j)
    clock[0] += TTL + 1
    assert c.get(j) is None
    assert c.get(j) == 190_000_000 and j.calls == 3


def test_paper_exit_warm_cache_one_extra_quote(clock):
    jup = FakeJup(sell=lambda raw: sell_quote(raw, 400_000_000))
    bot = _bot(jup)
    assert bot._sol_usdc_raw() == 200_000_000  # warm (e.g. from an entry)
    jup.calls.clear()
    bot._sell(MINT, 1.0, 0.85, "stop_loss")
    assert jup.calls == [(MINT, SOL_MINT, 100_000_000, True)]
    assert bot.portfolio.paper_exit_quotes[-1]["booked"] == "quote"


def test_paper_entry_sizing_reuses_cache(clock):
    jup = FakeJup()
    bot = _bot(jup)
    a, b = bot._usd_to_lamports(2.0), bot._usd_to_lamports(2.0)
    assert a == b == 10_000_000 and len(jup.calls) == 1


def test_live_path_never_cached(clock, monkeypatch):
    jup = FakeJup()
    bot = _bot(jup)
    monkeypatch.setattr(type(bot.cfg), "is_armed", property(lambda self: True))
    bot._usd_to_lamports(2.0); bot._usd_to_lamports(2.0); bot._sol_usdc_raw()
    assert len(jup.calls) == 3
    assert getattr(bot, "_sol_usd_cache", None) is None


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, math.nan])
def test_cache_guard_rejects_invalid_even_if_helper_returns_it(clock, monkeypatch, bad):
    seq = [bad, 200_000_000]
    monkeypatch.setattr(main_module, "sol_usdc_raw_per_sol", lambda _j: seq.pop(0))
    c = main_module._SolUsdCache()
    assert c.get(None) is None
    assert c.get(None) == 200_000_000 and seq == []


def test_live_sizing_bypasses_paper_cache_wrapper(clock, monkeypatch):
    jup = FakeJup()
    bot = _bot(jup)
    monkeypatch.setattr(type(bot.cfg), "is_armed", property(lambda self: True))
    monkeypatch.setattr(bot, "_sol_usdc_raw",
                        lambda: (_ for _ in ()).throw(AssertionError("live via cache")))
    assert bot._usd_to_lamports(2.0) == bot._usd_to_lamports(2.0) == 10_000_000
    assert len(jup.calls) == 2
