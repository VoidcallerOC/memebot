"""Endpoint-wide Jupiter quote 429 cooldown, rate-limit vs no-route error
kinds in logs, and SOL/USD cache DEBUG lines. Offline: fake sessions only."""
from __future__ import annotations

import logging
import types

import pytest
import requests

import bot.jupiter as J
import bot.main as main_module
from bot.config import Config
from bot.jupiter import SOL_MINT, USDC_MINT, JupiterClient, SwapExecutor
from bot.portfolio import Portfolio
from bot.risk import RiskManager

MINT = "Mint111111111111111111111111111111111111111"
OK = {"inputMint": SOL_MINT, "outputMint": USDC_MINT,
      "inAmount": str(10**9), "outAmount": str(200 * 10**6)}


def _cfg(**kw):
    v = dict(live_trading=False, wallet_private_key="", bankroll_usd=100.0,
             max_slippage_bps=150)
    v.update(kw)
    return Config(**v)


class Resp:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code, self._data, self.headers = status, data, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._data


class Sess:
    def __init__(self, *resps):
        self.resps, self.calls = list(resps), 0

    def get(self, *a, **k):
        self.calls += 1
        r = self.resps.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(J.time, "monotonic", c)
    monkeypatch.setattr(J.random, "uniform", lambda a, b: 1.0)
    return c


def _q(jup, **kw):
    return jup.quote(SOL_MINT, USDC_MINT, 10**9, **kw)


def test_429_sets_cooldown_and_next_call_makes_no_request(clock, caplog):
    s = Sess(Resp(429))
    jup = JupiterClient(_cfg(), s)
    assert _q(jup) is None and jup.last_error_kind == J.QUOTE_RATE_LIMITED
    with caplog.at_level(logging.DEBUG, logger=J.log.name):
        assert _q(jup) is None
    assert s.calls == 1
    assert "jupiter quote skipped: rate-limit cooldown 2.0s remaining" in caplog.text


def test_cooldown_expiry_allows_request_and_success_resets(clock):
    s = Sess(Resp(429), Resp(200, OK), Resp(429))
    jup = JupiterClient(_cfg(), s)
    _q(jup)
    clock.t += 2.01
    assert _q(jup) == OK and jup.last_error_kind is None and s.calls == 2
    _q(jup)  # streak was reset -> back to 2s, not 4s
    assert jup.rate_limit_remaining() == pytest.approx(2.0)


def test_retry_after_honoured_and_capped(clock):
    jup = JupiterClient(_cfg(), Sess(Resp(429, headers={"Retry-After": "7"}),
                                     Resp(429, headers={"Retry-After": "3600"})))
    _q(jup)
    assert jup.rate_limit_remaining() == pytest.approx(7.0)
    clock.t += 7.01
    _q(jup)
    assert jup.rate_limit_remaining() == pytest.approx(60.0)


@pytest.mark.parametrize("bad", ["abc", "-5", "nan", ""])
def test_insane_retry_after_falls_back_to_exponential(clock, bad):
    jup = JupiterClient(_cfg(), Sess(Resp(429, headers={"Retry-After": bad})))
    _q(jup)
    assert jup.rate_limit_remaining() == pytest.approx(2.0)


def test_exponential_growth_capped(clock):
    jup = JupiterClient(_cfg(), Sess(*[Resp(429) for _ in range(8)]))
    seen = []
    for _ in range(7):
        _q(jup)
        seen.append(round(jup.rate_limit_remaining(), 3))
        clock.t += 61
    assert seen == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


@pytest.mark.parametrize("u,exp", [(0.75, 1.5), (1.25, 2.5)])
def test_jitter_bounds(clock, monkeypatch, u, exp):
    got = []
    monkeypatch.setattr(J.random, "uniform", lambda a, b: got.append((a, b)) or u)
    jup = JupiterClient(_cfg(), Sess(Resp(429)))
    _q(jup)
    assert got == [(0.75, 1.25)]
    assert jup.rate_limit_remaining() == pytest.approx(exp)


def test_jitter_never_exceeds_cap(clock, monkeypatch):
    monkeypatch.setattr(J.random, "uniform", lambda a, b: b)
    jup = JupiterClient(_cfg(), Sess(*[Resp(429) for _ in range(7)]))
    for _ in range(7):
        _q(jup)
        assert jup.rate_limit_remaining() <= 60.0
        clock.t += 61


@pytest.mark.parametrize("resp,kind", [
    (Resp(200, {}), J.QUOTE_NO_ROUTE),
    (Resp(500), J.QUOTE_HTTP_ERROR),
    (requests.Timeout("t"), J.QUOTE_TIMEOUT),
    (requests.ConnectionError("c"), J.QUOTE_NETWORK_ERROR),
])
def test_non_429_errors_unchanged_no_cooldown(clock, resp, kind):
    s = Sess(resp, Resp(200, OK))
    jup = JupiterClient(_cfg(), s)
    assert _q(jup) is None and jup.last_error_kind == kind
    assert jup.rate_limit_remaining() == 0
    assert _q(jup) == OK and s.calls == 2


def test_paper_executor_respects_cooldown_armed_bypasses():
    calls = []

    class FakeJup:
        last_error_kind = J.QUOTE_RATE_LIMITED

        def quote(self, *a, **k):
            calls.append(k)
            return None
    r = SwapExecutor(_cfg(), FakeJup(), Sess()).swap(MINT, SOL_MINT, 1000)
    assert not r.ok and "rate-limited" in r.error
    assert calls == [{"direct_only": True}]


def test_live_bypass_still_requests_during_cooldown(clock):
    s = Sess(Resp(429), Resp(200, OK))
    jup = JupiterClient(_cfg(), s)
    _q(jup)
    assert _q(jup, bypass_cooldown=True) == OK and s.calls == 2


# --- main.py logging -------------------------------------------------------

class RLJup:
    """Every quote fails as rate-limited (as during cooldown)."""
    last_error_kind = J.QUOTE_RATE_LIMITED

    def quote(self, *a, **k):
        return None


class NoRouteJup(RLJup):
    last_error_kind = J.QUOTE_NO_ROUTE


def _bot(jup):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = _cfg(take_profit_ladder=[(50.0, 0.5)])
    bot.portfolio, bot.risk, bot.jup = Portfolio(), RiskManager(bot.cfg), jup
    bot.portfolio.open(MINT, "PEPE", 1.0, 100.0, 100.0, token_decimals=6)
    return bot


@pytest.mark.parametrize("jup,want,not_want", [
    (RLJup(), "rate-limited", "no route"),
    (NoRouteJup(), "no route / quote unavailable", "rate-limited"),
])
def test_paper_exit_fallback_names_cause(caplog, jup, want, not_want):
    bot = _bot(jup)
    with caplog.at_level(logging.WARNING):
        price = bot._paper_exit_quote(MINT, bot.portfolio.positions[MINT], 0.5, 1.0, "tp")
    assert price == 1.0
    msg = [r.getMessage() for r in caplog.records if "exit quote unavailable" in r.getMessage()]
    assert msg and want in msg[0] and not_want not in msg[0]


@pytest.mark.parametrize("jup,want", [(RLJup(), "rate-limited"), (NoRouteJup(), "no route")])
def test_sell_route_gate_log_names_cause(caplog, jup, want):
    bot = _bot(jup)
    with caplog.at_level(logging.WARNING):
        assert bot._sell_route_ok(MINT, 10**7) is False
    line = [r.getMessage() for r in caplog.records if "sell-route check" in r.getMessage()][0]
    assert want in line and (want == "no route" or "no route" not in line)


def test_price_missing_debug_names_rate_limit(caplog):
    with caplog.at_level(logging.DEBUG):
        assert main_module._price_usd(RLJup(), MINT, 6) is None
    assert any("(missing): rate-limited" in r.getMessage() for r in caplog.records)


def test_sol_usd_cache_debug_lines(monkeypatch, caplog):
    t = [500.0]
    monkeypatch.setattr(main_module, "monotonic", lambda: t[0])
    vals = [None, 200 * 10**6]
    monkeypatch.setattr(main_module, "sol_usdc_raw_per_sol", lambda jup: vals.pop(0))
    c = main_module._SolUsdCache()
    with caplog.at_level(logging.DEBUG, logger=main_module.log.name):
        assert c.get(None) is None
        assert c.get(None) == 200 * 10**6
        t[0] += 3
        assert c.get(None) == 200 * 10**6
    msgs = [r.getMessage() for r in caplog.records if "sol/usd cache" in r.getMessage()]
    assert msgs == ["sol/usd cache miss -> fetch", "sol/usd cache fetch failed, not cached",
                    "sol/usd cache miss -> fetch", "sol/usd cache hit (age 3.0s)"]


def test_armed_executor_quote_bypasses_cooldown(monkeypatch):
    """Live safety: an armed executable sell quote is never suppressed by the
    client-side cooldown (a real 429 still fails closed -> exit retry backoff)."""
    monkeypatch.setattr(Config, "is_armed", property(lambda self: True))
    calls = []

    class FakeJup:
        last_error_kind = J.QUOTE_RATE_LIMITED

        def quote(self, *a, **k):
            calls.append(k)
            return None
    ex = SwapExecutor(_cfg(), FakeJup(), Sess())
    monkeypatch.setattr(ex, "_missing_pre_broadcast_callback", lambda *a: None)
    r = ex.swap(MINT, SOL_MINT, 1000, allocation_usd=1.0)
    assert not r.ok and "rate-limited" in r.error
    assert calls == [{"direct_only": True, "bypass_cooldown": True}]


# --- armed held-position exit marks bypass the cooldown; nothing else does ---

PRICE_OK = {"inputMint": MINT, "outputMint": USDC_MINT,
            "inAmount": str(10**6), "outAmount": str(10**6)}  # $1, == entry: no exit


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(Config, "is_armed", property(lambda self: True))


def _cooling_bot(clock, *resps):
    """Bot with a real JupiterClient already in a 429 cooldown (1 request)."""
    s = Sess(Resp(429), *resps)
    jup = JupiterClient(_cfg(), s)
    _q(jup)
    assert s.calls == 1 and jup.rate_limit_remaining() == pytest.approx(2.0)
    bot = _bot(jup)
    bot._sell_frozen = lambda mint: False
    bot._sell = lambda *a, **k: (_ for _ in ()).throw(AssertionError("sell"))
    return bot, s, jup


def test_armed_held_position_read_bypasses_cooldown(clock, armed):
    bot, s, jup = _cooling_bot(clock, Resp(200, PRICE_OK))
    bot._manage_positions()
    assert s.calls == 2 and jup.last_error_kind is None


def test_armed_bypassed_429_extends_cooldown_and_holds(clock, armed, caplog):
    bot, s, jup = _cooling_bot(clock, Resp(429))
    with caplog.at_level(logging.WARNING):
        bot._manage_positions()
    assert s.calls == 2 and jup.last_error_kind == J.QUOTE_RATE_LIMITED
    assert jup.rate_limit_remaining() == pytest.approx(4.0)  # streak 2
    assert "exit quote unavailable — holding position" in caplog.text
    assert MINT in bot.portfolio.positions


def test_paper_held_position_read_still_skipped(clock):
    bot, s, jup = _cooling_bot(clock, Resp(200, PRICE_OK))
    bot._manage_positions()
    assert s.calls == 1 and jup.last_error_kind == J.QUOTE_RATE_LIMITED


def test_price_usd_default_honours_cooldown(clock, armed):
    """Any other (e.g. entry-side) caller of _price_usd keeps the cooldown."""
    bot, s, jup = _cooling_bot(clock, Resp(200, PRICE_OK))
    assert main_module._price_usd(jup, MINT, 6) is None and s.calls == 1


def test_armed_sol_sizing_still_skipped(clock, armed):
    bot, s, jup = _cooling_bot(clock, Resp(200, OK))
    assert bot._usd_to_lamports(5.0) is None and s.calls == 1


def test_armed_sell_route_gate_still_skipped(clock, armed, caplog):
    bot, s, jup = _cooling_bot(clock, Resp(200, OK))
    with caplog.at_level(logging.WARNING):
        assert bot._sell_route_ok(MINT, 10**7) is False
    assert s.calls == 1 and "buy quote rejected: rate-limited" in caplog.text


def _route_quote(inp, outp, raw, out):
    return {"inputMint": inp, "outputMint": outp, "inAmount": str(raw),
            "outAmount": str(out), "otherAmountThreshold": str(out * 99 // 100),
            "swapMode": "ExactIn"}


def test_sell_route_gate_sell_leg_rate_limited(caplog):
    class Jup:
        last_error_kind = None

        def quote(self, i, o, amount, **k):
            if i == SOL_MINT:
                return _route_quote(SOL_MINT, MINT, amount, 5 * 10**6)
            self.last_error_kind = J.QUOTE_RATE_LIMITED
            return None
    bot = _bot(Jup())
    with caplog.at_level(logging.WARNING):
        assert bot._sell_route_ok(MINT, 10**7) is False
    line = [r.getMessage() for r in caplog.records if "sell-route check" in r.getMessage()][0]
    assert "sell quote rejected: rate-limited" in line and "no route" not in line
