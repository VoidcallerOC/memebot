"""P0-2: a perpetually refused live exit is bounded, persisted and alerted.

Each definitively failed live sell attempt (refused before send, or a
``confirmed_failure`` outcome) increments a per-position counter persisted in
state, pushes the next attempt out by a bounded exponential backoff
(``EXIT_RETRY_BASE_SECONDS * 2**(n-1)`` capped at ``EXIT_RETRY_MAX_SECONDS``)
and, once the counter reaches ``EXIT_FAILURE_ALERT_THRESHOLD``, sends ONE
operator alert for that failure streak. A settled sell resets it. The
position, cost basis and P&L are never touched by a refusal.

Offline fakes only (clock, chain, notifier); the ``real_swap_policy`` tests
drive the REAL ``SwapExecutor`` + direct-route policy with a fake Jupiter
that only offers multi-hop routes: nothing is built, signed or sent.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

import bot.main as main_module
from bot.config import Config, _validate
from bot.jupiter import SOL_MINT, USDC_MINT, JupiterClient, SwapExecutor, TransactionStatus
from tests.swap_tx_factory import direct_route
from tests.test_p1_stuck_sell_intents import (  # noqa: F401  (env is a fixture)
    MINT, SL_MARK, confirmed, env, failure, intents, make_bot, open_pos, sells_of,
)

_REAL_PRICE_USD = main_module._price_usd  # captured before the env fixture fakes it
REFUSED = failure(TransactionStatus.CONFIRMED_FAILURE, error="route policy: 2 steps")


class Refusing:
    """Refuses the first ``n`` sells (definitive, nothing sent), then fills."""

    def __init__(self, n):
        self.n, self.calls = n, []

    def swap(self, i, o, amount, *, allocation_usd=None, before_broadcast=None):
        self.calls.append((i, o, amount))
        return REFUSED if len(self.calls) <= self.n else confirmed(amount)


def stuck_bot(env, n_refusals):
    bot = make_bot(env)
    bot.executor = Refusing(n_refusals)
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    return bot


def pos(bot):
    return bot.portfolio.positions[MINT]


def untouched(bot):
    p = pos(bot)
    assert p.tokens == 2.0 and p.raw_tokens == 2_000_000 and p.size_usd == 2.0
    assert bot.portfolio.realized_pnl == 0.0 and bot.notifier.sells == []
    assert not bot._open_sell_intents() or all(
        i["status"] in ("unresolved", "blocked") for i in bot._open_sell_intents())
    assert [s for _, s in intents(bot)].count("settled") == 0


def stuck_alerts(bot):
    return [t for t in bot.notifier.sent if "EXIT STUCK" in t]


def test_defaults_and_backoff_schedule():
    cfg = Config()
    assert (cfg.exit_retry_base_seconds, cfg.exit_retry_max_seconds,
            cfg.exit_failure_alert_threshold) == (30.0, 900.0, 3)
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg
    assert [bot._exit_backoff_seconds(n) for n in range(1, 9)] == \
        [30, 60, 120, 240, 480, 900, 900, 900]
    assert bot._exit_backoff_seconds(10_000) == 900  # bounded, no overflow


@pytest.mark.parametrize("field,value", [
    ("exit_retry_base_seconds", 0), ("exit_retry_base_seconds", float("nan")),
    ("exit_retry_base_seconds", 4000), ("exit_retry_max_seconds", 10),
    ("exit_retry_max_seconds", 100_000), ("exit_failure_alert_threshold", 0),
])
def test_invalid_retry_config_rejected(field, value):
    cfg = dataclasses.replace(Config(), **{field: value})
    with pytest.raises(ValueError):
        _validate(cfg)


def test_repeated_refusals_backoff_timing_and_single_alert(env):
    bot = stuck_bot(env, n_refusals=10**6)
    expected_delays = [30, 60, 120, 240, 480, 900, 900, 900]
    for n, delay in enumerate(expected_delays, start=1):
        bot._tick()  # attempt n: refused
        assert len(sells_of(bot, MINT)) == n
        assert pos(bot).exit_failures == n
        assert pos(bot).exit_retry_at == pytest.approx(env.clock.now + delay)
        untouched(bot)
        # Every tick inside the backoff window: no new intent, no attempt.
        n_intents = len(bot.portfolio.pending_intents)
        for dt in (1, delay / 2, delay - 1):
            env.clock.now = pos(bot).exit_retry_at - delay + dt
            bot._tick()
            assert len(sells_of(bot, MINT)) == n
        assert len(bot.portfolio.pending_intents) == n_intents
        env.clock.now = pos(bot).exit_retry_at
        assert len(stuck_alerts(bot)) == (1 if n >= 3 else 0)  # exactly once per streak
    assert MINT in bot.portfolio.positions
    alert = stuck_alerts(bot)[0]
    assert "Position still OPEN" in alert and "route policy" in alert


def test_recovery_settles_exactly_once_and_resets_counter(env):
    bot = stuck_bot(env, n_refusals=4)
    for _ in range(4):
        bot._tick()
        env.clock.now = pos(bot).exit_retry_at
    assert pos(bot).exit_failures == 4 and pos(bot).exit_failure_alerted
    bot._tick()  # 5th attempt fills
    assert MINT not in bot.portfolio.positions
    assert [s for _, s in intents(bot)].count("settled") == 1
    assert len(bot.notifier.sells) == 1 and bot.portfolio.realized_pnl == pytest.approx(-1.0)
    for _ in range(3):
        env.clock.now += 1000
        bot._tick()
    assert len(sells_of(bot, MINT)) == 5 and len(bot.notifier.sells) == 1


def test_partial_take_profit_success_resets_streak_and_rearms_alert(env):
    bot = stuck_bot(env, n_refusals=3)
    env.marks[MINT] = 1.6  # +60%: first ladder rung (50% of the position)
    for _ in range(3):
        bot._tick()
        env.clock.now = pos(bot).exit_retry_at
    assert pos(bot).exit_failures == 3 and len(stuck_alerts(bot)) == 1
    bot._tick()  # rung fills
    p = pos(bot)
    assert p.raw_tokens == 1_000_000 and 50.0 in p.ladder_filled
    assert (p.exit_failures, p.exit_retry_at, p.exit_failure_alerted) == (0, None, False)
    assert len(bot.notifier.sells) == 1


def test_restart_preserves_counter_and_backoff(env):
    bot = stuck_bot(env, n_refusals=10**6)
    for _ in range(3):
        bot._tick()
        env.clock.now = pos(bot).exit_retry_at
    env.clock.now -= 100  # still inside the 120s window
    retry_at = pos(bot).exit_retry_at

    fresh = make_bot(env, load=True)
    fresh.executor = Refusing(10**6)
    p = fresh.portfolio.positions[MINT]
    assert (p.exit_failures, p.exit_retry_at, p.exit_failure_alerted) == (3, retry_at, True)
    fresh._tick()
    assert fresh.executor.calls == []          # backoff honoured after restart
    env.clock.now = retry_at
    fresh._tick()
    assert len(fresh.executor.calls) == 1 and fresh.portfolio.positions[MINT].exit_failures == 4
    assert stuck_alerts(fresh) == []            # streak already alerted: no re-alert


def test_possibly_sent_sell_is_not_counted_as_exit_failure(env):
    bot = make_bot(env, [failure(TransactionStatus.TIMEOUT, sig="sig-maybe")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]
    assert pos(bot).exit_failures == 0 and pos(bot).exit_retry_at is None
    assert bot._sell_frozen(MINT)


def test_legacy_state_without_retry_fields_loads_clean(env):
    bot = make_bot(env)
    open_pos(bot)
    bot._save()
    data = json.load(open(env.state))
    for p in data["portfolio"]["positions"]:
        for k in ("exit_failures", "exit_retry_at", "exit_failure_alerted"):
            p.pop(k, None)
    json.dump(data, open(env.state, "w"))
    fresh = make_bot(env, load=True)
    p = fresh.portfolio.positions[MINT]
    assert (p.exit_failures, p.exit_retry_at, p.exit_failure_alerted) == (0, None, False)


# -- P1-4: lifecycle with the REAL direct-route policy -------------------------

class MultiHopJup:
    """Jupiter fake that only knows multi-hop routes (e.g. token -> SOL -> X)."""

    def __init__(self):
        self.calls = []

    def _headers(self):
        return {}

    def quote(self, i, o, amount, **kw):
        self.calls.append((i, o, amount, kw.get("direct_only", False)))
        q = direct_route({"inputMint": i, "outputMint": o, "inAmount": str(amount),
                          "outAmount": "6000000", "otherAmountThreshold": "5900000",
                          "swapMode": "ExactIn", "priceImpactPct": "0"})
        q["routePlan"] = q["routePlan"] * 2
        return q


class PolicyThenFill:
    """First ``n`` sells go through the REAL SwapExecutor (real route policy
    refuses a multi-hop quote before /swap); then a direct route 'appears'
    (fake confirmed fill)."""

    def __init__(self, cfg, n):
        self.jup = MultiHopJup()
        self.real = SwapExecutor(cfg, self.jup, object())
        self.n, self.calls, self.results = n, [], []

    def swap(self, i, o, amount, *, allocation_usd=None, before_broadcast=None):
        self.calls.append((i, o, amount))
        if len(self.calls) <= self.n:
            r = self.real.swap(i, o, amount, allocation_usd=allocation_usd,
                               before_broadcast=before_broadcast)
            self.results.append(r)
            return r
        return confirmed(amount)


@pytest.mark.real_swap_policy
def test_real_policy_refusal_is_bounded_and_exit_recovers_once(env):
    import bot.jupiter as jupiter
    assert jupiter.route_quote_error.__module__ == "bot.swap_policy"  # real, not stubbed
    bot = make_bot(env)
    bot.executor = PolicyThenFill(bot.cfg, 4)
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    for n in range(1, 5):
        bot._tick()
        r = bot.executor.results[-1]
        assert r.status is TransactionStatus.CONFIRMED_FAILURE and r.tx_signature is None
        assert r.error.startswith("route policy:")
        assert pos(bot).exit_failures == n
        untouched(bot)
        env.clock.now = pos(bot).exit_retry_at
    # every executable quote asked Jupiter for direct routes only
    assert all(c[3] is True for c in bot.executor.jup.calls)
    assert len(stuck_alerts(bot)) == 1
    bot._tick()
    assert MINT not in bot.portfolio.positions
    assert [s for _, s in intents(bot)].count("settled") == 1 and len(bot.notifier.sells) == 1


class PriceSession:
    """HTTP fake for JupiterClient: only multi-hop routes exist for MINT
    (MINT -> SOL -> USDC). With onlyDirectRoutes Jupiter would find nothing."""

    def __init__(self):
        self.params = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.params.append(dict(params))
        if params.get("onlyDirectRoutes") == "true":
            return Resp({"error": "No routes found"})
        plan = [{"swapInfo": {"ammKey": "A1", "inputMint": params["inputMint"],
                              "outputMint": SOL_MINT, "inAmount": params["amount"],
                              "outAmount": "1"}, "percent": 100},
                {"swapInfo": {"ammKey": "A2", "inputMint": SOL_MINT,
                              "outputMint": params["outputMint"], "inAmount": "1",
                              "outAmount": "500000"}, "percent": 100}]
        return Resp({"inputMint": params["inputMint"], "outputMint": params["outputMint"],
                     "inAmount": params["amount"], "outAmount": "500000",
                     "otherAmountThreshold": "492500", "swapMode": "ExactIn",
                     "slippageBps": 150, "routePlan": plan})


class Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.mark.real_swap_policy
def test_price_lookup_not_direct_only_and_prices_indirect_only_token():
    s = PriceSession()
    jup = JupiterClient(Config(), s)
    price = main_module._price_usd(jup, MINT, 6)
    assert price == pytest.approx(0.5)
    (params,) = s.params
    assert "onlyDirectRoutes" not in params and params["outputMint"] == USDC_MINT


@pytest.mark.real_swap_policy
def test_executable_swap_sends_direct_only_and_rejects_multi_hop():
    s = PriceSession()
    cfg = Config()
    jup = JupiterClient(cfg, s)
    r = SwapExecutor(cfg, jup, s).swap(MINT, SOL_MINT, 1_000_000)  # paper: never armed
    assert s.params[-1]["onlyDirectRoutes"] == "true"
    assert not r.ok and r.error == "no route / quote unavailable"
    # Even if Jupiter ignored the flag and returned a multi-hop plan, the
    # policy refuses it before anything is built.
    class Ignore(PriceSession):
        def get(self, url, params=None, **k):
            p = dict(params)
            p.pop("onlyDirectRoutes", None)
            self.params.append(dict(params))
            return PriceSession.get(PriceSession(), url, p)
    s2 = Ignore()
    r2 = SwapExecutor(cfg, JupiterClient(cfg, s2), s2).swap(MINT, SOL_MINT, 1_000_000)
    assert s2.params[-1]["onlyDirectRoutes"] == "true"
    assert not r2.ok and r2.error.startswith("route policy:") and "2 steps" in r2.error


@pytest.mark.real_swap_policy
def test_stop_loss_fires_for_indirect_only_token_via_real_price_and_policy(env, monkeypatch):
    """Real _price_usd + real JupiterClient: a token with only multi-hop
    liquidity is still marked, the stop-loss fires and the executable sell is
    attempted (refused by the real policy -> bounded retry, position kept).
    Before the fix the mark quote was direct-only, so no mark -> no exit."""
    bot = make_bot(env)
    s = PriceSession()
    bot.jup = JupiterClient(bot.cfg, s)
    bot.executor = SwapExecutor(bot.cfg, bot.jup, s)
    open_pos(bot)
    monkeypatch.setattr(main_module, "_price_usd", _REAL_PRICE_USD)
    bot._tick()
    marks = [p for p in s.params if p["outputMint"] == USDC_MINT]
    assert marks and all("onlyDirectRoutes" not in p for p in marks)
    sells = [p for p in s.params if p["outputMint"] == SOL_MINT and p["inputMint"] == MINT]
    assert sells and all(p["onlyDirectRoutes"] == "true" for p in sells)  # SL fired
    assert pos(bot).exit_failures == 1
    untouched(bot)

