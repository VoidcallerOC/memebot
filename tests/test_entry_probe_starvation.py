"""Paper entry fairness under the one-candidate Jupiter cap (``_seek_entry``).

A paper candidate whose Jupiter-stage evaluation fails is backed off per mint
(bounded exponential delay, bounded memory) and skipped BEFORE the cap, so it
cannot starve later candidates. Candidates deferred by the cap or skipped by
backoff are retained in a bounded, TTL-expiring queue and re-offered in later
cycles through every local screen (one-shot sources emit a mint once).
Armed live entries use neither. Offline fakes only: no network, signing or sends.
"""
from __future__ import annotations

import asyncio
import logging
import types

import pytest

import bot.main as main_module
from bot.jupiter import QUOTE_RATE_LIMITED, SOL_MINT, USDC_MINT
from tests.test_entry_probe_dedup_cap import A, B, C, FAIL_SELL, entry_bot
from tests.test_p1_sellability import RouteJup, armed_cfg

POLL_S = 15.0
A_ONLY_FAIL = dict(buy=lambda q, a: None if q["outputMint"] == A else q)


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(main_module, "monotonic", c)
    return c


def cycles(bot, clock, n, step=POLL_S):
    """Run ``n`` entry cycles; per cycle, the mints that reached the route check."""
    out = []
    for _ in range(n):
        before = len(bot.route_checked)
        bot._seek_entry()
        out.append(bot.route_checked[before:])
        clock.t += step
    return out


def one_shot_feed(bot, first):
    """smart_money-like source: emits ``first`` once, then nothing."""
    batches = [list(first)]
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [
        types.SimpleNamespace(mint=m) for m in (batches.pop(0) if batches else [])])


def msgs(caplog, text):
    return [r.getMessage() for r in caplog.records if text in r.getMessage()]


# -- starvation is bounded ---------------------------------------------------

def test_always_failing_first_candidate_cannot_block_later_ones(clock):
    bot = entry_bot(RouteJup(**A_ONLY_FAIL), [A, B, C])
    per_cycle = cycles(bot, clock, 3)
    assert per_cycle[0] == [A]
    assert per_cycle[1] == [B]                        # bound: B evaluated by cycle 2
    assert B in bot.portfolio.positions


def test_all_failing_candidates_rotate_within_bounded_cycles(clock):
    bot = entry_bot(RouteJup(**FAIL_SELL), [A, B, C])
    per_cycle = cycles(bot, clock, 120)               # 30 minutes of 15 s cycles
    assert per_cycle[:3] == [[A], [B], [C]]           # each mint reached within 3 cycles
    assert all(len(c) <= 1 for c in per_cycle)        # cap still one per cycle
    flat = [m for c in per_cycle for m in c]
    for m in (A, B, C):
        idx = [i for i, c in enumerate(per_cycle) if m in c]
        gaps = [b - a for a, b in zip(idx, idx[1:])]
        # never re-evaluated sooner than its backoff, never later than the cap (+ a cycle)
        assert max(gaps) * POLL_S <= main_module.ENTRY_BACKOFF_MAX_S + 2 * POLL_S
        assert flat.count(m) >= 4


def test_backoff_skip_does_not_consume_the_slot(clock, caplog):
    jup = RouteJup(**A_ONLY_FAIL)
    bot = entry_bot(jup, [A])
    bot._seek_entry()                                 # A fails -> backed off
    clock.t += POLL_S
    bot.set_feed([A, B])
    calls_before = len(jup.calls)
    with caplog.at_level(logging.DEBUG, logger="bot"):
        bot._seek_entry()
    assert bot.route_checked == [A, B]
    assert msgs(caplog, f"entry candidate {A[:8]} skipped: per-mint backoff")
    assert not msgs(caplog, "deferred: per-cycle Jupiter candidate cap reached")
    cycle_calls = [c for c in jup.calls[calls_before:] if c[1] != USDC_MINT]
    assert [c[:2] for c in cycle_calls] == [(SOL_MINT, B), (B, SOL_MINT), (SOL_MINT, B)]
    assert B in bot.portfolio.positions               # route legs + paper buy, B only


# -- backoff bounds -----------------------------------------------------------

def test_backoff_delays_are_exponential_and_capped(clock, caplog):
    bot = entry_bot(RouteJup(**FAIL_SELL), [A])
    delays = []
    with caplog.at_level(logging.INFO, logger="bot"):
        for _ in range(40):
            bot._note_entry_failure(A, "AAAA", clock.t)
            until, streak = bot._entry_backoff[A]
            delays.append(until - clock.t)
            clock.t = until                            # retried as soon as allowed
    assert delays[:6] == [30.0, 60.0, 120.0, 240.0, 480.0, 900.0]
    assert max(delays) == main_module.ENTRY_BACKOFF_MAX_S
    assert set(delays[5:]) == {main_module.ENTRY_BACKOFF_MAX_S}
    assert streak == 40
    assert msgs(caplog, f"entry backoff for {A[:8]} (AAAA): 900s after Jupiter-stage failure")


def test_backoff_memory_is_bounded_and_forgotten_after_quiet_period(clock):
    bot = entry_bot(RouteJup(), [])
    for i in range(1000):
        bot._note_entry_failure(f"M{i:043d}", "M", clock.t)
    assert len(bot._entry_backoff) == main_module.ENTRY_BACKOFF_MAX_MINTS
    clock.t += main_module.ENTRY_BACKOFF_BASE_S + main_module.ENTRY_BACKOFF_MAX_S
    bot._note_entry_failure(A, "AAAA", clock.t)       # evicts every forgotten record
    assert list(bot._entry_backoff) == [A]


def test_rate_limited_failure_gets_base_delay_without_raising_streak(clock, caplog):
    jup = RouteJup()
    # The route check's own buy quote hits the endpoint-wide 429 cooldown.
    jup.buy = lambda q, a: setattr(jup, "last_error_kind", QUOTE_RATE_LIMITED)
    bot = entry_bot(jup, [A])
    with caplog.at_level(logging.INFO, logger="bot"):
        for _ in range(5):
            bot._seek_entry()
            assert bot._entry_backoff[A] == (clock.t + main_module.ENTRY_BACKOFF_BASE_S, 0)
            clock.t += main_module.ENTRY_BACKOFF_BASE_S
    assert bot.route_checked == [A] * 5
    assert msgs(caplog, "rate limit (endpoint-wide; streak not raised)")


def test_success_clears_backoff(clock):
    fail = {"on": True}
    jup = RouteJup(buy=lambda q, a: None if fail["on"] else q)
    bot = entry_bot(jup, [A])
    bot._seek_entry()
    assert A in bot._entry_backoff
    fail["on"] = False
    clock.t += 1.0
    bot._seek_entry()                                 # still backed off
    assert bot.route_checked == [A] and A not in bot.portfolio.positions
    clock.t += main_module.ENTRY_BACKOFF_BASE_S
    bot._seek_entry()
    assert A in bot.portfolio.positions
    assert A not in bot._entry_backoff


def test_failed_paper_swap_backs_off(clock):
    bot = entry_bot(RouteJup(), [A, B])
    bot.executor = types.SimpleNamespace(swap=lambda i, o, amt, **k: types.SimpleNamespace(
        ok=False, error="no route / quote unavailable", simulated=True))
    per_cycle = cycles(bot, clock, 2)
    assert per_cycle == [[A], [B]]
    assert A in bot._entry_backoff and B in bot._entry_backoff


def test_sol_usd_unavailable_is_not_the_mints_fault(clock):
    bot = entry_bot(RouteJup(), [A, B])
    bot._usd_to_lamports = lambda usd: None
    bot._seek_entry()
    assert getattr(bot, "_entry_backoff", {}) == {}   # no per-mint backoff
    assert A in bot._entry_queue                      # still retained, not lost
    assert B not in bot.screened                      # cycle still ends (unchanged)


# -- retention queue -------------------------------------------------------------

def test_one_shot_deferred_candidates_are_evaluated_later(clock):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, [A, B, C])
    per_cycle = cycles(bot, clock, 3)
    assert per_cycle == [[A], [B], [C]]
    assert bot._entry_queue == {}


def test_one_shot_deferred_candidate_expires_explicitly(clock, caplog):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, [A, B])
    bot._seek_entry()
    assert list(bot._entry_queue) == [B]
    clock.t += main_module.ENTRY_RETRY_QUEUE_TTL_S
    with caplog.at_level(logging.INFO, logger="bot"):
        bot._seek_entry()
    assert msgs(caplog, f"entry retry queue: {B[:8]} expired (TTL 1800s) before being (re)evaluated")
    assert bot.route_checked == [A] and bot.screened.count(B) == 1
    assert bot._entry_queue == {}


def test_requeued_candidate_keeps_its_original_expiry(clock, caplog):
    # B stays backed off longer than the TTL: it must expire, not live forever.
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, [B])
    bot._entry_backoff = {B: (clock.t + 10 * main_module.ENTRY_RETRY_QUEUE_TTL_S, 5)}
    with caplog.at_level(logging.INFO, logger="bot"):
        for _ in range(int(main_module.ENTRY_RETRY_QUEUE_TTL_S // POLL_S) + 2):
            bot._seek_entry()
            clock.t += POLL_S
    assert msgs(caplog, f"entry retry queue: {B[:8]} expired")
    assert bot._entry_queue == {} and bot.route_checked == []


def test_retry_queue_size_is_bounded(clock, caplog):
    mints = [f"Q{i:043d}" for i in range(100)]
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, mints)
    with caplog.at_level(logging.INFO, logger="bot"):
        bot._seek_entry()
    assert len(bot._entry_queue) == main_module.ENTRY_RETRY_QUEUE_MAX
    assert list(bot._entry_queue) == mints[1:1 + main_module.ENTRY_RETRY_QUEUE_MAX]
    assert len(msgs(caplog, "dropped, not retained")) == 100 - 1 - main_module.ENTRY_RETRY_QUEUE_MAX


def test_reoffered_candidates_rerun_local_screens(clock):
    unsafe = set()
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, [A, B])
    real_screen = bot.screener.screen

    def screen(mint, **k):
        s = real_screen(mint, **k)
        s.passed = mint not in unsafe
        return s
    bot.screener = types.SimpleNamespace(screen=screen)
    bot._seek_entry()
    unsafe.add(B)                                     # B turns unsafe while queued
    clock.t += POLL_S
    bot._seek_entry()
    assert bot.screened == [A, B, B]                  # screened again, fresh result
    assert bot.route_checked == [A]
    assert bot._entry_queue == {}                     # rejected by screens: not retained


def test_queue_and_feed_duplicates_are_suppressed(clock):
    bot = entry_bot(RouteJup(**FAIL_SELL), [A, B])
    bot._seek_entry()                                 # B deferred -> queued
    clock.t += POLL_S
    bot._seek_entry()                                 # B from queue and from feed
    # B once per cycle (queue first); A is backed off before screening.
    assert bot.screened == [A, B, B]
    assert bot.route_checked == [A, B]


def test_queued_candidate_now_held_is_dropped_by_existing_screen(clock):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, [A, B])
    bot._seek_entry()
    bot.portfolio.open(B, "BBBB", 0.002, 2.0, 1000.0, token_decimals=6)
    clock.t += POLL_S
    bot._seek_entry()
    assert bot.screened == [A, B] and bot._entry_queue == {}


# -- live (armed) unaffected -------------------------------------------------------

def test_armed_live_has_no_backoff_and_no_queue(clock, caplog):
    bot = entry_bot(RouteJup(**FAIL_SELL), [A, B, C], cfg=armed_cfg())
    bot._live_entry = lambda *a: pytest.fail("route fails; no live entry")
    with caplog.at_level(logging.DEBUG, logger="bot"):
        per_cycle = cycles(bot, clock, 3)
    assert per_cycle == [[A, B, C]] * 3
    assert getattr(bot, "_entry_backoff", None) is None
    assert getattr(bot, "_entry_queue", None) is None
    assert not msgs(caplog, "backoff") and not msgs(caplog, "retry queue")


def test_armed_ignores_existing_paper_state(clock):
    bot = entry_bot(RouteJup(**FAIL_SELL), [A], cfg=armed_cfg())
    bot._live_entry = lambda *a: pytest.fail("route fails; no live entry")
    bot._entry_backoff = {A: (clock.t + 900.0, 3)}
    bot._entry_queue = {B: clock.t + 900.0}
    bot._seek_entry()
    assert bot.route_checked == [A]                   # not skipped by backoff
    assert B not in bot.screened                      # queue not re-offered


# -- review fix D1: backoff check runs before any snapshot / screen / journal ----

def test_backed_off_candidate_costs_no_snapshot_holders_screen_or_journal(clock, monkeypatch):
    import bot.meta.adapters as adapters
    snaps, holders, journal = [], [], []
    monkeypatch.setattr(adapters, "snapshot_from_dexscreener",
                        lambda mint, session, now=None: snaps.append(mint) or types.SimpleNamespace(mint=mint))
    monkeypatch.setattr(adapters, "attach_holder_concentration",
                        lambda snap, rpc, session: holders.append(snap.mint))
    bot = entry_bot(RouteJup(**A_ONLY_FAIL), [A])
    del bot._candidate_market_snapshot                # the real method (DexScreener + holders)
    bot.session = None
    bot.decision_shadow = types.SimpleNamespace(
        evaluate_candidate=lambda safety, snapshot=None: journal.append(safety.mint) or "eng",
        allows_paper_entry=lambda eng: True)
    bot._seek_entry()                                 # A fails -> backed off
    assert snaps == holders == journal == bot.screened == [A]
    assert A in bot._entry_backoff
    bot._entry_queue[A] = expiry = clock.t + 5.0      # also queued: must be kept as is
    clock.t += 1.0
    bot.set_feed([A, B])
    bot._seek_entry()
    assert snaps == holders == journal == bot.screened == [A, B]   # zero work for A
    assert bot.route_checked == [A, B]                # B got the slot this cycle
    assert B in bot.portfolio.positions
    assert bot._entry_queue == {A: expiry}            # retained, expiry not extended


def test_backed_off_feed_candidate_is_retained_without_screening(clock):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, [A])
    bot._entry_backoff = {A: (clock.t + 60.0, 1)}
    bot._seek_entry()
    assert bot.screened == [] and list(bot._entry_queue) == [A]
    assert bot._entry_queue[A] == clock.t + main_module.ENTRY_RETRY_QUEUE_TTL_S


# -- review fix D2: a queued entry is only removed on a final disposition -------

def _queued_b(clock):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, [A, B])
    bot._seek_entry()                                 # A evaluated, B deferred -> queued
    expiry = bot._entry_queue[B]
    clock.t += POLL_S
    return bot, expiry


def test_snapshot_exception_keeps_queued_candidate_with_original_expiry(clock, caplog):
    # v3: a queued paper candidate's snapshot error no longer aborts the cycle
    # (v2 asserted propagation); it stays queued with its original expiry.
    bot, expiry = _queued_b(clock)
    bot._candidate_market_snapshot = lambda m: (_ for _ in ()).throw(RuntimeError("dexscreener down"))
    with caplog.at_level(logging.INFO, logger="bot"):
        bot._seek_entry()                             # no exception escapes
    assert bot._entry_queue == {B: expiry}
    assert msgs(caplog, f"entry retry queue: {B[:8]} snapshot failed (RuntimeError)")
    assert not msgs(caplog, "dexscreener down")       # type name only, no message
    bot._candidate_market_snapshot = lambda m: None
    clock.t += POLL_S
    bot._seek_entry()
    assert bot.route_checked == [A, B] and bot._entry_queue == {}


def test_screen_exception_keeps_queued_candidate_with_original_expiry(clock, caplog):
    # v3: as above for screener.screen (v2 asserted propagation each cycle).
    bot, expiry = _queued_b(clock)
    real = bot.screener
    bot.screener = types.SimpleNamespace(screen=lambda m, **k: (_ for _ in ()).throw(ValueError("bad")))
    with caplog.at_level(logging.INFO, logger="bot"):
        for _ in range(3):
            bot._seek_entry()
            clock.t += POLL_S
    assert bot._entry_queue == {B: expiry}            # no duplicate, never extended
    assert len(msgs(caplog, f"entry retry queue: {B[:8]} screen failed (ValueError)")) == 3
    bot.screener = real
    clock.t = expiry
    bot._seek_entry()                                 # TTL reached: explicit expiry
    assert bot._entry_queue == {} and bot.route_checked == [A]


def test_exceptions_never_grow_the_queue_past_its_bound(clock):
    # v3: the 32 queued mints are isolated; the raise that still escapes comes
    # from feed mint Q0 (never queued), which keeps the pre-existing behavior.
    mints = [f"Q{i:043d}" for i in range(40)]
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    one_shot_feed(bot, mints)
    bot._seek_entry()
    before = dict(bot._entry_queue)
    assert len(before) == main_module.ENTRY_RETRY_QUEUE_MAX
    bot._candidate_market_snapshot = lambda m: (_ for _ in ()).throw(RuntimeError("x"))
    bot.set_feed(mints)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            bot._seek_entry()
    assert bot._entry_queue == before


@pytest.mark.parametrize("disposition, reason", [
    ("held", "already held"),
    ("cooldown", "stop-loss re-entry cooldown"),
    ("frozen", "buy frozen"),
    ("unsafe", "safety screen rejected"),
    ("shadow_skip", "decision shadow skip"),
    ("shadow_error", "decision shadow failed closed"),
    ("route_fail", "evaluated: sell-route check failed"),
    ("buy_fail", "evaluated: paper buy failed"),
    ("opened", "evaluated: position opened"),
])
def test_queued_candidate_removal_is_logged_with_reason(clock, caplog, monkeypatch, disposition, reason):
    bot, _ = _queued_b(clock)
    bot.route_checked.clear()
    if disposition == "held":
        bot.portfolio.open(B, "BBBB", 0.002, 2.0, 1000.0, token_decimals=6)
    elif disposition == "cooldown":
        bot._reentry_cooldown_active = lambda m: m == B
    elif disposition == "frozen":
        bot._buy_frozen = lambda m: m == B
    elif disposition == "unsafe":
        real = bot.screener.screen

        def screen(m, **k):
            out = real(m, **k)
            out.passed = False
            return out
        bot.screener = types.SimpleNamespace(screen=screen)
    elif disposition in ("shadow_skip", "shadow_error"):
        def evaluate(safety, snapshot=None):
            if disposition == "shadow_error":
                raise RuntimeError("engine")
            return types.SimpleNamespace(decision=types.SimpleNamespace(action="skip"),
                                         risk=types.SimpleNamespace(status="test"))
        bot.decision_shadow = types.SimpleNamespace(evaluate_candidate=evaluate,
                                                    allows_paper_entry=lambda eng: False)
    elif disposition == "buy_fail":
        bot.jup.sell = None
        bot.executor = types.SimpleNamespace(swap=lambda *a, **k: types.SimpleNamespace(
            ok=False, error="no route / quote unavailable", simulated=True))
    elif disposition == "opened":
        bot.jup.sell = None
    with caplog.at_level(logging.INFO, logger="bot"):
        bot._seek_entry()
    assert msgs(caplog, f"entry retry queue: {B[:8]} removed ({reason})")
    assert bot._entry_queue == {}
    assert (B in bot.portfolio.positions) == (disposition in ("held", "opened"))


def test_sol_usd_abort_keeps_queued_candidate_with_original_expiry(clock):
    bot, expiry = _queued_b(clock)
    bot._usd_to_lamports = lambda usd: None
    bot._seek_entry()
    assert bot._entry_queue == {B: expiry}


# -- review fix D3: exponent clamp ---------------------------------------------------

def test_very_long_failure_sequence_never_overflows_and_stays_capped(clock):
    bot = entry_bot(RouteJup(), [])
    delays = []
    for _ in range(5000):
        bot._note_entry_failure(A, "AAAA", clock.t)   # must never raise OverflowError
        until, streak = bot._entry_backoff[A]
        delays.append(until - clock.t)
        clock.t = until
    assert streak == 5000
    assert max(delays) <= main_module.ENTRY_BACKOFF_MAX_S
    assert delays[-1] == main_module.ENTRY_BACKOFF_MAX_S
    assert delays[:6] == [30.0, 60.0, 120.0, 240.0, 480.0, 900.0]
    for i in range(300):                              # > 256 distinct mints
        bot._note_entry_failure(f"M{i:043d}", "M", clock.t)
    assert len(bot._entry_backoff) <= main_module.ENTRY_BACKOFF_MAX_MINTS


# -- v3: a queued candidate that raises moves to the back, the search goes on ----

def raising(mints_that_raise, exc=None, calls=None):
    exc = exc if exc is not None else RuntimeError("boom")

    def fn(mint, **_k):
        if calls is not None:
            calls.append(mint)
        if mint in mints_that_raise:
            raise exc
        return None
    return fn


def fresh_feed(bot, prefix="F"):
    """A new, never-seen feed mint every cycle (boosted-like churn)."""
    n = {"i": 0}

    def find():
        n["i"] += 1
        return [types.SimpleNamespace(mint=f"{prefix}{n['i']:043d}")]
    bot.strategy = types.SimpleNamespace(find_candidates=find)


@pytest.mark.parametrize("stage", ["snapshot", "screen"])
def test_queued_poison_mint_never_blocks_later_candidates(clock, caplog, stage):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    expiry = clock.t + main_module.ENTRY_RETRY_QUEUE_TTL_S
    bot._entry_queue = {B: expiry}
    fresh_feed(bot)
    if stage == "snapshot":
        bot._candidate_market_snapshot = lambda m: raising({B})(m)
    else:
        real = bot.screener.screen
        bot.screener = types.SimpleNamespace(
            screen=lambda m, **k: raising({B})(m) if m == B else real(m, **k))
    with caplog.at_level(logging.INFO, logger="bot"):
        per_cycle = cycles(bot, clock, 20, step=60.0)  # 20 minutes, inside the TTL
    assert per_cycle == [[f"F{i:043d}"] for i in range(1, 21)]   # later mint, same cycle
    assert bot._entry_queue[B] == expiry                         # still queued, same expiry
    assert len(msgs(caplog, f"{B[:8]} {stage} failed (RuntimeError)")) == 20
    assert B not in bot.route_checked and B not in bot._entry_backoff   # no backoff escalation


def test_screen_exception_is_never_a_pass(clock):
    journal, swaps = [], []
    jup = RouteJup()
    bot = entry_bot(jup, [])
    bot._entry_queue = {B: clock.t + 600.0}
    real = bot.screener.screen
    bot.screener = types.SimpleNamespace(
        screen=lambda m, **k: raising({B})(m) if m == B else real(m, **k))
    bot.decision_shadow = types.SimpleNamespace(
        evaluate_candidate=lambda safety, snapshot=None: journal.append(safety.mint) or "eng",
        allows_paper_entry=lambda eng: True)
    bot.executor = types.SimpleNamespace(swap=lambda *a, **k: swaps.append(a) or pytest.fail("no swap"))
    bot._seek_entry()
    assert jup.calls == [] and bot.route_checked == [] and swaps == [] and journal == []
    assert bot.portfolio.positions == {}
    bot.executor = main_module.SwapExecutor(bot.cfg, jup=jup, session=object())
    bot.set_feed([C])                                 # the slot was not consumed by B
    bot._seek_entry()
    assert bot.route_checked == [C] and C in bot.portfolio.positions and journal == [C]


def test_failing_queued_mint_moves_behind_others_with_original_expiries(clock):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    eb, ec = clock.t + 900.0, clock.t + 1200.0
    bot._entry_queue = {B: eb, C: ec}
    bot._entry_backoff = {C: (clock.t + 300.0, 1)}    # C retained in place (backed off)
    bot._candidate_market_snapshot = lambda m: raising({B})(m)
    bot._seek_entry()
    assert list(bot._entry_queue.items()) == [(C, ec), (B, eb)]


def test_full_queue_of_failing_mints_stays_bounded_without_duplicates(clock):
    mints = [f"Q{i:043d}" for i in range(main_module.ENTRY_RETRY_QUEUE_MAX)]
    bot = entry_bot(RouteJup(**FAIL_SELL), mints + mints)   # feed repeats every queued mint
    bot._entry_queue = {m: clock.t + 1000.0 + i for i, m in enumerate(mints)}
    before = dict(bot._entry_queue)
    calls = []
    bot._candidate_market_snapshot = lambda m: raising(set(mints), calls=calls)(m)
    for n in range(1, 6):
        bot._seek_entry()
        assert len(calls) == n * len(mints)           # each mint once per cycle, no re-loop
        assert len(bot._entry_queue) == main_module.ENTRY_RETRY_QUEUE_MAX
        assert len(set(bot._entry_queue)) == len(bot._entry_queue)
        assert bot._entry_queue == before             # same mints, same expiries
        clock.t += POLL_S
    # a new backed-off candidate cannot squeeze in past the bound
    bot.set_feed([A])
    bot._entry_backoff = {A: (clock.t + 60.0, 1)}
    bot._seek_entry()
    assert A not in bot._entry_queue and len(bot._entry_queue) == main_module.ENTRY_RETRY_QUEUE_MAX


def test_mint_failing_until_expiry_is_removed_with_explicit_log(clock, caplog):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    bot._entry_queue = {B: clock.t + 100.0}
    bot._candidate_market_snapshot = lambda m: raising({B})(m)
    with caplog.at_level(logging.INFO, logger="bot"):
        cycles(bot, clock, 10)
    assert bot._entry_queue == {}
    assert len(msgs(caplog, f"entry retry queue: {B[:8]} expired")) == 1
    assert len(msgs(caplog, f"{B[:8]} snapshot failed")) == 7       # t=0..90 s, then expired


def test_mint_expiring_during_its_failure_is_removed_with_explicit_log(clock, caplog):
    bot = entry_bot(RouteJup(**FAIL_SELL), [])
    bot._entry_queue = {B: clock.t + 5.0}

    def snap(m):
        clock.t += 10.0                               # the failing fetch outlives the TTL
        raise TimeoutError("slow")
    bot._candidate_market_snapshot = snap
    with caplog.at_level(logging.INFO, logger="bot"):
        bot._seek_entry()
    assert bot._entry_queue == {}
    assert msgs(caplog, f"{B[:8]} snapshot failed (TimeoutError)")
    assert msgs(caplog, f"entry retry queue: {B[:8]} expired")


@pytest.mark.parametrize("stage", ["snapshot", "screen"])
def test_paper_feed_candidate_exception_still_propagates(clock, stage):
    bot = entry_bot(RouteJup(**FAIL_SELL), [B, C])
    bot._entry_queue = {A: clock.t + 600.0}           # an unrelated queued mint
    if stage == "snapshot":
        bot._candidate_market_snapshot = lambda m: raising({B})(m)
    else:
        real = bot.screener.screen
        bot.screener = types.SimpleNamespace(
            screen=lambda m, **k: raising({B})(m) if m == B else real(m, **k))
    with pytest.raises(RuntimeError, match="boom"):
        bot._seek_entry()
    assert B not in bot._entry_queue and C not in bot.screened


@pytest.mark.parametrize("stage", ["snapshot", "screen"])
def test_live_exception_still_propagates_and_never_touches_queue(clock, stage):
    bot = entry_bot(RouteJup(**FAIL_SELL), [B, C], cfg=armed_cfg())
    bot._live_entry = lambda *a: pytest.fail("no live entry")
    if stage == "snapshot":
        bot._candidate_market_snapshot = lambda m: raising({B})(m)
    else:
        bot.screener = types.SimpleNamespace(screen=lambda m, **k: raising({B})(m))
    with pytest.raises(RuntimeError, match="boom"):
        bot._seek_entry()
    assert getattr(bot, "_entry_queue", None) is None
    bot._entry_queue = {B: clock.t + 600.0}           # stale paper state: still not used
    with pytest.raises(RuntimeError, match="boom"):
        bot._seek_entry()
    assert bot._entry_queue == {B: clock.t + 600.0}


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
                         ids=["KeyboardInterrupt", "SystemExit", "CancelledError"])
@pytest.mark.parametrize("stage", ["snapshot", "screen"])
def test_base_exceptions_from_queued_candidate_still_propagate(clock, exc, stage):
    bot, expiry = _queued_b(clock)
    if stage == "snapshot":
        bot._candidate_market_snapshot = lambda m: raising({B}, exc())(m)
    else:
        bot.screener = types.SimpleNamespace(screen=lambda m, **k: raising({B}, exc())(m))
    with pytest.raises(exc):
        bot._seek_entry()
    assert bot._entry_queue == {B: expiry}            # peeked, not lost


# -- review hardening: a stale 429 is never inherited by a later failure ------

def test_stale_rate_limit_is_not_inherited_by_non_429_route_failure(clock, caplog):
    # A 429 left on the client by an earlier, unrelated request; the fake's
    # failing sell leg never touches last_error_kind (a non-429 rejection).
    jup = RouteJup(**FAIL_SELL)
    bot = entry_bot(jup, [A])
    for streak, delay in ((1, 30.0), (2, 60.0)):
        jup.last_error_kind = QUOTE_RATE_LIMITED
        bot._seek_entry()
        assert bot._entry_backoff[A] == (clock.t + delay, streak)
        clock.t += delay
    assert bot.route_checked == [A, A]


def test_genuine_429_during_route_check_stays_flat_and_non_escalating(clock):
    jup = RouteJup()
    jup.sell = lambda q, a: setattr(jup, "last_error_kind", QUOTE_RATE_LIMITED)
    bot = entry_bot(jup, [A])
    for _ in range(3):
        bot._seek_entry()
        assert bot._entry_backoff[A] == (clock.t + main_module.ENTRY_BACKOFF_BASE_S, 0)
        clock.t += main_module.ENTRY_BACKOFF_BASE_S
    assert bot.route_checked == [A] * 3


def test_armed_live_does_not_reset_last_error_kind(clock):
    jup = RouteJup(**FAIL_SELL)
    jup.last_error_kind = QUOTE_RATE_LIMITED
    bot = entry_bot(jup, [A], cfg=armed_cfg())
    bot._live_entry = lambda *a: pytest.fail("route fails; no live entry")
    bot._seek_entry()
    assert bot.route_checked == [A]
    assert jup.last_error_kind == QUOTE_RATE_LIMITED   # untouched in live
    assert getattr(bot, "_entry_backoff", None) is None
