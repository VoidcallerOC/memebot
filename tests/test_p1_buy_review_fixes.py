"""P1 review fixes for the buy path (#40 review, carried by #41).

1. Corrupt persisted intents fail closed. ``buy_intents`` ABSENT (legacy) ->
   []. Present but not a list, or any malformed entry (validated against what
   ``TradingBot._live_entry`` writes) -> ``Portfolio.corrupt``: the raw field
   is preserved verbatim and written back unchanged by every save; armed
   startup refuses ("buy_intents corrupt: state recovery required"), an armed
   tick halts, and no entry is searched / quoted / signed / sent (paper too:
   exits still simulated, entries refused). The same type-level check now
   covers ``pending_intents`` (sell intents): a present non-list value or a
   non-object entry used to be silently ignored by the sell freeze.
2. SOL-input fills: ``actual_in`` only from ``_native_sol_spent`` (exact,
   balance-reconciled). The old owner-lamport-delta fallback could UNDERSTATE
   the spend when SOL also arrives in the same transaction; such a buy now
   has an incomplete fill and stays unresolved (no position, entries blocked).

Fixture ``synthetic_native_sol_buy_incoming_sol.json`` is SYNTHETIC: derived
deterministically from the pinned synthetic buy by adding one incoming system
transfer of 5,000,000 lamports to the owner (sha256 pinned below). The base
synthetic fixture is unchanged and still pinned.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest

from bot.chain import _native_delta, _native_sol_spent, extract_fill
from bot.jupiter import SOL_MINT
from bot.portfolio import BUY_INTENT_STATUSES
import bot.main as main_module
from tests.test_p1_buy_prebroadcast_signature import (  # noqa: F401  (bw is a fixture)
    FEE, FIXTURE, FIXTURE_SHA256, MINT, OWNER_PLACEHOLDER, PLANNED, SOL_IN, boot, bw,
    buys_of, persisted, recorded_once, run, start,
)

INCOMING = pathlib.Path(__file__).parent / "fixtures" / "synthetic_native_sol_buy_incoming_sol.json"
INCOMING_SHA256 = "65ce3e13cc3ce25a9887e274c6433551814d63ae2a07243705e7c9d6c73ec9de"
INCOMING_LAMPORTS = 5_000_000


def good_intent(**over):
    intent = {"side": "buy", "mint": MINT, "symbol": "SYN", "lamports": PLANNED,
              "allocation_usd": 2.0, "screened_decimals": 6, "status": "failed",
              "tx_signature": None, "created_at": 1_800_000_000.0}
    intent.update(over)
    return intent


def missing(key):
    i = good_intent()
    del i[key]
    return i


CORRUPT = [
    pytest.param({"a": 1}, id="field-dict"),
    pytest.param("oops", id="field-string"),
    pytest.param(0, id="field-zero"),
    pytest.param(None, id="field-null"),
    pytest.param([good_intent(), "x"], id="entry-not-object"),
    pytest.param([missing("side")], id="missing-side"),
    pytest.param([good_intent(side="sell")], id="wrong-side"),
    pytest.param([missing("mint")], id="missing-mint"),
    pytest.param([good_intent(mint="")], id="empty-mint"),
    pytest.param([good_intent(mint=7)], id="non-string-mint"),
    pytest.param([good_intent(status="weird")], id="unknown-status"),
    pytest.param([missing("status")], id="missing-status"),
    pytest.param([missing("lamports")], id="missing-lamports"),
    pytest.param([good_intent(lamports=True)], id="bool-lamports"),
    pytest.param([good_intent(lamports=0)], id="zero-lamports"),
    pytest.param([good_intent(lamports="13333333")], id="string-lamports"),
    pytest.param([good_intent(lamports=2**64)], id="u64-overflow-lamports"),
    pytest.param([missing("created_at")], id="missing-created_at"),
    pytest.param([good_intent(created_at="yesterday")], id="string-created_at"),
    pytest.param([missing("allocation_usd")], id="missing-allocation"),
    pytest.param([missing("tx_signature")], id="missing-signature-field"),
    pytest.param([good_intent(tx_signature=12345)], id="non-string-signature"),
    pytest.param([good_intent(tx_signature="")], id="empty-signature"),
    pytest.param([good_intent(status="settled")], id="settled-without-signature"),
    pytest.param([good_intent(status="unresolved", tx_signature="S" * 88), good_intent(lamports=-5)],
                 id="second-entry-bad"),
]


def write_state(w, **portfolio):
    base = {"realized_pnl": 0.0, "positions": [], "pending_intents": []}
    base.update(portfolio)
    with open(w.state, "w") as f:
        json.dump({"version": 1, "portfolio": base, "risk": {}}, f)


def raw_field(w, name):
    return persisted(w)["portfolio"][name]


def assert_no_trading(w, bot):
    assert bot.jup.swap_quotes() == [] and w.builds == 0 and w.sends == []
    assert bot.portfolio.positions == {}


# -- 1. corrupt buy intents fail closed, raw data preserved --------------------------

def test_lifecycle_statuses_match_the_validator():
    assert main_module.OPEN_BUY_INTENT_STATUSES | {"failed", "settled"} == BUY_INTENT_STATUSES


def test_intents_written_by_the_lifecycle_validate_on_reload(bw):
    bot = start(bw)
    bw.plan = ["drop"]
    bot._tick()                                       # pending -> unresolved with signature
    assert boot(bw).portfolio.corrupt == {}
    bw.now += main_module.BUY_INTENT_EXPIRY_SECONDS + 1
    bot._tick()                                       # failed, then a new settled intent
    reloaded = boot(bw)
    assert reloaded.portfolio.corrupt == {} and len(reloaded.portfolio.buy_intents) == 2


def test_legacy_state_without_buy_intents_key_loads_and_trades(bw):
    write_state(bw)                                   # no "buy_intents" key at all
    bot = boot(bw)
    assert bot.portfolio.corrupt == {} and bot.portfolio.buy_intents == []
    assert run(bot, 1) == 1
    recorded_once(bw, bw.sends[0])


@pytest.mark.parametrize("raw", CORRUPT)
def test_corrupt_buy_intents_refuse_armed_trading_and_survive_saves(bw, raw):
    write_state(bw, buy_intents=copy.deepcopy(raw))
    before = json.dumps(raw_field(bw, "buy_intents"), sort_keys=True)
    bot = boot(bw)
    assert "buy_intents" in bot.portfolio.corrupt
    assert "buy_intents corrupt" in bot._corrupt_state_reason()
    assert "state recovery required" in bot._corrupt_state_reason()
    assert run(bot, 3) == 0                           # armed startup refused
    assert_no_trading(bw, bot)
    bot._running = True
    bot._tick()                                       # a direct armed tick halts
    assert bot._running is False
    bot._seek_entry()                                 # and the entry search refuses
    assert bot._live_entry(MINT, None, 2.0, PLANNED) == "stop"
    assert_no_trading(bw, bot)
    assert bot._save() is True                        # never overwritten with []
    assert raw_field(bw, "buy_intents") == raw
    assert json.dumps(raw_field(bw, "buy_intents"), sort_keys=True) == before
    again = boot(bw)                                  # reload: still corrupt, still preserved
    assert "buy_intents" in again.portfolio.corrupt and run(again, 2) == 0
    assert raw_field(bw, "buy_intents") == raw
    assert_no_trading(bw, again)


@pytest.mark.parametrize("raw", [CORRUPT[0], CORRUPT[4], CORRUPT[10]])
def test_corrupt_buy_intents_paper_session_runs_exits_but_opens_no_entry(bw, raw):
    write_state(bw, buy_intents=copy.deepcopy(raw))
    paper = boot(bw, armed=False)
    assert run(paper, 2) == 2                         # paper keeps running (exits simulated)
    assert paper.entry_scans == [] and paper.jup.quotes == [] and paper.portfolio.positions == {}
    assert raw_field(bw, "buy_intents") == raw        # the final save preserved it


@pytest.mark.parametrize("raw", [pytest.param({"x": 1}, id="field-dict"),
                                 pytest.param("x", id="field-string"),
                                 pytest.param([{"mint": MINT, "status": "pending"}, 3], id="entry-not-object")])
def test_corrupt_sell_intents_also_fail_closed_and_are_preserved(bw, raw):
    write_state(bw, pending_intents=copy.deepcopy(raw))
    bot = boot(bw)
    assert "pending_intents" in bot.portfolio.corrupt
    assert run(bot, 2) == 0
    assert_no_trading(bw, bot)
    assert bot._save() is True and raw_field(bw, "pending_intents") == raw


def test_null_or_empty_sell_intents_stay_legacy_ok(bw):
    for value in (None, []):
        write_state(bw, pending_intents=value)
        assert boot(bw).portfolio.corrupt == {}


# -- 2. SOL-input fill: exact principal only ------------------------------------------

def test_incoming_sol_fixture_pinned_synthetic_and_base_fixture_unchanged():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    data = INCOMING.read_bytes()
    assert hashlib.sha256(data).hexdigest() == INCOMING_SHA256
    src = json.loads(data)["_source"]
    assert src["kind"] == "SYNTHETIC" and "NOT a real" in src["note"]


def test_incoming_sol_old_fallback_understates_new_fill_is_incomplete():
    tx = json.loads(INCOMING.read_text())["result"]
    old = _native_delta(tx, OWNER_PLACEHOLDER, FEE)   # the removed fallback's value
    assert old == SOL_IN + 2_039_280 - INCOMING_LAMPORTS == 10_372_613
    assert old < SOL_IN                               # understates the true principal
    assert _native_sol_spent(tx, OWNER_PLACEHOLDER) is None
    fill = extract_fill(tx, OWNER_PLACEHOLDER, SOL_MINT, MINT)
    assert fill.actual_in_amount is None and not fill.complete


def test_sol_input_without_parsed_instructions_has_no_actual_in():
    tx = json.loads(FIXTURE.read_text())["result"]
    del tx["meta"]["innerInstructions"]               # principal no longer provable
    fill = extract_fill(tx, OWNER_PLACEHOLDER, SOL_MINT, MINT)
    assert fill.actual_in_amount is None and not fill.complete


def test_buy_with_incoming_sol_stays_unresolved_no_position_entries_blocked(bw):
    bw.fill_tx = json.loads(INCOMING.read_text().replace(OWNER_PLACEHOLDER, bw.owner))["result"]
    bot = start(bw)
    bot._tick()
    (sig,) = bw.sends
    (intent,) = bot.portfolio.buy_intents
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert "SOL in None" in intent["resolution"]
    assert bot.portfolio.positions == {}
    scans = len(bot.entry_scans)
    for _ in range(3):
        bot._tick()                                   # resolver: confirmed, fill does not verify
    assert bw.sends == [sig] and len(bot.entry_scans) == scans
    (intent,) = buys_of(persisted(bw))
    assert intent["status"] == "unresolved" and "does not verify" in intent["resolution"]
    assert persisted(bw)["portfolio"]["positions"] == []
