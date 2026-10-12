"""Re-review P1s on #41 (Nick @ d9f07b4).

1. ``_native_sol_spent`` / ``_native_sol_proceeds`` counted NET wSOL flow on
   the owner's wSOL accounts, so an unrelated SPL transfer into an owner wSOL
   account still reconciled the grouped lamport equation: buy principal
   understated, sell proceeds overstated. Now a wSOL transfer touching the
   owner's wSOL accounts counts only as a CPI of the single top-level Jupiter
   route instruction; anything else -> None -> incomplete fill -> unresolved.
2. ``pending_intents: null`` was loaded as []. A present non-list (null
   included) is corrupt; only an absent key is legacy [].
3. Sell-intent entries were not validated. Each entry must now have a
   non-empty mint and a known string status (plus, for confirmed_unbooked,
   its signature and booking inputs); every historical writer shape loads.

Fixtures: the two ``*_unrelated_wsol.json`` files are SYNTHETIC derivations
(one extra top-level wSOL transferChecked from a third-party account into the
owner's temporary wSOL account) of the base synthetic buy and the real
fWueMsPH sell; both bases are hash-pinned and unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest

import bot.main as main_module
from bot.chain import JUPITER_V6_PROGRAM, _native_sol_proceeds, _native_sol_spent, extract_fill
from bot.jupiter import SOL_MINT
from bot.portfolio import SELL_INTENT_STATUSES, sell_intent_problem
import tests.test_p1_buy_prebroadcast_signature as B
import tests.test_p1_prebroadcast_signature as S
from tests.test_p1_buy_prebroadcast_signature import bw  # noqa: F401  (fixture)
from tests.test_p1_prebroadcast_signature import rw  # noqa: F401  (fixture)
from tests.test_p1_native_sol_settlement_e2e import (
    IN_RAW, MINT as SELL_MINT, OUT_LAMPORTS, OWNER as SELL_OWNER, intents_of, position,
)

FIX = pathlib.Path(__file__).parent / "fixtures"
BUY_BASE, SELL_BASE = FIX / "synthetic_native_sol_buy.json", FIX / "native_sol_sell_fWueMsPH.json"
BUY_WSOL = FIX / "synthetic_native_sol_buy_unrelated_wsol.json"
SELL_WSOL = FIX / "synthetic_native_sol_sell_unrelated_wsol.json"
BUY_ROUTE = FIX / "synthetic_native_sol_buy_route_inner_wsol.json"
SELL_ROUTE = FIX / "synthetic_native_sol_sell_route_inner_wsol.json"
PINS = {
    SELL_BASE: "730382ed56c73866e74cc80bbadfa75732c9ab943d3f4d8d971d61ec14be75b1",
    BUY_BASE: B.FIXTURE_SHA256,
    BUY_WSOL: "254e6e63578be4e78dff157309e046d4ad7c94ac55f8b9e8d2e4fb75676825ad",
    SELL_WSOL: "b1e7e556c408de2e4b7f02bec947166b89969925d252eb5da2d671e2afc33552",
    BUY_ROUTE: "b14f2fd81c19decf9bf403de5275d4f824cebfd2b6572bcc41bdb0158b4dd06b",
    SELL_ROUTE: "ce877bf58cd42a75694c4061bf0cce1fc89d4ea079a9253ef76b55ec0dd7f76a",
}
BUY_DONATION, SELL_DONATION = 4_000_000, 1_000_000


def tx_of(path):
    return json.loads(path.read_text())["result"]


def into_route(tx):
    """Move the injected top-level transfer inside the Jupiter route group
    (as if the route itself had made it). The grouped lamport equation still
    reconciles, so 7cc26cb (route-membership attribution) returned the skewed
    net amount; the committed *_route_inner_wsol.json fixtures are exactly this."""
    tops = tx["transaction"]["message"]["instructions"]
    at = next(i for i, ix in enumerate(tops)
              if (ix.get("parsed") or {}).get("info", {}).get("source", "").startswith("SynthDonor"))
    ix = tops.pop(at)
    route = next(i for i, t in enumerate(tops) if t.get("programId") == JUPITER_V6_PROGRAM)
    next(g for g in tx["meta"]["innerInstructions"] if g["index"] == route)["instructions"].append(ix)
    return tx


# -- 1. unrelated wSOL flows -----------------------------------------------------------

def test_fixtures_pinned_bases_unchanged_derivations_synthetic():
    for path, sha in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, path.name
    for path, base in ((BUY_WSOL, BUY_BASE), (SELL_WSOL, SELL_BASE),
                       (BUY_ROUTE, BUY_WSOL), (SELL_ROUTE, SELL_WSOL)):
        src = json.loads(path.read_text())["_source"]
        assert src["kind"] == "SYNTHETIC" and "NOT a real" in src["note"]
        assert src["derived_from"] == base.name and src["derived_from_sha256"] == PINS[base]


def test_base_buy_and_real_sell_still_parse_exactly():
    assert _native_sol_spent(tx_of(BUY_BASE), B.OWNER_PLACEHOLDER) == B.SOL_IN
    buy = extract_fill(tx_of(BUY_BASE), B.OWNER_PLACEHOLDER, SOL_MINT, B.MINT)
    assert buy.complete and buy.actual_in_amount == B.SOL_IN
    assert _native_sol_proceeds(tx_of(SELL_BASE), SELL_OWNER) == OUT_LAMPORTS
    sell = extract_fill(tx_of(SELL_BASE), SELL_OWNER, SELL_MINT, SOL_MINT)
    assert sell.complete and (sell.actual_in_amount, sell.actual_out_amount) == (IN_RAW, OUT_LAMPORTS)


def test_buy_unrelated_wsol_old_value_understated_new_is_none():
    # d9f07b4 (net flow) returned 13,333,333 - 4,000,000 = 9,333,333 here.
    tx = tx_of(BUY_WSOL)
    assert _native_sol_spent(tx, B.OWNER_PLACEHOLDER) is None
    fill = extract_fill(tx, B.OWNER_PLACEHOLDER, SOL_MINT, B.MINT)
    assert fill.actual_in_amount is None and not fill.complete


def test_sell_unrelated_wsol_old_value_overstated_new_is_none():
    # d9f07b4 (net flow) returned 8,656,067 + 1,000,000 = 9,656,067 here.
    tx = tx_of(SELL_WSOL)
    assert _native_sol_proceeds(tx, SELL_OWNER) is None
    fill = extract_fill(tx, SELL_OWNER, SELL_MINT, SOL_MINT)
    assert fill.actual_out_amount is None and not fill.complete


@pytest.mark.parametrize("path,owner,fn", [(BUY_BASE, B.OWNER_PLACEHOLDER, _native_sol_spent),
                                           (SELL_BASE, SELL_OWNER, _native_sol_proceeds)])
def test_no_or_two_route_instructions_fail_closed(path, owner, fn):
    tx = tx_of(path)
    tops = tx["transaction"]["message"]["instructions"]
    route = next(t for t in tops if t.get("programId") == JUPITER_V6_PROGRAM)
    route["programId"] = "NotJupiter111111111111111111111111111111111"
    assert fn(tx, owner) is None
    tx = tx_of(path)
    tx["transaction"]["message"]["instructions"].append(
        {"programId": JUPITER_V6_PROGRAM, "accounts": [], "data": ""})
    assert fn(tx, owner) is None


def test_buy_with_unrelated_wsol_unresolved_no_position_entries_blocked(bw):
    bw.fill_tx = json.loads(BUY_WSOL.read_text().replace(B.OWNER_PLACEHOLDER, bw.owner))["result"]
    bot = B.start(bw)
    bot._tick()
    (sig,) = bw.sends
    (intent,) = bot.portfolio.buy_intents
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert bot.portfolio.positions == {}
    scans = len(bot.entry_scans)
    for _ in range(3):
        bot._tick()
    assert bw.sends == [sig] and len(bot.entry_scans) == scans       # entries blocked, no resend
    (intent,) = B.buys_of(B.persisted(bw))
    assert intent["status"] == "unresolved"
    assert B.persisted(bw)["portfolio"]["positions"] == []


def test_sell_with_unrelated_wsol_unresolved_no_booking_no_settle_no_resend(rw):
    rw.fill_tx = json.loads(SELL_WSOL.read_text().replace(SELL_OWNER, rw.owner))["result"]
    bot = S.start(rw)
    bot._tick()
    (sig,) = rw.sends
    for _ in range(3):
        bot._tick()
    rw.now += S.EXPIRY + 1
    bot2 = S.boot(rw)
    S.run(bot2, 3)
    state = S.persisted(rw)
    (intent,) = intents_of(state)
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert intent.get("booked_at") is None and intent.get("actual_out_amount") is None
    assert rw.sends == [sig] and bot.jup.quotes == [(SELL_MINT, SOL_MINT, IN_RAW)] and rw.builds == 1
    assert bot2.jup.quotes == [] and bot2.entry_scans == []
    assert state["portfolio"]["realized_pnl"] == 0.0 and position(state)["raw_tokens"] == IN_RAW


# -- 2. pending_intents: null is corrupt -----------------------------------------------

def _corrupt_sells_in_sell_world(rw, raw):
    bot = S.start(rw)                                     # open position at a stop-loss mark
    state = S.persisted(rw)
    state["portfolio"]["pending_intents"] = copy.deepcopy(raw)
    with open(rw.state, "w") as f:
        json.dump(state, f)
    bot = S.boot(rw)
    assert "pending_intents" in bot.portfolio.corrupt
    assert S.run(bot, 3) == 0                             # armed startup refused
    bot._running = True
    bot._tick()
    assert bot._running is False                          # a direct armed tick halts
    assert bot.jup.quotes == [] and rw.builds == 0 and rw.sends == []
    assert bot._save() is True and S.persisted(rw)["portfolio"]["pending_intents"] == raw
    again = S.boot(rw)
    assert "pending_intents" in again.portfolio.corrupt and S.run(again, 2) == 0
    assert again.jup.quotes == [] and rw.builds == 0 and rw.sends == []
    assert S.persisted(rw)["portfolio"]["pending_intents"] == raw


def test_null_pending_intents_corrupt_armed_refuses_raw_preserved(rw):
    _corrupt_sells_in_sell_world(rw, None)


def test_null_pending_intents_paper_opens_no_entry(bw):
    from tests.test_p1_buy_review_fixes import write_state, raw_field
    write_state(bw, pending_intents=None)
    paper = B.boot(bw, armed=False)
    assert "pending_intents" in paper.portfolio.corrupt
    assert B.run(paper, 2) == 2
    assert paper.entry_scans == [] and paper.jup.quotes == [] and paper.portfolio.positions == {}
    assert raw_field(bw, "pending_intents") is None


def test_absent_pending_intents_is_legacy_ok(bw):
    from tests.test_p1_buy_review_fixes import write_state
    write_state(bw)
    state = B.persisted(bw)
    del state["portfolio"]["pending_intents"]
    with open(bw.state, "w") as f:
        json.dump(state, f)
    bot = B.boot(bw)
    assert bot.portfolio.corrupt == {} and bot.portfolio.pending_intents == []
    assert B.run(bot, 1) == 1 and len(bw.sends) == 1


# -- 3. sell-intent entries validated ----------------------------------------------------

SIG = "5" * 88
C733 = {"mint": SELL_MINT, "fraction": 1.0, "reason": "stop_loss", "requested_price": 1.0,
        "status": "pending", "tx_signature": None}                          # c733c9e
R1975 = dict(C733, created_at=1_800_000_000.0, raw_amount=None)              # 1975efd
R4587 = dict(C733, created_at=1_800_000_000.0, raw_amount=IN_RAW, rung=0)    # 4587d65 / d9f07b4

LEGACY = [
    pytest.param(dict(C733), id="c733-pending"),
    pytest.param(dict(C733, status="blocked"), id="c733-blocked"),
    pytest.param(dict(C733, status="unresolved", tx_signature=SIG), id="c733-unresolved"),
    pytest.param(dict(C733, status="settled", tx_signature=SIG), id="c733-settled"),
    pytest.param(dict(R1975, status="failed", error="x", resolution="r", resolved_at=1.0), id="1975-failed"),
    pytest.param(dict(R1975, status="unresolved", tx_signature=SIG, first_seen_at=1.0, alerted=True,
                      outcome="unknown"), id="1975-unresolved"),
    pytest.param(dict(R1975, status="settled", tx_signature=SIG, raw_amount=IN_RAW, actual_in_amount=IN_RAW,
                      actual_out_amount=OUT_LAMPORTS, fee_lamports=5000), id="1975-settled"),
    pytest.param(dict(R4587, status="confirmed_unbooked", tx_signature=SIG, actual_in_amount=IN_RAW,
                      actual_out_amount=OUT_LAMPORTS, fee_lamports=5000, confirmed_at=1.0),
                 id="4587-confirmed_unbooked"),
    pytest.param(dict(R4587, status="settled", tx_signature=SIG, booked_at=1.0, booked_pnl=-0.1,
                      booked_raw=IN_RAW), id="4587-settled"),
    pytest.param(dict(R4587, status="pending", rung=None), id="4587-pending-no-rung"),
]


def _drop(entry, key):
    e = dict(entry)
    del e[key]
    return e


CONFIRMED = dict(R4587, status="confirmed_unbooked", tx_signature=SIG)
BAD = [
    pytest.param(_drop(C733, "status"), id="missing-status"),
    pytest.param(dict(C733, status="weird"), id="unknown-status"),
    pytest.param(dict(C733, status=1), id="non-string-status"),
    pytest.param(dict(C733, status=None), id="null-status"),
    pytest.param(dict(C733, status="confirmed_unrecorded"), id="buy-only-status"),
    pytest.param(_drop(C733, "mint"), id="missing-mint"),
    pytest.param(dict(C733, mint=""), id="empty-mint"),
    pytest.param(dict(C733, mint=5), id="non-string-mint"),
    pytest.param(dict(C733, tx_signature=7), id="non-string-signature"),
    pytest.param(dict(C733, tx_signature=""), id="empty-signature"),
    pytest.param(dict(CONFIRMED, tx_signature=None), id="confirmed-without-signature"),
    pytest.param(_drop(CONFIRMED, "fraction"), id="confirmed-without-fraction"),
    pytest.param(_drop(CONFIRMED, "requested_price"), id="confirmed-without-price"),
    pytest.param(_drop(CONFIRMED, "reason"), id="confirmed-without-reason"),
    pytest.param(dict(CONFIRMED, booked_at=1.0), id="confirmed-already-booked"),
]


def test_status_set_covers_the_lifecycle():
    assert main_module.OPEN_SELL_INTENT_STATUSES < SELL_INTENT_STATUSES
    assert SELL_INTENT_STATUSES - main_module.OPEN_SELL_INTENT_STATUSES == {"failed", "settled"}


@pytest.mark.parametrize("entry", LEGACY)
def test_every_historical_shape_loads(bw, entry):
    from tests.test_p1_buy_review_fixes import write_state
    assert sell_intent_problem(entry) is None
    write_state(bw, pending_intents=[entry])
    bot = B.boot(bw)
    assert bot.portfolio.corrupt == {} and bot.portfolio.pending_intents == [entry]


def test_intents_written_by_the_sell_lifecycle_validate_on_reload(rw):
    bot = S.start(rw)
    rw.plan = ["land_hidden"]
    bot._tick()                                           # pending -> unresolved
    assert S.boot(rw).portfolio.corrupt == {}
    rw.show(rw.sends[0])
    for _ in range(3):
        bot._tick()                                       # confirmed_unbooked -> settled
    reloaded = S.boot(rw)
    assert reloaded.portfolio.corrupt == {}
    assert [i["status"] for i in reloaded.portfolio.pending_intents] == ["settled"]


@pytest.mark.parametrize("entry", BAD)
def test_unclassifiable_sell_intent_corrupt_no_quote_build_send(rw, entry):
    assert sell_intent_problem(entry) is not None
    _corrupt_sells_in_sell_world(rw, [entry])


@pytest.mark.parametrize("entry", BAD[:6])
def test_unclassifiable_sell_intent_after_a_valid_one_still_corrupt(rw, entry):
    _corrupt_sells_in_sell_world(rw, [dict(C733, status="failed"), entry])


@pytest.mark.parametrize("entry", BAD[:6])
def test_unclassifiable_sell_intent_paper_opens_no_entry(bw, entry):
    from tests.test_p1_buy_review_fixes import write_state, raw_field
    write_state(bw, pending_intents=[entry])
    paper = B.boot(bw, armed=False)
    assert "pending_intents" in paper.portfolio.corrupt
    assert B.run(paper, 2) == 2
    assert paper.entry_scans == [] and paper.jup.quotes == [] and paper.portfolio.positions == {}
    assert raw_field(bw, "pending_intents") == [entry]


# -- Route attribution fix (Nick @ 7cc26cb): membership is not proof ------------------

def route_group(tx):
    tops = tx["transaction"]["message"]["instructions"]
    idx = next(i for i, t in enumerate(tops) if t.get("programId") == JUPITER_V6_PROGRAM)
    return next(g for g in tx["meta"]["innerInstructions"] if g["index"] == idx)["instructions"]


def owner_wsol_transfer(tx):
    return next(ix for ix in route_group(tx) if (ix.get("parsed") or {}).get("info", {}).get("mint") == SOL_MINT)


def split(tx, first):
    """Split the single owner wSOL transfer into two same-direction route
    CPIs with the same total: balances and the equation are unchanged."""
    ix = owner_wsol_transfer(tx)
    amt = int(ix["parsed"]["info"]["tokenAmount"]["amount"])
    two = copy.deepcopy(ix)
    ix["parsed"]["info"]["tokenAmount"]["amount"] = str(first)
    two["parsed"]["info"]["tokenAmount"]["amount"] = str(amt - first)
    group = route_group(tx)
    group.insert(group.index(ix) + 1, two)
    return tx


def test_route_inner_fixtures_equal_into_route_derivation():
    assert tx_of(BUY_ROUTE) == into_route(tx_of(BUY_WSOL))
    assert tx_of(SELL_ROUTE) == into_route(tx_of(SELL_WSOL))


def test_route_inner_unrelated_wsol_buy_is_none_incomplete():
    tx = tx_of(BUY_ROUTE)              # 7cc26cb returned 9,333,333 (true 13,333,333)
    assert _native_sol_spent(tx, B.OWNER_PLACEHOLDER) is None
    fill = extract_fill(tx, B.OWNER_PLACEHOLDER, SOL_MINT, B.MINT)
    assert fill.actual_in_amount is None and not fill.complete


def test_route_inner_unrelated_wsol_sell_is_none_incomplete():
    tx = tx_of(SELL_ROUTE)             # 7cc26cb returned 9,656,067 (true 8,656,067)
    assert _native_sol_proceeds(tx, SELL_OWNER) is None
    fill = extract_fill(tx, SELL_OWNER, SELL_MINT, SOL_MINT)
    assert fill.actual_out_amount is None and not fill.complete


def test_two_same_direction_owner_wsol_transfers_fail_closed():
    buy = split(tx_of(BUY_BASE), 6_666_666)           # two owner -> pool outflows
    assert _native_sol_spent(buy, B.OWNER_PLACEHOLDER) is None
    sell = split(tx_of(SELL_BASE), 4_000_000)          # two pool -> owner inflows
    assert _native_sol_proceeds(sell, SELL_OWNER) is None


def test_owner_wsol_to_owner_wsol_transfer_fails_closed():
    for path, owner, fn in ((BUY_BASE, B.OWNER_PLACEHOLDER, _native_sol_spent),
                            (SELL_BASE, SELL_OWNER, _native_sol_proceeds)):
        tx = tx_of(path)
        ix = copy.deepcopy(owner_wsol_transfer(tx))
        info = ix["parsed"]["info"]
        own = info["source"] if fn is _native_sol_spent else info["destination"]
        info["source"] = info["destination"] = own
        route_group(tx).append(ix)
        assert fn(tx, owner) is None


def test_buy_route_inner_unrelated_wsol_full_flow_unresolved(bw):
    bw.fill_tx = json.loads(BUY_ROUTE.read_text().replace(B.OWNER_PLACEHOLDER, bw.owner))["result"]
    bot = B.start(bw)
    bot._tick()
    (sig,) = bw.sends
    scans = len(bot.entry_scans)
    for _ in range(3):
        bot._tick()
    assert bw.sends == [sig] and len(bot.entry_scans) == scans
    (intent,) = B.buys_of(B.persisted(bw))
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert bot.portfolio.positions == {} and B.persisted(bw)["portfolio"]["positions"] == []


def test_sell_route_inner_unrelated_wsol_full_flow_unresolved(rw):
    rw.fill_tx = json.loads(SELL_ROUTE.read_text().replace(SELL_OWNER, rw.owner))["result"]
    bot = S.start(rw)
    bot._tick()
    (sig,) = rw.sends
    for _ in range(3):
        bot._tick()
    rw.now += S.EXPIRY + 1
    bot2 = S.boot(rw)
    S.run(bot2, 3)
    state = S.persisted(rw)
    (intent,) = intents_of(state)
    assert intent["status"] == "unresolved" and intent.get("booked_at") is None
    assert rw.sends == [sig] and rw.builds == 1 and bot2.jup.quotes == []
    assert state["portfolio"]["realized_pnl"] == 0.0 and position(state)["raw_tokens"] == IN_RAW
