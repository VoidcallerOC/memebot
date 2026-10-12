"""Buy output attribution fix (Nick @ b8a2a95).

``extract_fill`` took the bought token amount from the owner's aggregate
pre/post token-balance delta, so an unrelated incoming transfer of the same
mint inflated ``raw_tokens``/``raw_tokens_initial``. Now
``_route_token_received`` must prove it: exactly one transfer/transferChecked
(SPL Token or Token-2022) of the mint into an owner token account, as a CPI
of the single top-level Jupiter route, from a non-owner account; no other
token instruction touching the owner's accounts of that mint; aggregate delta
EQUAL to the transfer. Otherwise actual_out is None (incomplete fill, unresolved).

Fixtures: SYNTHETIC, hash-pinned, derived from the pinned base synthetic buy
(fixture (a) is the base itself). b8a2a95 values: (b)/(c) 4,201,000,000,
(d) 4,200,000,000 (accepted though split), (e) 4,200,000,001 / 4,199,999,999.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest

import bot.main as main_module
from bot.chain import _route_token_received, extract_fill
from bot.jupiter import SOL_MINT
import tests.test_p1_buy_prebroadcast_signature as B
from tests.test_p1_buy_prebroadcast_signature import bw  # noqa: F401  (fixture)

FIX = pathlib.Path(__file__).parent / "fixtures"
OUT = 4_200_000_000
V = {name: FIX / f"synthetic_native_sol_buy_output_{name}.json" for name in (
    "b_unrelated_top_level", "c_unrelated_route_inner", "d_two_route_transfers",
    "e_delta_plus_one", "e_delta_minus_one")}
PINS = {
    B.FIXTURE: B.FIXTURE_SHA256,
    V["b_unrelated_top_level"]: "15ac2d15a9d4db90c88ea2f30271400a1cab60cb065b431e26e80575a925b9e8",
    V["c_unrelated_route_inner"]: "cf6e6be8e24c85fd96d32a579d03b642bc6b6bab317c6d47df57630c5093719f",
    V["d_two_route_transfers"]: "48f84a47beac6b76c297ba6efa6a3da47e99b8cf5c860b9b2116ca81d764dd8b",
    V["e_delta_plus_one"]: "bc15de1a526c4ef2b0b6c0253fe6af8532f8df9c5e446beec79bf89a0cca6ab5",
    V["e_delta_minus_one"]: "cd7ab08b98b2fe850aed85f77d3122b589e341f5d222bd9478ccb17f2a3c9cd6",
}
OWNER = B.OWNER_PLACEHOLDER
ATA = "SynthTokenAta111111111111111111111111111111"


def tx_of(path):
    return json.loads(path.read_text())["result"]


def fill(tx):
    return extract_fill(tx, OWNER, SOL_MINT, B.MINT)


def route_group(tx):
    tops = tx["transaction"]["message"]["instructions"]
    idx = next(i for i, t in enumerate(tops) if t.get("programId") == "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")
    return next(g for g in tx["meta"]["innerInstructions"] if g["index"] == idx)["instructions"]


def out_transfer(tx):
    return next(ix for ix in route_group(tx) if ix["parsed"]["info"].get("mint") == B.MINT)


def test_fixtures_pinned_and_synthetic():
    for path, sha in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, path.name
    for path in V.values():
        src = json.loads(path.read_text())["_source"]
        assert src["kind"] == "SYNTHETIC" and "NOT a real" in src["note"]
        assert src["derived_from"] == B.FIXTURE.name and src["derived_from_sha256"] == B.FIXTURE_SHA256


def test_a_valid_buy_still_exact():
    f = fill(tx_of(B.FIXTURE))
    assert f.complete and (f.actual_in_amount, f.actual_out_amount, f.out_decimals) == (B.SOL_IN, OUT, 6)


@pytest.mark.parametrize("name", list(V))
def test_b_to_e_output_not_provable(name):
    tx = tx_of(V[name])
    assert _route_token_received(tx, OWNER, B.MINT) is None
    f = fill(tx)
    assert f.actual_out_amount is None and not f.complete
    assert f.actual_in_amount == B.SOL_IN                 # SOL leg unaffected


def _mutated(edit):
    tx = tx_of(B.FIXTURE)
    edit(tx)
    return _route_token_received(tx, OWNER, B.MINT)


def test_owner_outflow_of_the_mint_fails_closed():
    def edit(tx):
        ix = copy.deepcopy(out_transfer(tx))
        info = ix["parsed"]["info"]
        info["source"], info["destination"] = ATA, "Elsewhere111"
        info["tokenAmount"]["amount"] = "1"
        route_group(tx).append(ix)
    assert _mutated(edit) is None


def test_owner_to_owner_transfer_fails_closed():
    def edit(tx):
        ix = copy.deepcopy(out_transfer(tx))
        ix["parsed"]["info"]["source"] = ATA
        route_group(tx).append(ix)
    assert _mutated(edit) is None


@pytest.mark.parametrize("kind", ["mintTo", "burn", "transferCheckedWithFee", "closeAccount"])
def test_other_token_instruction_on_owner_account_fails_closed(kind):
    def edit(tx):
        route_group(tx).append({"program": "spl-token-2022", "programId": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
                                "parsed": {"type": kind, "info": {"account": ATA, "destination": ATA,
                                                                   "mint": B.MINT, "amount": "1"}}})
    assert _mutated(edit) is None


def test_token_2022_route_transfer_is_accepted_when_exact():
    def edit(tx):
        ix = out_transfer(tx)
        ix["program"], ix["programId"] = "spl-token-2022", "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
    assert _mutated(edit) == OUT


def test_plain_transfer_variant_is_accepted_when_exact():
    def edit(tx):
        info = out_transfer(tx)["parsed"]
        info["type"] = "transfer"
        info["info"]["amount"] = info["info"].pop("tokenAmount")["amount"]
        info["info"].pop("mint")
    assert _mutated(edit) == OUT


@pytest.mark.parametrize("name", ["b_unrelated_top_level", "c_unrelated_route_inner"])
def test_full_buy_lifecycle_unresolved_no_position_no_resend(bw, name):
    bw.fill_tx = json.loads(V[name].read_text().replace(OWNER, bw.owner))["result"]
    bot = B.start(bw)
    bot._tick()
    (sig,) = bw.sends
    (intent,) = bot.portfolio.buy_intents
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert bot.portfolio.positions == {}
    scans = len(bot.entry_scans)
    for _ in range(3):
        bot._tick()
    assert bw.sends == [sig] and len(bot.entry_scans) == scans          # entries blocked
    bw.now += main_module.BUY_INTENT_EXPIRY_SECONDS + 1
    bot2 = B.boot(bw)                                                   # restart past expiry
    B.run(bot2, 3)
    state = B.persisted(bw)
    (intent,) = B.buys_of(state)
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert bw.sends == [sig] and bot2.entry_scans == []
    assert state["portfolio"]["positions"] == []


def test_single_output_transfer_outside_the_route_fails_closed():
    """The ONLY owner inflow sits outside the Jupiter route (top-level): it is
    not provably the swap's output (mutation M4 guard; in the other fixtures
    a route-external transfer is also a second inflow)."""
    def edit(tx):
        ix = out_transfer(tx)
        route_group(tx).remove(ix)
        tops = tx["transaction"]["message"]["instructions"]
        tops.insert(len(tops) - 1, ix)                    # before closeAccount
    assert _mutated(edit) is None
