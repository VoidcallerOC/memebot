"""Unparsed token-instruction fix (Nick @ 989d5fe).

``_route_token_received`` skipped any instruction without a parsed dict body
or a ``spl-token``/``spl-token-2022`` label, so an unparsed (or partially
decoded) SPL Token / Token-2022 instruction touching the owner's output-token
account was invisible. ``_opaque_token_touch`` now identifies the token
programs by ``programId`` (or ``programIdIndex`` into the account keys) as
well as by label; any token-program or unknown-program instruction that the
parser does not decode and whose normalised account refs intersect the
owner's accounts (or cannot be read) fails the fill closed. Raw data is never
decoded; token instructions touching only unrelated accounts are ignored. The
same guard covers the owner's wSOL accounts in ``_native_sol_spent`` /
``_native_sol_proceeds``.

Fixtures: SYNTHETIC, hash-pinned, derived from the pinned base synthetic buy
(989d5fe returned 4,200,000,000 for every one of them).
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest

import bot.main as main_module
from bot.chain import (
    JUPITER_V6_PROGRAM, TOKEN_2022_PROGRAM, _native_sol_proceeds, _native_sol_spent,
    _route_token_received, extract_fill,
)
from bot.jupiter import SOL_MINT
import tests.test_p1_buy_prebroadcast_signature as B
from tests.test_p1_buy_prebroadcast_signature import bw  # noqa: F401  (fixture)
from tests.test_p1_native_sol_settlement_e2e import (
    IN_RAW, MINT as SELL_MINT, OUT_LAMPORTS, OWNER as SELL_OWNER,
)

FIX = pathlib.Path(__file__).parent / "fixtures"
SPL = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
V = {n: FIX / f"synthetic_native_sol_buy_unparsed_{n}.json" for n in (
    "spl_token_unparsed", "token2022_unparsed", "spl_token_labelled_no_parsed", "unrelated_unparsed")}
PINS = {
    B.FIXTURE: B.FIXTURE_SHA256,
    V["spl_token_unparsed"]: "66b62e678d0009835d46022f940334c936857a8d5e1aa13da8f071567fb4e55d",
    V["token2022_unparsed"]: "ddd615be4cd8359a97b733abccc05633d9da53ac09ac6f176b1d3c566066c023",
    V["spl_token_labelled_no_parsed"]: "bd2ec17e29176fcb799dd65ca608cb8f2a11dedaec662009bc9ecaa6c624587d",
    V["unrelated_unparsed"]: "971a2c81d471cc6fcb75c4ed1798ba7899af30a0939ed18e110bd587636bd057",
}
OWNER, OUT = B.OWNER_PLACEHOLDER, 4_200_000_000
ATA, POOL_TOK = "SynthTokenAta111111111111111111111111111111", "SynthPoolTok1111111111111111111111111111111"
REAL_SELL = FIX / "native_sol_sell_fWueMsPH.json"
TEMP = "6YGszGUZxKuJ4q7jd6CBtzZtncDnmycpA46mCnk87CWD"


def tx_of(path):
    return json.loads(path.read_text())["result"]


def fill(tx):
    return extract_fill(tx, OWNER, SOL_MINT, B.MINT)


def route_group(tx):
    tops = tx["transaction"]["message"]["instructions"]
    idx = next(i for i, t in enumerate(tops) if t.get("programId") == JUPITER_V6_PROGRAM)
    return next(g for g in tx["meta"]["innerInstructions"] if g["index"] == idx)["instructions"]


def keys_of(tx):
    return [k["pubkey"] if isinstance(k, dict) else k for k in tx["transaction"]["message"]["accountKeys"]]


def with_ix(ix, path=B.FIXTURE):
    tx = tx_of(path)
    route_group(tx).append(ix)
    return tx


def test_fixtures_pinned_and_synthetic():
    for path, sha in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, path.name
    for path in V.values():
        src = json.loads(path.read_text())["_source"]
        assert src["kind"] == "SYNTHETIC" and "NOT a real" in src["note"]
        assert src["derived_from_sha256"] == B.FIXTURE_SHA256


def test_base_buy_still_exact():
    f = fill(tx_of(B.FIXTURE))
    assert f.complete and (f.actual_out_amount, f.actual_in_amount) == (OUT, B.SOL_IN)
    assert (OUT, B.SOL_IN) == (4_200_000_000, 13_333_333)


@pytest.mark.parametrize("name", ["spl_token_unparsed", "token2022_unparsed", "spl_token_labelled_no_parsed"])
def test_unparsed_token_ix_touching_owner_output_account_is_none(name):
    tx = tx_of(V[name])
    assert _route_token_received(tx, OWNER, B.MINT) is None
    f = fill(tx)
    assert f.actual_out_amount is None and not f.complete


def test_unparsed_token_ix_touching_only_unrelated_accounts_still_exact():
    f = fill(tx_of(V["unrelated_unparsed"]))
    assert f.complete and (f.actual_out_amount, f.actual_in_amount) == (OUT, B.SOL_IN)


def test_compiled_index_form_is_recognised():
    # programIdIndex + account indices (compiled form) instead of pubkeys
    tx = tx_of(B.FIXTURE)
    keys = keys_of(tx)
    tx["transaction"]["message"]["accountKeys"].append(
        {"pubkey": SPL, "signer": False, "source": "transaction", "writable": False})
    tx["meta"]["preBalances"].append(1)
    tx["meta"]["postBalances"].append(1)
    route_group(tx).append({"programIdIndex": len(keys), "accounts": [keys.index(ATA)], "data": "3Bxs"})
    assert fill(tx).actual_out_amount is None
    tx2 = tx_of(B.FIXTURE)                 # same compiled form, unrelated account: exact
    tx2["transaction"]["message"]["accountKeys"].append(
        {"pubkey": SPL, "signer": False, "source": "transaction", "writable": False})
    tx2["meta"]["preBalances"].append(1)
    tx2["meta"]["postBalances"].append(1)
    route_group(tx2).append({"programIdIndex": len(keys), "accounts": [keys.index(POOL_TOK)], "data": "3Bxs"})
    assert fill(tx2).actual_out_amount == OUT


@pytest.mark.parametrize("ix", [
    pytest.param({"programId": SPL, "accounts": [{"pubkey": ATA}], "data": "x"}, id="dict-ref"),
    pytest.param({"programId": SPL, "accounts": ATA, "data": "x"}, id="accounts-not-list"),
    pytest.param({"programId": SPL, "accounts": [999], "data": "x"}, id="bad-index"),
    pytest.param({"programId": SPL, "accounts": [None], "data": "x"}, id="odd-ref"),
    pytest.param({"accounts": [ATA], "data": "x"}, id="missing-programId"),
    pytest.param({"programIdIndex": 999, "accounts": [ATA], "data": "x"}, id="bad-programIdIndex"),
    pytest.param({"programIdIndex": True, "accounts": [ATA]}, id="bool-programIdIndex"),
    pytest.param({"program": "spl-token", "programId": SPL, "parsed": "opaque", "accounts": [ATA]},
                 id="parsed-not-dict"),
    pytest.param({"program": "spl-token", "programId": SPL, "parsed": {"type": "transfer", "info": "x"}},
                 id="info-not-dict"),
    pytest.param({"program": "spl-token", "programId": "Fake1111", "parsed": {"type": "approve", "info": {
        "source": ATA}}}, id="label-programId-mismatch"),
    pytest.param({"program": "spl-token-2022", "programId": TOKEN_2022_PROGRAM, "parsed": {
        "type": "approve", "info": {"source": ATA}}}, id="parsed-unsupported-type"),
])
def test_malformed_or_incomplete_ix_no_crash_no_fill(ix):
    tx = with_ix(ix)
    f = fill(tx)                                           # never raises
    assert f.actual_out_amount is None and not f.complete


@pytest.mark.parametrize("ix", [
    pytest.param("not-a-dict", id="non-dict-ix"),
    pytest.param(7, id="int-ix"),
])
def test_non_dict_instruction_no_crash_nothing_provable(ix):
    tx = with_ix(ix)
    f = fill(tx)
    assert f.actual_out_amount is None and f.actual_in_amount is None and not f.complete


def test_unknown_program_not_touching_owner_accounts_is_ignored():
    f = fill(with_ix({"accounts": [POOL_TOK], "data": "x"}))
    assert f.actual_out_amount == OUT
    f = fill(with_ix({"programId": "SomeAmm111111", "accounts": [ATA, POOL_TOK], "data": "x"}))
    assert f.actual_out_amount == OUT                      # known non-token program: ignored


# -- same rule on the owner's wSOL accounts ------------------------------------------

def test_real_sell_still_exact():
    f = extract_fill(tx_of(REAL_SELL), SELL_OWNER, SELL_MINT, SOL_MINT)
    assert f.complete and (f.actual_in_amount, f.actual_out_amount) == (IN_RAW, OUT_LAMPORTS)


@pytest.mark.parametrize("pid", [SPL, TOKEN_2022_PROGRAM])
def test_unparsed_token_ix_on_owner_wsol_sell_is_none(pid):
    tx = with_ix({"programId": pid, "accounts": [TEMP, "Pool111"], "data": "x"}, REAL_SELL)
    assert _native_sol_proceeds(tx, SELL_OWNER) is None
    tx = with_ix({"programId": pid, "accounts": ["Pool111", "Pool222"], "data": "x"}, REAL_SELL)
    assert _native_sol_proceeds(tx, SELL_OWNER) == OUT_LAMPORTS


def test_unparsed_token_ix_on_owner_wsol_buy_is_none():
    wsol = "SynthWsolAta1111111111111111111111111111111"
    assert _native_sol_spent(with_ix({"programId": SPL, "accounts": [wsol], "data": "x"}), OWNER) is None
    assert _native_sol_spent(with_ix({"programId": SPL, "accounts": [POOL_TOK], "data": "x"}), OWNER) == B.SOL_IN


# -- full lifecycle ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["spl_token_unparsed", "token2022_unparsed"])
def test_full_buy_lifecycle_with_unparsed_token_ix_unresolved(bw, name):
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
    assert bw.sends == [sig] and len(bot.entry_scans) == scans
    bw.now += main_module.BUY_INTENT_EXPIRY_SECONDS + 1
    bot2 = B.boot(bw)
    B.run(bot2, 3)
    state = B.persisted(bw)
    (intent,) = B.buys_of(state)
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert bw.sends == [sig] and bot2.entry_scans == [] and state["portfolio"]["positions"] == []
