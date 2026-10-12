"""Token-2022 output tokens as real RPC encodes them.

Root cause: mainnet ``getTransaction(jsonParsed)`` labels Token-2022
instructions ``"program": "spl-token"`` while ``programId`` is Token-2022.
``_opaque_token_touch`` required the label's own program id, so the parsed
``initializeAccount3`` / ``transferChecked`` on the buyer's Token-2022 account
counted as undecoded and every real Token-2022 buy came back incomplete.

Fix: for the BOUGHT token only, the program id is authoritative: label
``spl-token`` is accepted with either token program id, ``spl-token-2022``
only with Token-2022. Any other pairing stays undecoded (fail closed). The
wSOL (SOL principal) parsers keep the strict one-to-one map.

Fixtures are REAL, unmodified, hash-pinned responses; provenance in
``tests/fixtures/REAL_FIXTURES.md``. Mutations are applied to deep copies.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest

from bot.chain import (
    SOL_MINT,
    TOKEN_2022_PROGRAM,
    _native_sol_spent,
    _route_token_received,
    extract_fill,
)

FIX = pathlib.Path(__file__).parent / "fixtures"
TOKENKEG = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
JUP = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"

T22 = {
    "file": "real_token2022_buy_41WvNWct.json",
    "sha256": "5c4a501481daa433409f3506e16b7b9a9bac532d3c24bcd7a97a6ccad7b40eae",
    "sig": "41WvNWctkG8GQQ9w3tDMG1uQA1upTGtMqYx5Eu5dEKenQQkthdH2RwLX2QPmRFMg8jQbh3CZjmbzqhinVGaNuzmy",
    "slot": 456150440,
    "owner": "FcQQtjEGTNRXDJQ6e6mtAHPe6syW2wdF9ENN8T6T9fZQ",
    "mint": "J6Y3vyed8iStNBwcHyoc3ct63kd9SsxsTainjgNYzQJ4",
    "sol_in": 7_202_296, "raw_out": 15_195_659_864, "decimals": 6,
    "fee": 10_000, "token_ata_rent": 1_513_840,
}
SPL = {
    "file": "real_spl_buy_4GVYwAeQ.json",
    "sha256": "ce5fb194b2f49a40ecaa57a590cd4478ef27368d05ddc9d051fabb8be06e8b7b",
    "sig": "4GVYwAeQuRwzWirFyrgXneJro7SHcYsvF3cVWa6YVUBHFHedzAMKsqiBDQghH4EqDK3hefrsHu7UWV7dGA8dN4gV",
    "slot": 456122647,
    "owner": "7hARSicdYvWR9saun1F3JDQjnnHY3k6wvNH2XXp2EA6A",
    "mint": "85VBFQZC9TZkfaptBWjvUw7YbZjy52A6mjtPGjstQAmQ",
    "sol_in": 50_000_000, "raw_out": 254_683_995, "decimals": 6,
    "fee": 55_778, "token_ata_rent": 0,
}


def load(spec):
    data = (FIX / spec["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == spec["sha256"]
    tx = json.loads(data)["result"]
    assert tx["transaction"]["signatures"][0] == spec["sig"] and tx["slot"] == spec["slot"]
    return tx


def all_ixs(tx):
    """(inner group index or None, instruction) for every instruction."""
    out = [(None, ix) for ix in tx["transaction"]["message"]["instructions"]]
    for g in tx["meta"]["innerInstructions"]:
        out += [(g["index"], ix) for ix in g["instructions"]]
    return out


def route_index(tx):
    (i,) = [i for i, ix in enumerate(tx["transaction"]["message"]["instructions"])
            if ix.get("programId") == JUP]
    return i


def derive(tx, spec):
    """Expected values straight from instructions and raw balances, without
    the production parser: the owner's wSOL accounts are those initialised
    for (mint=SOL, owner); the principal is the wSOL transfer out of them; the
    output is the single transfer into the owner's output-mint account, and it
    must equal the owner's raw post-pre token balance; the owner's lamport
    drop must equal principal + fee + new token-account rent."""
    owner, mint = spec["owner"], spec["mint"]
    keys = [k["pubkey"] for k in tx["transaction"]["message"]["accountKeys"]]
    meta = tx["meta"]
    wsol = {ix["parsed"]["info"]["account"] for _, ix in all_ixs(tx)
            if isinstance(ix.get("parsed"), dict) and ix["parsed"].get("type") == "initializeAccount3"
            and ix["parsed"]["info"].get("mint") == SOL_MINT and ix["parsed"]["info"].get("owner") == owner}
    out_accts = {keys[e["accountIndex"]] for e in meta["postTokenBalances"]
                 if e["mint"] == mint and e["owner"] == owner}

    def amt(info):
        return int(info["tokenAmount"]["amount"]) if "tokenAmount" in info else int(info["amount"])

    xfers = [(g, ix["parsed"]["info"]) for g, ix in all_ixs(tx)
             if isinstance(ix.get("parsed"), dict) and ix["parsed"].get("type") in ("transfer", "transferChecked")
             and ix.get("program") == "spl-token"]
    wsol_out = [(g, amt(i)) for g, i in xfers if i.get("source") in wsol]
    deliveries = [(g, amt(i)) for g, i in xfers if i.get("destination") in out_accts]
    assert [g for g, _ in wsol_out] == [route_index(tx)] and [g for g, _ in deliveries] == [route_index(tx)]

    def raw(entries):
        return sum(int(e["uiTokenAmount"]["amount"]) for e in entries
                   if e["mint"] == mint and e["owner"] == owner)
    delta = raw(meta["postTokenBalances"]) - raw(meta["preTokenBalances"])
    (dec,) = {e["uiTokenAmount"]["decimals"] for e in meta["postTokenBalances"] if e["mint"] == mint}
    principal, delivered = wsol_out[0][1], deliveries[0][1]
    assert delivered == delta
    oi = keys.index(owner)
    assert meta["preBalances"][oi] - meta["postBalances"][oi] == principal + meta["fee"] + spec["token_ata_rent"]
    return principal, delivered, dec, meta["fee"]


@pytest.mark.parametrize("spec", [T22, SPL], ids=["token2022", "spl"])
def test_real_buy_independent_derivation_matches_pinned_constants(spec):
    tx = load(spec)
    assert derive(tx, spec) == (spec["sol_in"], spec["raw_out"], spec["decimals"], spec["fee"])


def test_real_t22_fixture_uses_spl_token_label_with_token2022_id():
    tx = load(T22)
    labels = {(ix.get("program"), ix.get("programId")) for _, ix in all_ixs(tx)
              if ix.get("programId") == TOKEN_2022_PROGRAM}
    assert labels == {("spl-token", TOKEN_2022_PROGRAM)}


@pytest.mark.parametrize("spec", [T22, SPL], ids=["token2022", "spl"])
def test_real_buy_extract_fill_exact_and_complete(spec):
    fill = extract_fill(load(spec), spec["owner"], SOL_MINT, spec["mint"])
    assert fill.actual_in_amount == spec["sol_in"] and fill.in_decimals == 9
    assert fill.actual_out_amount == spec["raw_out"] and fill.out_decimals == spec["decimals"]
    assert fill.fee_lamports == spec["fee"] and fill.complete


def delivery(tx, spec):
    owner_accts = {tx["transaction"]["message"]["accountKeys"][e["accountIndex"]]["pubkey"]
                   for e in tx["meta"]["postTokenBalances"] if e["mint"] == spec["mint"]}
    (ix,) = [ix for _, ix in all_ixs(tx) if isinstance(ix.get("parsed"), dict)
             and ix["parsed"]["type"] == "transferChecked"
             and ix["parsed"]["info"].get("destination") in owner_accts]
    return ix


def assert_out_unproven(tx, spec):
    assert _route_token_received(tx, spec["owner"], spec["mint"]) is None
    fill = extract_fill(tx, spec["owner"], SOL_MINT, spec["mint"])
    assert fill.actual_out_amount is None and not fill.complete


@pytest.mark.parametrize("label,pid", [
    ("spl-token-2022", TOKENKEG),                                # label says T22, id says Tokenkeg
    ("spl-token", "Rand0mProgram1111111111111111111111111111111"),  # token label, foreign id
    ("spl-token-2022", "Rand0mProgram1111111111111111111111111111111"),
])
def test_conflicting_label_and_program_id_fail_closed(label, pid):
    tx = copy.deepcopy(load(T22))
    ix = delivery(tx, T22)
    ix["program"], ix["programId"] = label, pid
    assert_out_unproven(tx, T22)


def test_label_with_mismatched_resolved_program_id_index_fails_closed():
    tx = copy.deepcopy(load(T22))
    ix = delivery(tx, T22)
    keys = [k["pubkey"] for k in tx["transaction"]["message"]["accountKeys"]]
    del ix["programId"]
    ix["programIdIndex"] = keys.index("11111111111111111111111111111111")
    assert_out_unproven(tx, T22)


def test_label_with_resolved_token2022_program_id_index_is_accepted():
    tx = copy.deepcopy(load(T22))
    ix = delivery(tx, T22)
    keys = [k["pubkey"] for k in tx["transaction"]["message"]["accountKeys"]]
    del ix["programId"]
    ix["programIdIndex"] = keys.index(TOKEN_2022_PROGRAM)
    assert extract_fill(tx, T22["owner"], SOL_MINT, T22["mint"]).actual_out_amount == T22["raw_out"]


@pytest.mark.parametrize("pid", [TOKEN_2022_PROGRAM, TOKENKEG])
def test_unrelated_unparsed_token_ix_still_exact(pid):
    tx = copy.deepcopy(load(T22))
    group = next(g for g in tx["meta"]["innerInstructions"] if g["index"] == route_index(tx))
    group["instructions"].append({"programId": pid, "accounts": ["9DDMysUnrelatedPool111111111111111111111111"],
                                  "data": "3Bxs4h24hBtQy9rw", "stackHeight": 3})
    fill = extract_fill(tx, T22["owner"], SOL_MINT, T22["mint"])
    assert (fill.actual_in_amount, fill.actual_out_amount) == (T22["sol_in"], T22["raw_out"])


def test_unparsed_token2022_ix_on_owner_account_fails_closed():
    tx = copy.deepcopy(load(T22))
    dst = delivery(tx, T22)["parsed"]["info"]["destination"]
    group = next(g for g in tx["meta"]["innerInstructions"] if g["index"] == route_index(tx))
    group["instructions"].append({"programId": TOKEN_2022_PROGRAM, "accounts": [dst], "data": "3Bxs4h24hBtQy9rw"})
    assert_out_unproven(tx, T22)


def test_transfer_checked_with_fee_stays_unresolved():
    tx = copy.deepcopy(load(T22))
    ix = delivery(tx, T22)
    ix["parsed"]["type"] = "transferCheckedWithFee"
    ix["parsed"]["info"]["feeAmount"] = {"amount": "1", "decimals": 6}
    assert_out_unproven(tx, T22)


def test_fee_bearing_received_less_than_sent_stays_unresolved():
    tx = copy.deepcopy(load(T22))
    for e in tx["meta"]["postTokenBalances"]:
        if e["mint"] == T22["mint"] and e["owner"] == T22["owner"]:
            e["uiTokenAmount"]["amount"] = str(T22["raw_out"] - 1)
    assert_out_unproven(tx, T22)


def test_wsol_parser_keeps_strict_label_map():
    """Scope: only the bought-token parser accepts spl-token + Token-2022 id.
    A wSOL instruction claiming the Token-2022 id stays unproven."""
    tx = copy.deepcopy(load(T22))
    for _, ix in all_ixs(tx):
        p = ix.get("parsed")
        if isinstance(p, dict) and p.get("type") == "transferChecked" and p["info"].get("mint") == SOL_MINT:
            ix["programId"] = TOKEN_2022_PROGRAM
    assert _native_sol_spent(tx, T22["owner"]) is None
    fill = extract_fill(tx, T22["owner"], SOL_MINT, T22["mint"])
    assert fill.actual_in_amount is None and not fill.complete
