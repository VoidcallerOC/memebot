"""P1: extract_fill reads native SOL proceeds when the temp wSOL account closes.

Real fixture (public on-chain data, unmodified getTransaction result):
  tests/fixtures/native_sol_sell_fWueMsPH.json
  signature fWueMsPHe5MiGePaSXQiS1B9VEMjA7MH65QieuMKNme2PWTTecoQgHUnKvUyRTCXgrA4rfSygZzCGPvpwhmqo6f
  slot 455709567, v0, finalized, err null; fetched read-only from the public
  mainnet-beta RPC (getTransaction, jsonParsed, maxSupportedTransactionVersion 0).
  Owner HckQ93Xq... sells 274,425 raw XsCPL9dN... (Token-2022, 8 dp) through
  Jupiter v6 into a temp wSOL ATA (6YGszGUZ..., createIdempotent +
  initializeAccount3) that receives 8,656,067 lamports of wSOL and is closed
  to the owner in the same tx. Owner lamports: -10,143 fee -1,488,440 rent
  +1,488,440 refund +8,656,067 = +8,645,924.
On main extract_fill returned actual_out_amount=None for this transaction.
Every synthetic case below is a documented, targeted edit of that fixture.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from bot.chain import extract_fill

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "native_sol_sell_fWueMsPH.json"
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
OWNER = "HckQ93Xqjjo8mwt5pNPWvyCTZXQZ858rvzmm7ZRrZg9t"
MINT = "XsCPL9dNWBMvFtTmwcCA5v3xWPSMEBCszbQdiLLq6aN"
TEMP = "6YGszGUZxKuJ4q7jd6CBtzZtncDnmycpA46mCnk87CWD"
IN_RAW, OUT_LAMPORTS, FEE = 274_425, 8_656_067, 10_143
OWNER_IDX, TEMP_IDX, OWNER_X_IDX = 0, 3, 4


def real():
    data = json.loads(FIXTURE.read_text())
    assert data["_source"]["signature"].startswith("fWueMsPHe5")
    return copy.deepcopy(data["result"])


def fill(tx, owner=OWNER, mint=MINT, out=SOL):
    return extract_fill(tx, owner, mint, out)


def all_ixs(tx):
    out = list(tx["transaction"]["message"]["instructions"])
    for group in tx["meta"]["innerInstructions"]:
        out += group["instructions"]
    return out


def find(tx, kind, **match):
    hits = [ix for ix in all_ixs(tx) if isinstance(ix.get("parsed"), dict)
            and ix["parsed"].get("type") == kind
            and all(ix["parsed"]["info"].get(k) == v for k, v in match.items())]
    assert len(hits) == 1, (kind, match, len(hits))
    return hits[0]


def wsol_in(tx):
    return find(tx, "transferChecked", destination=TEMP)


def remove(tx, ix):
    msg = tx["transaction"]["message"]
    if ix in msg["instructions"]:
        msg["instructions"].remove(ix)
        return
    for group in tx["meta"]["innerInstructions"]:
        if ix in group["instructions"]:
            group["instructions"].remove(ix)
            return
    raise AssertionError("instruction not found")


def add_top(tx, program, kind, info):
    tx["transaction"]["message"]["instructions"].append(
        {"program": program, "programId": "x", "parsed": {"type": kind, "info": info}, "stackHeight": None})


def bump_post(tx, idx, delta):
    tx["meta"]["postBalances"][idx] += delta


# -- the real transaction -------------------------------------------------------

def test_real_fixture_native_sol_proceeds():
    f = fill(real())
    assert f.actual_in_amount == IN_RAW and f.in_decimals == 8
    assert f.actual_out_amount == OUT_LAMPORTS and f.out_decimals == 9
    assert type(f.actual_out_amount) is int  # raw lamports, the existing contract
    assert f.fee_lamports == FEE
    assert f.complete


def test_real_fixture_temp_account_absent_from_token_balances():
    tx = real()
    meta = tx["meta"]
    for key in ("preTokenBalances", "postTokenBalances"):
        assert all(b["accountIndex"] != TEMP_IDX for b in meta[key])
    keys = [k["pubkey"] for k in tx["transaction"]["message"]["accountKeys"]]
    assert keys[TEMP_IDX] == TEMP and keys[OWNER_IDX] == OWNER
    delta = meta["postBalances"][OWNER_IDX] - meta["preBalances"][OWNER_IDX]
    assert delta == OUT_LAMPORTS - FEE == 8_645_924
    assert find(tx, "createAccount", newAccount=TEMP)["parsed"]["info"]["lamports"] == 1_488_440
    find(tx, "closeAccount", account=TEMP, destination=OWNER)
    # intermediate hops go through owner USDC / pump accounts, never a wSOL one
    assert wsol_in(tx)["parsed"]["info"]["source"] != TEMP


# -- persistent wSOL and token -> token keep existing behaviour ----------------

def test_persistent_wsol_output_uses_token_balance_delta():
    tx = real()
    remove(tx, find(tx, "closeAccount", account=TEMP))
    bal = lambda amt: {"accountIndex": TEMP_IDX, "mint": SOL, "owner": OWNER,
                       "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                       "uiTokenAmount": {"amount": str(amt), "decimals": 9}}
    tx["meta"]["preTokenBalances"].append(bal(0))
    tx["meta"]["postTokenBalances"].append(bal(OUT_LAMPORTS))
    f = fill(tx)
    assert f.actual_out_amount == OUT_LAMPORTS and f.out_decimals == 9


def test_token_to_token_output_unchanged():
    f = fill(real(), out=USDC)  # owner USDC account is 0 -> 0: no USDC output
    assert f.actual_in_amount == IN_RAW
    assert f.actual_out_amount is None


# -- fee payer, rent ------------------------------------------------------------

def _swap_payer(tx, other_idx):
    """Make account ``other_idx`` the fee payer (index 0) and move the owner there."""
    msg = tx["transaction"]["message"]
    keys, meta = msg["accountKeys"], tx["meta"]
    keys[0], keys[other_idx] = keys[other_idx], keys[0]
    for arr in ("preBalances", "postBalances"):
        meta[arr][0], meta[arr][other_idx] = meta[arr][other_idx], meta[arr][0]
    for b in meta["preTokenBalances"] + meta["postTokenBalances"]:
        if b["accountIndex"] == 0:
            b["accountIndex"] = other_idx
        elif b["accountIndex"] == other_idx:
            b["accountIndex"] = 0
    return other_idx


def test_fee_payer_not_owner():
    tx = real()
    last = len(tx["transaction"]["message"]["accountKeys"]) - 1
    owner_idx = _swap_payer(tx, last)
    bump_post(tx, owner_idx, +FEE)   # the owner no longer pays the fee ...
    bump_post(tx, 0, -FEE)           # ... the new fee payer does
    assert fill(tx).actual_out_amount == OUT_LAMPORTS


def test_fee_payer_not_owner_but_owner_charged_fee_fails_closed():
    tx = real()
    _swap_payer(tx, len(tx["transaction"]["message"]["accountKeys"]) - 1)
    assert fill(tx).actual_out_amount is None  # owner delta short by an unexplained fee


def test_rent_for_new_non_wsol_account_is_not_a_proceed_deduction():
    tx = real()  # owner also funds an intermediate account that stays open
    add_top(tx, "system", "createAccount", {"source": OWNER, "newAccount": "NewAta1111",
                                            "lamports": 2_039_280, "space": 165, "owner": "Tokenkeg"})
    bump_post(tx, OWNER_IDX, -2_039_280)
    assert fill(tx).actual_out_amount == OUT_LAMPORTS


def test_unexplained_owner_debit_fails_closed():
    tx = real()
    bump_post(tx, OWNER_IDX, -2_039_280)  # rent/tip paid with no parsed evidence
    assert fill(tx).actual_out_amount is None


def test_temp_rent_not_refunded_fails_closed():
    tx = real()  # closeAccount refunds to someone else
    find(tx, "closeAccount", account=TEMP)["parsed"]["info"]["destination"] = "Elsewhere111"
    assert fill(tx).actual_out_amount is None


# -- unrelated SOL must never inflate ------------------------------------------

def test_unrelated_sol_transfer_into_owner_fails_closed():
    tx = real()
    add_top(tx, "system", "transfer", {"source": "Gifter1111", "destination": OWNER, "lamports": 1_000})
    bump_post(tx, OWNER_IDX, +1_000)
    assert fill(tx).actual_out_amount is None


def test_unexplained_owner_credit_fails_closed():
    tx = real()
    bump_post(tx, OWNER_IDX, +1_000)  # e.g. a program crediting lamports directly
    assert fill(tx).actual_out_amount is None


def test_sol_transfer_out_of_owner_fails_closed():
    tx = real()
    add_top(tx, "system", "transfer", {"source": OWNER, "destination": "Tip1111", "lamports": 1_000})
    bump_post(tx, OWNER_IDX, -1_000)
    assert fill(tx).actual_out_amount is None


# -- malformed / failed / wrong --------------------------------------------------

def test_missing_meta():
    tx = real()
    del tx["meta"]
    f = fill(tx)
    assert (f.actual_in_amount, f.actual_out_amount) == (None, None)


@pytest.mark.parametrize("value", ["delete", None, "bad"])
def test_missing_inner_instructions(value):
    tx = real()
    if value == "delete":
        del tx["meta"]["innerInstructions"]
    else:
        tx["meta"]["innerInstructions"] = value
    f = fill(tx)
    assert f.actual_out_amount is None and not f.complete


def test_failed_transaction_yields_nothing():
    tx = real()
    tx["meta"]["err"] = {"InstructionError": [3, {"Custom": 6001}]}
    f = fill(tx)
    assert (f.actual_in_amount, f.actual_out_amount) == (None, None)
    assert f.fee_lamports == FEE


def test_wrong_owner():
    f = fill(real(), owner="SomeoneElse1111111111111111111111111111111")
    assert (f.actual_in_amount, f.actual_out_amount) == (None, None)


def test_wrong_input_mint_is_incomplete():
    f = fill(real(), mint="WrongMint1111111111111111111111111111111111")
    assert f.actual_in_amount is None and not f.complete


def test_mismatched_input_reports_actual_amount():
    tx = real()
    post = next(b for b in tx["meta"]["postTokenBalances"] if b["accountIndex"] == OWNER_X_IDX)
    post["uiTokenAmount"]["amount"] = "1"  # one raw unit left behind
    assert fill(tx).actual_in_amount == IN_RAW - 1  # != intent raw -> #32 will not settle


def test_zero_output():
    tx = real()
    wsol_in(tx)["parsed"]["info"]["tokenAmount"]["amount"] = "0"
    bump_post(tx, OWNER_IDX, -OUT_LAMPORTS)
    assert fill(tx).actual_out_amount is None


@pytest.mark.parametrize("delta", [+1, -1])
def test_parsed_transfer_disagrees_with_balances(delta):
    tx = real()
    amt = wsol_in(tx)["parsed"]["info"]["tokenAmount"]
    amt["amount"] = str(OUT_LAMPORTS + delta)
    assert fill(tx).actual_out_amount is None


def test_non_wsol_transfer_into_wsol_account_fails_closed():
    tx = real()
    wsol_in(tx)["parsed"]["info"]["mint"] = USDC
    assert fill(tx).actual_out_amount is None


def test_wrap_inside_sell_fails_closed():
    tx = real()
    add_top(tx, "spl-token", "syncNative", {"account": TEMP})
    assert fill(tx).actual_out_amount is None


def test_wsol_out_of_owner_account_outside_route_fails_closed():
    # Re-review P1: a top-level (non-route) wSOL transfer out of the temp
    # account reconciles the balance equation but is not provably part of the
    # swap. It used to be netted (OUT - 1,000); now the fill is not proven.
    tx = real()
    add_top(tx, "spl-token", "transfer", {"source": TEMP, "destination": "Pool1111",
                                          "authority": OWNER, "amount": "1000"})
    bump_post(tx, OWNER_IDX, -1_000)
    assert fill(tx).actual_out_amount is None


def test_wsol_out_of_owner_account_inside_route_is_netted():
    tx = real()  # a later route hop (CPI of the Jupiter instruction) moves 1,000 out
    route = next(g for g in tx["meta"]["innerInstructions"] if g["index"] == 3)
    route["instructions"].append({"program": "spl-token", "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                                  "parsed": {"type": "transfer", "info": {
                                      "source": TEMP, "destination": "Pool1111",
                                      "authority": OWNER, "amount": "1000"}},
                                  "stackHeight": 2})
    bump_post(tx, OWNER_IDX, -1_000)
    assert fill(tx).actual_out_amount == OUT_LAMPORTS - 1_000


def test_unparseable_amount_fails_closed():
    tx = real()
    wsol_in(tx)["parsed"]["info"]["tokenAmount"]["amount"] = "8.6e6"
    assert fill(tx).actual_out_amount is None


# -- wSOL account discovery variants -------------------------------------------

def test_found_via_initialize_account3_only():
    tx = real()
    find(tx, "createIdempotent", account=TEMP)["parsed"]["info"]["mint"] = "NotWsol"
    assert fill(tx).actual_out_amount == OUT_LAMPORTS


def test_found_via_ata_create_idempotent_only():
    tx = real()
    remove(tx, find(tx, "initializeAccount3", account=TEMP))
    assert fill(tx).actual_out_amount == OUT_LAMPORTS


@pytest.mark.parametrize("kind", ["initializeAccount", "initializeAccount2"])
def test_found_via_other_initialize_variants(kind):
    tx = real()
    find(tx, "createIdempotent", account=TEMP)["parsed"]["info"]["mint"] = "NotWsol"
    find(tx, "initializeAccount3", account=TEMP)["parsed"]["type"] = kind
    assert fill(tx).actual_out_amount == OUT_LAMPORTS


def test_ata_create_plain_variant():
    tx = real()
    remove(tx, find(tx, "initializeAccount3", account=TEMP))
    find(tx, "createIdempotent", account=TEMP)["parsed"]["type"] = "create"
    assert fill(tx).actual_out_amount == OUT_LAMPORTS


def test_unrecognised_temp_account_fails_closed():
    tx = real()
    find(tx, "createIdempotent", account=TEMP)["parsed"]["info"]["mint"] = "NotWsol"
    remove(tx, find(tx, "initializeAccount3", account=TEMP))
    assert fill(tx).actual_out_amount is None


def test_temp_account_owned_by_someone_else_fails_closed():
    tx = real()
    find(tx, "createIdempotent", account=TEMP)["parsed"]["info"]["wallet"] = "Other111"
    find(tx, "initializeAccount3", account=TEMP)["parsed"]["info"]["owner"] = "Other111"
    assert fill(tx).actual_out_amount is None


@pytest.mark.parametrize("tx", [{}, {"meta": None}, {"meta": {"err": None}},
                                {"meta": {"innerInstructions": [], "preBalances": "x"}}])
def test_garbage_never_raises(tx):
    f = extract_fill(tx, OWNER, MINT, SOL)
    assert f.actual_out_amount is None
