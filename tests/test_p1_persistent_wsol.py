"""Persistent wSOL fix (Nick @ b9690ee).

``extract_fill`` used to accept SOL output from the owner's wSOL
token-balance delta first, so a persistent (never closed) wSOL account
skipped the strict parser: an unrelated wSOL transfer into it inflated the
proceeds and the fill looked complete. SOL out is now accepted only from
``_native_sol_proceeds`` (exactly one pool -> owner wSOL inflow as a Jupiter
route CPI, no owner wSOL outflow, no owner -> owner transfer, every lamport
reconciled); a non-zero wSOL token-balance delta must equal it exactly.

Fixtures (SYNTHETIC, hash-pinned, derived from the pinned real fWueMsPH sell):
* ``synthetic_native_sol_sell_persistent_wsol.json``: the same sell settling
  into a pre-existing owner wSOL account (no create, no close).
* ``..._unrelated_top_level.json`` / ``..._unrelated_route_inner.json``: plus
  one unrelated 1,000,000 wSOL transfer into that account, outside the route
  / as a route CPI. b9690ee settled both at 9,656,067 (true 8,656,067).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from bot.chain import _native_sol_proceeds, extract_fill
from bot.jupiter import SOL_MINT
import tests.test_p1_prebroadcast_signature as S
from tests.test_p1_prebroadcast_signature import rw  # noqa: F401  (fixture)
from tests.test_p1_native_sol_settlement_e2e import (
    IN_RAW, MINT, OUT_LAMPORTS, OWNER, intents_of, position,
)

FIX = pathlib.Path(__file__).parent / "fixtures"
REAL = FIX / "native_sol_sell_fWueMsPH.json"
PERSIST = FIX / "synthetic_native_sol_sell_persistent_wsol.json"
TOP = FIX / "synthetic_native_sol_sell_persistent_wsol_unrelated_top_level.json"
INNER = FIX / "synthetic_native_sol_sell_persistent_wsol_unrelated_route_inner.json"
PINS = {
    REAL: "730382ed56c73866e74cc80bbadfa75732c9ab943d3f4d8d971d61ec14be75b1",
    PERSIST: "4b9c3635d9e45fd312cb3eb1563dfbc86b49ffd1a7966757805a8966b4c6c583",
    TOP: "b42f2dde369ad2ee36c8a2611114aab4e8d6c1decde0f8d6a8334627d1e98bf1",
    INNER: "c98c3adc3c044f52e774ae672bd0ad06a1ba219d78b5982864bfa3addf80df40",
}
TEMP = "6YGszGUZxKuJ4q7jd6CBtzZtncDnmycpA46mCnk87CWD"
X0, DONATION = 25_000_000, 1_000_000


def tx_of(path):
    return json.loads(path.read_text())["result"]


def fill(tx):
    return extract_fill(tx, OWNER, MINT, SOL_MINT)


def wsol_delta(tx):
    amt = lambda side: sum(int(e["uiTokenAmount"]["amount"]) for e in tx["meta"][side]
                           if e["mint"] == SOL_MINT and e.get("owner") == OWNER)
    return amt("postTokenBalances") - amt("preTokenBalances")


def test_fixtures_pinned_and_synthetic():
    for path, sha in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, path.name
    for path, parent in ((PERSIST, REAL), (TOP, PERSIST), (INNER, PERSIST)):
        src = json.loads(path.read_text())["_source"]
        assert src["kind"] == "SYNTHETIC" and "NOT a real" in src["note"]
        assert src["derived_from"] == parent.name and src["derived_from_sha256"] == PINS[parent]


def test_persistent_fixture_has_no_create_or_close():
    tx = tx_of(PERSIST)
    kinds = [(ix.get("parsed") or {}).get("type") for ix in tx["transaction"]["message"]["instructions"]
             if isinstance(ix.get("parsed"), dict)]
    assert "closeAccount" not in kinds and "createIdempotent" not in kinds
    assert wsol_delta(tx) == OUT_LAMPORTS                 # tokens stay wrapped in the account


def test_real_sell_still_exact():
    f = fill(tx_of(REAL))
    assert f.complete and (f.actual_in_amount, f.actual_out_amount) == (IN_RAW, OUT_LAMPORTS)


def test_valid_persistent_wsol_sell_settles_exactly():
    tx = tx_of(PERSIST)
    assert _native_sol_proceeds(tx, OWNER) == OUT_LAMPORTS
    f = fill(tx)
    assert f.complete and f.out_decimals == 9
    assert (f.actual_in_amount, f.actual_out_amount) == (IN_RAW, OUT_LAMPORTS)


@pytest.mark.parametrize("path", [TOP, INNER], ids=["top-level", "route-inner"])
def test_unrelated_wsol_into_persistent_account_is_none(path):
    tx = tx_of(path)
    assert wsol_delta(tx) == OUT_LAMPORTS + DONATION == 9_656_067   # what b9690ee settled
    assert _native_sol_proceeds(tx, OWNER) is None
    f = fill(tx)
    assert f.actual_out_amount is None and not f.complete


@pytest.mark.parametrize("skew", [1, -1, OUT_LAMPORTS])
def test_token_delta_not_equal_to_strict_result_is_none(skew):
    tx = tx_of(PERSIST)
    for e in tx["meta"]["postTokenBalances"]:
        if e["mint"] == SOL_MINT and e.get("owner") == OWNER:
            e["uiTokenAmount"]["amount"] = str(int(e["uiTokenAmount"]["amount"]) + skew)
    assert _native_sol_proceeds(tx, OWNER) == OUT_LAMPORTS   # lamports alone still reconcile
    f = fill(tx)
    assert f.actual_out_amount is None and not f.complete


def test_valid_persistent_wsol_full_flow_books_once(rw):
    rw.fill_tx = json.loads(PERSIST.read_text().replace(OWNER, rw.owner))["result"]
    bot = S.start(rw)
    bot._tick()
    (sig,) = rw.sends
    S.booked_once(rw, sig)


@pytest.mark.parametrize("path", [TOP, INNER], ids=["top-level", "route-inner"])
def test_unrelated_wsol_full_flow_unresolved_no_booking_no_resend(rw, path):
    rw.fill_tx = json.loads(path.read_text().replace(OWNER, rw.owner))["result"]
    bot = S.start(rw)
    bot._tick()
    (sig,) = rw.sends
    for _ in range(3):
        bot._tick()
    rw.now += S.EXPIRY + 1
    bot2 = S.boot(rw)                                     # restart past expiry
    S.run(bot2, 3)
    state = S.persisted(rw)
    (intent,) = intents_of(state)
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert intent.get("booked_at") is None and intent.get("actual_out_amount") is None
    assert rw.sends == [sig] and rw.builds == 1 and bot2.jup.quotes == [] and bot2.entry_scans == []
    assert state["portfolio"]["realized_pnl"] == 0.0 and position(state)["raw_tokens"] == IN_RAW


# -- buy-side audit: SOL input never comes from a wSOL token-balance delta ------------

def test_buy_sol_input_ignores_wsol_token_delta():
    """SOL in is only ``_native_sol_spent``'s proven principal. A persistent
    wSOL account whose token balance moves differently (here -10,000,000 vs
    the proven 13,333,333) never replaces it (mutation M3 guard)."""
    import tests.test_p1_buy_prebroadcast_signature as B
    tx = json.loads(B.FIXTURE.read_text())["result"]
    keys = [k["pubkey"] for k in tx["transaction"]["message"]["accountKeys"]]
    idx = keys.index("SynthWsolAta1111111111111111111111111111111")
    bal = lambda amt: {"accountIndex": idx, "mint": SOL_MINT, "owner": B.OWNER_PLACEHOLDER,
                       "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                       "uiTokenAmount": {"amount": str(amt), "decimals": 9}}
    tx["meta"]["preTokenBalances"].append(bal(20_000_000))
    tx["meta"]["postTokenBalances"].append(bal(10_000_000))
    f = extract_fill(tx, B.OWNER_PLACEHOLDER, SOL_MINT, B.MINT)
    assert f.actual_in_amount == B.SOL_IN == 13_333_333 and f.complete
