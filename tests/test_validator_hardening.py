"""P1-3: pre-sign validator hardening, grounded in the REAL dry-run builds.

Evidence (``tests/fixtures/jupiter_dryrun/builds.json``, 6 real Metis builds:
buy/sell x USDC, BONK, PYUSD Token-2022), pinned by the tests below:

* Fixed route slots 0-8 always carry the same compiled signer/writable flags
  (``ROUTE_FIXED_FLAGS``); ATA creates, the wSOL funding transfer,
  SyncNative and CloseAccount likewise (``ATA_CREATE_FLAGS`` etc.).
* Every build creates the owner's wSOL ATA exactly once (classic program);
  buys additionally create the owner's ATA for the bought mint exactly once,
  under the same token program as route slot 0, and that ATA is route slot 3.
  Sells create no token ATA. Route slot 0 of a sell is the classic program
  (wSOL output) even for a Token-2022 input.
* Every quote ``ammKey`` appears among the route's remaining accounts (>= 9).
* LIMITATION (proven by the evidence, so NOT enforced): every legitimate build
  passes the owner (signer), the owner's wSOL ATA and the owner's token ATA
  again in the AMM slots (>= 9). Rejecting them would refuse every real swap.

Mutations below are SYNTHETIC edits of those real builds (or of the
synthetic factory). Nothing is signed or sent.
"""
from __future__ import annotations

import copy

import pytest
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageHeader
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from bot.swap_policy import (
    ATA_CREATE_FLAGS, ATA_PROGRAM, JUPITER_V6_PROGRAM, ROUTE_FIXED_FLAGS, SOL_MINT,
    SPL_TOKEN_PROGRAM, SYSTEM_PROGRAM, TOKEN_2022_PROGRAM, _account_flags, _ata,
    resolve_account_keys, swap_transaction_error,
)
from tests.swap_tx_factory import FAKE_AMM, conformant_instructions, direct_route, unsigned_tx_bytes
from tests.test_direct_route_policy import (
    DATA, OWNER, case, check, find, idx_of, ix, mutate, rebuild, tables,
)

pytestmark = pytest.mark.real_swap_policy

BUILDS = sorted(n for n, c in DATA["cases"].items() if (c.get("swap_response") or {}).get("swapTransaction"))
BUYS = [n for n in BUILDS if n.startswith("buy")]
SELLS = [n for n in BUILDS if n.startswith("sell")]


def route(msg):
    keys = resolve_account_keys(msg, tables())
    r = next(x for x in msg.instructions if keys[x.program_id_index] == JUPITER_V6_PROGRAM)
    return keys, list(bytes(r.accounts))


def token_mint(name):
    q, _ = case(name)
    return q["outputMint"] if q["inputMint"] == SOL_MINT else q["inputMint"]


# -- evidence pins ------------------------------------------------------------

def test_six_real_builds_present():
    assert len(BUILDS) == 6 and len(BUYS) == 3 and len(SELLS) == 3


@pytest.mark.parametrize("name", BUILDS)
def test_evidence_fixed_route_slot_flags(name):
    _, msg = case(name)
    keys, accts = route(msg)
    flags = _account_flags(msg, len(keys))
    assert [flags[i] for i in accts[:9]] == list(ROUTE_FIXED_FLAGS)


@pytest.mark.parametrize("name", BUILDS)
def test_evidence_ata_creates(name):
    _, msg = case(name)
    keys = resolve_account_keys(msg, tables())
    flags = _account_flags(msg, len(keys))
    creates = [list(bytes(x.accounts)) for x in msg.instructions if keys[x.program_id_index] == ATA_PROGRAM]
    for c in creates:
        assert [flags[i] for i in c] == list(ATA_CREATE_FLAGS)
    made = [(keys[c[3]], keys[c[5]]) for c in creates]
    _, accts = route(msg)
    if name.startswith("buy"):
        assert made == [(SOL_MINT, SPL_TOKEN_PROGRAM), (token_mint(name), keys[accts[0]])]
        assert keys[creates[1][1]] == keys[accts[3]]
    else:
        assert made == [(SOL_MINT, SPL_TOKEN_PROGRAM)]
        assert keys[accts[0]] == SPL_TOKEN_PROGRAM


@pytest.mark.parametrize("name", BUILDS)
def test_evidence_amm_key_in_route_accounts(name):
    q, msg = case(name)
    keys, accts = route(msg)
    amm = {s["swapInfo"]["ammKey"] for s in q["routePlan"]}
    assert amm <= {keys[i] for i in accts[9:]}


@pytest.mark.parametrize("name", BUILDS)
def test_evidence_limitation_owner_accounts_in_amm_slots(name):
    """Why owner/ATA-in-AMM-slot is NOT rejected: real builds do it."""
    _, msg = case(name)
    keys, accts = route(msg)
    amm_slot_keys = {keys[i] for i in accts[9:]}
    tok = token_mint(name)
    owner_accounts = {OWNER, _ata(OWNER, SOL_MINT, SPL_TOKEN_PROGRAM),
                      _ata(OWNER, tok, SPL_TOKEN_PROGRAM), _ata(OWNER, tok, TOKEN_2022_PROGRAM)}
    assert OWNER in amm_slot_keys
    assert len(amm_slot_keys & owner_accounts) >= 3
    assert check(name) is None


# -- mutations of the real builds --------------------------------------------

def _dup_ata(which):
    def fn(msg, ixs):
        keys = resolve_account_keys(msg, tables())
        creates = [i for i, x in enumerate(ixs) if keys[x.program_id_index] == ATA_PROGRAM]
        ixs.insert(creates[which], ixs[creates[which]])
    return fn




@pytest.mark.parametrize("name", BUYS)
@pytest.mark.parametrize("which,label", [(0, "duplicate wSOL ATA create"),
                                         (1, "duplicate token ATA create")])
def test_duplicate_ata_create_rejected(name, which, label):
    assert label in check(name, mutate(name, _dup_ata(which)))


@pytest.mark.parametrize("name", SELLS)
def test_token_ata_create_in_sell_rejected(name):
    """Insert a create of the owner's token ATA (route slot 2) into a real sell,
    when the token mint is among the message keys (USDC, PYUSD builds)."""
    q, msg = case(name)
    keys = resolve_account_keys(msg, tables())
    mint = token_mint(name)
    if mint not in keys:
        pytest.skip("token mint not referenced by this build (covered synthetically)")
    ixs = list(msg.instructions)
    i = find(msg, ixs, ATA_PROGRAM)
    r = list(bytes(ixs[find(msg, ixs, JUPITER_V6_PROGRAM)].accounts))
    prog = next(p for p in (SPL_TOKEN_PROGRAM, TOKEN_2022_PROGRAM) if _ata(OWNER, mint, p) == keys[r[2]])
    a = bytearray(bytes(ixs[i].accounts))
    a[1], a[3], a[5] = r[2], keys.index(mint), keys.index(prog)
    ixs.insert(i + 1, ix(ixs[i].program_id_index, bytes(ixs[i].data), a))
    assert "token ATA create in a sell" in check(name, rebuild(msg, ixs))


def test_probe_token_ata_create_in_sell_rejected():
    ixs = base(False)
    assert "token ATA create in a sell" in synth([ata_create(TOK, SPL_TOKEN_PROGRAM)] + ixs, False)


@pytest.mark.parametrize("name", SELLS)
def test_sell_duplicate_wsol_create_rejected(name):
    assert "duplicate wSOL ATA create" in check(name, mutate(name, _dup_ata(0)))


@pytest.mark.parametrize("name", BUILDS)
def test_missing_wsol_ata_create_rejected(name):
    def drop(msg, ixs):
        del ixs[find(msg, ixs, ATA_PROGRAM)]
    err = check(name, mutate(name, drop))
    assert err is not None


def test_t22_buy_route_program_swapped_rejected():
    name = "buy_sol_pyusd_t22"
    def fn(msg, ixs):
        i = find(msg, ixs, JUPITER_V6_PROGRAM)
        a = bytearray(bytes(ixs[i].accounts)); a[0] = idx_of(msg, SPL_TOKEN_PROGRAM)
        ixs[i] = ix(ixs[i].program_id_index, bytes(ixs[i].data), a)
    assert check(name, mutate(name, fn)) is not None


def test_t22_sell_route_program_swapped_rejected():
    name = "sell_pyusd_t22_sol"
    def fn(msg, ixs):
        i = find(msg, ixs, JUPITER_V6_PROGRAM)
        a = bytearray(bytes(ixs[i].accounts)); a[0] = idx_of(msg, TOKEN_2022_PROGRAM)
        ixs[i] = ix(ixs[i].program_id_index, bytes(ixs[i].data), a)
    assert "classic" in check(name, mutate(name, fn))


def test_t22_buy_token_ata_create_under_classic_program_rejected():
    name = "buy_sol_pyusd_t22"
    def fn(msg, ixs):
        keys = resolve_account_keys(msg, tables())
        creates = [i for i, x in enumerate(ixs) if keys[x.program_id_index] == ATA_PROGRAM]
        a = bytearray(bytes(ixs[creates[1]].accounts)); a[5] = idx_of(msg, SPL_TOKEN_PROGRAM)
        ixs[creates[1]] = ix(ixs[creates[1]].program_id_index, bytes(ixs[creates[1]].data), a)
    assert check(name, mutate(name, fn)) is not None  # address/program mismatch


@pytest.mark.parametrize("name", BUILDS)
def test_quote_amm_key_not_in_route_rejected(name):
    q, msg = case(name)
    q = copy.deepcopy(q)
    q["routePlan"][0]["swapInfo"]["ammKey"] = str(Keypair().pubkey())
    err = swap_transaction_error(
        msg, resolve_account_keys(msg, tables()), owner=OWNER, input_mint=q["inputMint"],
        output_mint=q["outputMint"], amount=int(q["inAmount"]), quote=q, slippage_bps=150,
        max_priority_fee_lamports=50_000)
    assert "ammKey not among" in err


@pytest.mark.parametrize("name", BUILDS)
@pytest.mark.parametrize("hdr", [
    (1, 1, None),     # owner becomes a READONLY signer
    (1, 0, 0),        # every static account writable (programs/mint writable)
])
def test_scrambled_header_flags_rejected(name, hdr):
    _, msg = case(name)
    h = msg.header
    n_sig, n_ro_s, n_ro_u = hdr
    new = MessageHeader(n_sig, n_ro_s, h.num_readonly_unsigned_accounts if n_ro_u is None else n_ro_u)
    assert "flags" in check(name, rebuild(msg, header=new))


@pytest.mark.parametrize("name", BUILDS)
def test_alt_writable_readonly_swapped_rejected(name):
    from solders.message import MessageAddressTableLookup, MessageV0
    _, msg = case(name)
    lookups = [MessageAddressTableLookup(l.account_key, bytes(l.readonly_indexes), bytes(l.writable_indexes))
               for l in msg.address_table_lookups]
    m = MessageV0(msg.header, list(msg.account_keys), msg.recent_blockhash,
                  list(msg.instructions), lookups)
    assert check(name, m) is not None


@pytest.mark.parametrize("name", BUILDS)
def test_header_inconsistent_with_keys_rejected(name):
    _, msg = case(name)
    h = msg.header
    bad = MessageHeader(h.num_required_signatures, h.num_readonly_signed_accounts, len(msg.account_keys))
    assert check(name, rebuild(msg, header=bad)) is not None


# -- probe cases on the synthetic factory (previously accepted) ---------------

KP = Keypair()
OWN = KP.pubkey()
O = str(OWN)
TOK = str(Keypair().pubkey())
AMT, OUT = 10_000_000, 123_456
pk = Pubkey.from_string


def quote(buy=True):
    i, o = (SOL_MINT, TOK) if buy else (TOK, SOL_MINT)
    return direct_route({"inputMint": i, "outputMint": o, "inAmount": str(AMT), "outAmount": str(OUT)})


def synth(ixs, buy=True):
    tx = VersionedTransaction.from_bytes(unsigned_tx_bytes(OWN, ixs))
    q = quote(buy)
    return swap_transaction_error(tx.message, resolve_account_keys(tx.message, {}), owner=O,
                                  input_mint=q["inputMint"], output_mint=q["outputMint"],
                                  amount=AMT, quote=q, slippage_bps=150,
                                  max_priority_fee_lamports=50_000)


def base(buy=True):
    q = quote(buy)
    return conformant_instructions(OWN, q["inputMint"], q["outputMint"], AMT, OUT)


def ata_create(mint, prog):
    return Instruction(pk(ATA_PROGRAM), b"\x01", [
        AccountMeta(OWN, True, True), AccountMeta(pk(_ata(O, mint, prog)), False, True),
        AccountMeta(OWN, False, False), AccountMeta(pk(mint), False, False),
        AccountMeta(pk(SYSTEM_PROGRAM), False, False), AccountMeta(pk(prog), False, False)])


def route_idx(ixs):
    return next(k for k, x in enumerate(ixs) if str(x.program_id) == JUPITER_V6_PROGRAM)


@pytest.mark.parametrize("buy", [True, False])
def test_synthetic_baseline_passes(buy):
    assert synth(base(buy), buy) is None


def test_probe_repeated_ata_creates_both_programs_rejected():
    ixs = base()
    extra = [ata_create(TOK, SPL_TOKEN_PROGRAM), ata_create(TOK, TOKEN_2022_PROGRAM)]
    assert synth(extra * 3 + ixs) is not None
    r = route_idx(ixs)
    assert "duplicate token ATA create" in synth(ixs[:r] + [ata_create(TOK, SPL_TOKEN_PROGRAM)] + ixs[r:])
    assert synth(ixs[:r] + [ata_create(TOK, TOKEN_2022_PROGRAM)] + ixs[r:]) is not None
    assert "duplicate wSOL ATA create" in synth([ata_create(SOL_MINT, SPL_TOKEN_PROGRAM)] + ixs)


def test_probe_token_ata_under_other_program_than_route_rejected():
    ixs = base()
    tok_create = next(k for k, x in enumerate(ixs) if str(x.program_id) == ATA_PROGRAM
                      and str(x.accounts[3].pubkey) == TOK)
    ixs[tok_create] = ata_create(TOK, TOKEN_2022_PROGRAM)
    assert synth(ixs) is not None


def test_probe_wsol_ata_under_token_2022_rejected():
    ixs = base(False)
    ixs[2] = ata_create(SOL_MINT, TOKEN_2022_PROGRAM)
    assert synth(ixs, False) is not None


def test_probe_token_ata_create_after_route_rejected():
    ixs = base()
    tok_create = next(k for k, x in enumerate(ixs) if str(x.program_id) == ATA_PROGRAM
                      and str(x.accounts[3].pubkey) == TOK)
    c = ixs.pop(tok_create)
    ixs.insert(route_idx(ixs) + 1, c)
    assert synth(ixs) is not None


def test_probe_route_mint_slot_made_writable_rejected():
    ixs = base()
    r = route_idx(ixs); x = ixs[r]; a = list(x.accounts)
    a[5] = AccountMeta(a[5].pubkey, False, True)
    ixs[r] = Instruction(x.program_id, x.data, a)
    assert "flags" in synth(ixs)


def test_probe_extra_signer_rejected():
    ixs = base()
    r = route_idx(ixs); x = ixs[r]
    ixs[r] = Instruction(x.program_id, x.data, list(x.accounts) + [AccountMeta(Keypair().pubkey(), True, True)])
    assert synth(ixs) is not None


def test_probe_amm_key_missing_from_synthetic_route_rejected():
    ixs = base()
    r = route_idx(ixs); x = ixs[r]
    ixs[r] = Instruction(x.program_id, x.data, list(x.accounts)[:9] + [AccountMeta(Keypair().pubkey(), False, True)])
    assert "ammKey" in synth(ixs)


def test_probe_owner_and_atas_in_amm_slots_still_accepted_limitation():
    """Documented limitation (matches every real build): owner/wSOL ATA/token
    ATA in AMM slots are allowed. An arbitrary extra writable account is also
    allowed there (AMM accounts are not individually verifiable offline)."""
    ixs = base()
    r = route_idx(ixs); x = ixs[r]
    extra = [AccountMeta(OWN, True, True), AccountMeta(pk(_ata(O, TOK, SPL_TOKEN_PROGRAM)), False, True),
             AccountMeta(pk(_ata(O, SOL_MINT, SPL_TOKEN_PROGRAM)), False, True)]
    ixs[r] = Instruction(x.program_id, x.data, list(x.accounts) + extra)
    assert synth(ixs) is None
    assert FAKE_AMM in {str(m.pubkey) for m in ixs[r].accounts[9:]}
