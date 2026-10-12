"""Direct-route policy: quote gate, capped priority fee, pre-sign validator.

Fixture provenance: ``tests/fixtures/jupiter_dryrun/builds.json`` holds REAL
Jupiter Metis ``/swap/v1`` quote + swap builds (lite-api.jup.ag,
``onlyDirectRoutes=true``, ``priorityLevelWithMaxLamports`` medium / 50,000)
for a throwaway, UNFUNDED public key whose secret was discarded: nothing was
ever signed or sent. The unsigned transaction bytes are exactly as returned;
the address lookup tables were fetched read-only at the same time. Its
sha256 is pinned below. Mutations of it in this file are SYNTHETIC.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import pathlib

import pytest
from solders.instruction import CompiledInstruction
from solders.keypair import Keypair
from solders.message import MessageHeader, MessageV0
from solders.transaction import VersionedTransaction

import bot.config as config_module
from bot.config import Config, load_config
from bot.jupiter import PRIORITY_LEVEL, JupiterClient, SwapExecutor, TransactionStatus
from bot.swap_policy import (
    ATA_PROGRAM, COMPUTE_BUDGET_PROGRAM, JUPITER_V6_PROGRAM, SOL_MINT, SYSTEM_PROGRAM,
    resolve_account_keys, route_quote_error, swap_transaction_error,
)
from tests.swap_tx_factory import conformant_instructions, direct_route, unsigned_tx_bytes

pytestmark = pytest.mark.real_swap_policy

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "jupiter_dryrun" / "builds.json"
FIXTURE_SHA256 = "2a18bc72ec6b47cea8e7b5e9a20096bcc566df27b671b0c6a7469505b229de47"
DATA = json.loads(FIXTURE.read_text())
OWNER = DATA["user_pubkey"]
CASES = sorted(DATA["cases"])
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
CAP = 50_000
RPC = "https://rpc.test"
SWAP_URL = "https://jupiter.test/swap"


def test_fixture_is_pinned():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def tables():
    from solders.address_lookup_table_account import AddressLookupTable
    return {k: [str(a) for a in AddressLookupTable.deserialize(base64.b64decode(v["data_base64"])).addresses]
            for k, v in DATA["lookup_tables"].items()}


def case(name):
    c = DATA["cases"][name]
    q = c["quote"]
    tx = VersionedTransaction.from_bytes(base64.b64decode(c["swap_response"]["swapTransaction"]))
    return q, tx.message


def check(name, message=None, *, keys=None, owner=OWNER, amount=None, slippage=150, cap=CAP):
    q, msg = case(name)
    msg = message if message is not None else msg
    keys = keys if keys is not None else resolve_account_keys(msg, tables())
    return swap_transaction_error(
        msg, keys, owner=owner, input_mint=q["inputMint"], output_mint=q["outputMint"],
        amount=int(q["inAmount"]) if amount is None else amount, quote=q,
        slippage_bps=slippage, max_priority_fee_lamports=cap)


# -- quote gate ------------------------------------------------------------------------

@pytest.mark.parametrize("name", CASES)
def test_real_dry_run_quotes_pass(name):
    q, _ = case(name)
    assert route_quote_error(q, q["inputMint"], q["outputMint"], int(q["inAmount"]), 150) is None


def _step(q):
    return q["routePlan"][0]


QUOTE_MUTATIONS = {
    "routePlan missing": lambda q: q.pop("routePlan"),
    "routePlan null": lambda q: q.update(routePlan=None),
    "routePlan dict": lambda q: q.update(routePlan=_step(q)),
    "routePlan empty": lambda q: q.update(routePlan=[]),
    "two steps (split)": lambda q: q.update(routePlan=[_step(q), _step(q)]),
    "step not a dict": lambda q: q.update(routePlan=["x"]),
    "swapInfo missing": lambda q: _step(q).pop("swapInfo"),
    "step via intermediate mint": lambda q: _step(q)["swapInfo"].update(outputMint=USDC + "x"),
    "step input mint differs": lambda q: _step(q)["swapInfo"].update(inputMint=USDC),
    "step ammKey missing": lambda q: _step(q)["swapInfo"].pop("ammKey"),
    "step inAmount differs": lambda q: _step(q)["swapInfo"].update(inAmount="1"),
    "step outAmount differs": lambda q: _step(q)["swapInfo"].update(outAmount="1"),
    "percent 50": lambda q: _step(q).update(percent=50),
    "percent and bps both missing": lambda q: _step(q).update(percent=None, bps=None),
    "percent bool": lambda q: _step(q).update(percent=True, bps=None),
    "percent string": lambda q: _step(q).update(percent="100"),
    "percent float": lambda q: _step(q).update(percent=100.0),
    "bps 5000": lambda q: _step(q).update(percent=None, bps=5000),
    "percent 100 conflicts with bps 5000": lambda q: _step(q).update(percent=100, bps=5000),
    "bps bool": lambda q: _step(q).update(percent=None, bps=True),
    "platformFee nonzero": lambda q: q.update(platformFee={"amount": "10", "feeBps": 20}),
    "platformFee zero amount but feeBps": lambda q: q.update(platformFee={"amount": "0", "feeBps": 5}),
    "platformFee feeBps bool": lambda q: q.update(platformFee={"amount": "0", "feeBps": False}),
    "platformFee not an object": lambda q: q.update(platformFee="0"),
    "platformFee extra key": lambda q: q.update(platformFee={"amount": "0", "feeBps": 0, "feeAccount": "x"}),
    "platformFee amount int": lambda q: q.update(platformFee={"amount": 0, "feeBps": 0}),
    "slippage mismatch": lambda q: q.update(slippageBps=300),
    "slippage bool": lambda q: q.update(slippageBps=True),
    "slippage missing": lambda q: q.pop("slippageBps"),
    "inAmount mismatch": lambda q: q.update(inAmount="1"),
    "inAmount int": lambda q: q.update(inAmount=int(q["inAmount"])),
    "outputMint mismatch": lambda q: q.update(outputMint=USDC + "x"),
}


@pytest.mark.parametrize("mutation", sorted(QUOTE_MUTATIONS))
def test_quote_mutations_rejected(mutation):
    q = copy.deepcopy(case("buy_sol_usdc")[0])
    QUOTE_MUTATIONS[mutation](q)
    assert route_quote_error(q, SOL_MINT, USDC, 10_000_000, 150) is not None


@pytest.mark.parametrize("alloc", [
    {"percent": 100, "bps": None}, {"percent": None, "bps": 10_000},
    {"percent": 100, "bps": 10_000}, {"bps": 10_000}])
def test_full_allocation_forms_accepted(alloc):
    q = copy.deepcopy(case("buy_sol_usdc")[0])
    _step(q).pop("percent"); _step(q).pop("bps")
    _step(q).update(alloc)
    assert route_quote_error(q, SOL_MINT, USDC, 10_000_000, 150) is None


@pytest.mark.parametrize("fee", [None, {"amount": "0", "feeBps": 0}, {"amount": "0"}, {}])
def test_zero_platform_fee_forms_accepted(fee):
    q = copy.deepcopy(case("buy_sol_usdc")[0])
    q["platformFee"] = fee
    assert route_quote_error(q, SOL_MINT, USDC, 10_000_000, 150) is None


def test_token_to_token_rejected():
    q = copy.deepcopy(case("buy_sol_usdc")[0])
    q["inputMint"] = _step(q)["swapInfo"]["inputMint"] = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    assert "SOL" in route_quote_error(q, q["inputMint"], USDC, 10_000_000, 150)


def test_route_quote_error_never_raises():
    assert route_quote_error(object(), SOL_MINT, USDC, 1, 150) is not None


# -- outgoing requests -------------------------------------------------------------

class CaptureSession:
    def __init__(self, payload=None):
        self.gets, self.posts, self.payload = [], [], payload

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets.append((url, dict(params)))
        return Resp(self.payload)

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json))
        return Resp(self.payload)


class Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_quote_request_is_direct_only():
    s = CaptureSession({"outAmount": "1"})
    JupiterClient(Config(), s).quote(SOL_MINT, USDC, 123)
    ((url, params),) = s.gets
    assert url.endswith("/quote")
    assert params == {"inputMint": SOL_MINT, "outputMint": USDC, "amount": "123",
                      "slippageBps": "150", "onlyDirectRoutes": "true"}


def test_swap_request_body_is_capped_and_has_no_auto_tip_or_fee_account():
    q = case("buy_sol_usdc")[0]
    body = SwapExecutor(Config()).swap_request_body(q, OWNER)
    assert body == {
        "quoteResponse": q, "userPublicKey": OWNER, "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "prioritizationFeeLamports": {"priorityLevelWithMaxLamports": {
            "priorityLevel": "medium", "maxLamports": 50_000}},
    }
    assert PRIORITY_LEVEL in ("medium", "high", "veryHigh")   # documented enum
    text = json.dumps({k: v for k, v in body.items() if k != "quoteResponse"})
    for banned in ('"auto"', "jito", "feeAccount", "platformFee", "referral", "trackingAccount"):
        assert banned.lower() not in text.lower()


def test_priority_fee_cap_config(monkeypatch):
    monkeypatch.delenv("PRIORITY_FEE_MAX_LAMPORTS", raising=False)
    assert load_config().priority_fee_max_lamports == 50_000
    monkeypatch.setenv("PRIORITY_FEE_MAX_LAMPORTS", "20000")
    assert load_config().priority_fee_max_lamports == 20_000
    for bad in ("0", "-1", str(config_module.PRIORITY_FEE_HARD_CEILING_LAMPORTS + 1)):
        monkeypatch.setenv("PRIORITY_FEE_MAX_LAMPORTS", bad)
        with pytest.raises(ValueError):
            load_config()


# -- pre-sign validator on the REAL dry-run builds ------------------------------------

@pytest.mark.parametrize("name", CASES)
def test_real_dry_run_builds_pass(name):
    assert check(name) is None


@pytest.mark.parametrize("name", CASES)
def test_real_builds_respect_requested_fee_cap(name):
    """Jupiter honoured maxLamports (49,999 <= 50,000 in the observed builds);
    a lower local cap than what was built is refused."""
    assert DATA["cases"][name]["swap_response"]["prioritizationFeeLamports"] <= CAP
    fee = DATA["cases"][name]["swap_response"]["prioritizationFeeLamports"]
    if fee > 1:
        assert "exceeds cap" in check(name, cap=fee - 1)


def rebuild(msg, instructions=None, header=None):
    return MessageV0(header or msg.header, list(msg.account_keys), msg.recent_blockhash,
                     instructions if instructions is not None else list(msg.instructions),
                     list(msg.address_table_lookups))


def ix(program_idx, data, accounts):
    return CompiledInstruction(program_idx, bytes(data), bytes(accounts))


def idx_of(msg, key):
    keys = resolve_account_keys(msg, tables())
    return keys.index(key)


def mutate(name, fn):
    _, msg = case(name)
    ixs = list(msg.instructions)
    fn(msg, ixs)
    return rebuild(msg, ixs)


def find(msg, ixs, program, first=b""):
    keys = resolve_account_keys(msg, tables())
    return next(i for i, x in enumerate(ixs)
                if keys[x.program_id_index] == program and bytes(x.data).startswith(first))


def _extra_transfer(msg, ixs):
    t = ixs[find(msg, ixs, SYSTEM_PROGRAM)]
    route = ixs[find(msg, ixs, JUPITER_V6_PROGRAM)]
    other = bytes(route.accounts)[9]                      # an AMM account, not ours
    ixs.append(ix(t.program_id_index, bytes(t.data), [0, other]))


def _tip_to_amm(msg, ixs):
    i = find(msg, ixs, SYSTEM_PROGRAM)
    route = ixs[find(msg, ixs, JUPITER_V6_PROGRAM)]
    ixs[i] = ix(ixs[i].program_id_index, bytes(ixs[i].data), [0, bytes(route.accounts)[9]])


def _transfer_amount(msg, ixs):
    i = find(msg, ixs, SYSTEM_PROGRAM)
    d = bytearray(bytes(ixs[i].data)); d[4:12] = (10_000_001).to_bytes(8, "little")
    ixs[i] = ix(ixs[i].program_id_index, d, bytes(ixs[i].accounts))


def _route_patch(offset, value):
    def fn(msg, ixs):
        i = find(msg, ixs, JUPITER_V6_PROGRAM)
        d = bytearray(bytes(ixs[i].data))
        start = offset % len(d)
        assert len(d[start:start + len(value)]) == len(value)
        d[start:start + len(value)] = value
        ixs[i] = ix(ixs[i].program_id_index, d, bytes(ixs[i].accounts))
    return fn


def _route_account(pos, src_pos):
    def fn(msg, ixs):
        i = find(msg, ixs, JUPITER_V6_PROGRAM)
        a = bytearray(bytes(ixs[i].accounts)); a[pos] = a[src_pos]
        ixs[i] = ix(ixs[i].program_id_index, bytes(ixs[i].data), a)
    return fn


def _second_route(msg, ixs):
    ixs.insert(-1, ixs[find(msg, ixs, JUPITER_V6_PROGRAM)])


def _cu_price(price):
    def fn(msg, ixs):
        i = find(msg, ixs, COMPUTE_BUDGET_PROGRAM, b"\x03")
        ixs[i] = ix(ixs[i].program_id_index, b"\x03" + price.to_bytes(8, "little"), b"")
    return fn


def _dup_cu_price(msg, ixs):
    ixs.insert(0, ixs[find(msg, ixs, COMPUTE_BUDGET_PROGRAM, b"\x03")])


def _cu_other(msg, ixs):
    i = find(msg, ixs, COMPUTE_BUDGET_PROGRAM, b"\x02")
    ixs[i] = ix(ixs[i].program_id_index, b"\x01" + (256 * 1024).to_bytes(4, "little"), b"")


def _index_out_of_range(msg, ixs):
    i = find(msg, ixs, SYSTEM_PROGRAM)
    ixs[i] = ix(ixs[i].program_id_index, bytes(ixs[i].data), [0, 250])


def _unknown_program(msg, ixs):
    i = find(msg, ixs, SYSTEM_PROGRAM)
    route = ixs[find(msg, ixs, JUPITER_V6_PROGRAM)]
    # a static non-allowlisted key: whatever static key the route uses as an AMM program
    keys = [str(k) for k in msg.account_keys]
    allowed = {COMPUTE_BUDGET_PROGRAM, SYSTEM_PROGRAM, ATA_PROGRAM, JUPITER_V6_PROGRAM,
               "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"}
    other = next(j for j in range(1, len(keys)) if keys[j] not in allowed)
    ixs.insert(i, ix(other, b"\x00", []))


def _close_to_other(msg, ixs):
    i = find(msg, ixs, "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", b"\x09")
    route = ixs[find(msg, ixs, JUPITER_V6_PROGRAM)]
    a = bytearray(bytes(ixs[i].accounts)); a[1] = bytes(route.accounts)[9]
    ixs[i] = ix(ixs[i].program_id_index, bytes(ixs[i].data), a)


def _drop_close(msg, ixs):
    del ixs[find(msg, ixs, "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", b"\x09")]


def _drop_sync(msg, ixs):
    del ixs[find(msg, ixs, "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", b"\x11")]


def _ata_create_non_idempotent(msg, ixs):
    i = find(msg, ixs, ATA_PROGRAM)
    ixs[i] = ix(ixs[i].program_id_index, b"\x00", bytes(ixs[i].accounts))


def _ata_for_other_wallet(msg, ixs):
    i = find(msg, ixs, ATA_PROGRAM)
    route = ixs[find(msg, ixs, JUPITER_V6_PROGRAM)]
    a = bytearray(bytes(ixs[i].accounts)); a[2] = bytes(route.accounts)[9]
    ixs[i] = ix(ixs[i].program_id_index, bytes(ixs[i].data), a)


def _token_transfer(msg, ixs):
    i = find(msg, ixs, "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", b"\x11")
    ixs.insert(i + 1, ix(ixs[i].program_id_index, b"\x03" + (1).to_bytes(8, "little"),
                         [bytes(ixs[i].accounts)[0], 0, 0]))


TX_MUTATIONS = {
    "extra SOL transfer to a third party": _extra_transfer,
    "funding transfer redirected (tip-like)": _tip_to_amm,
    "funding amount differs": _transfer_amount,
    "second route instruction": _second_route,
    "route discriminator unknown": _route_patch(0, b"\x00" * 8),
    "route plan two steps": _route_patch(8, (2).to_bytes(4, "little")),
    "route platform fee bps": _route_patch(-1, b"\x05"),
    "route percent 50": _route_patch(-22, b"\x32"),
    "route slippage differs": _route_patch(-3, (300).to_bytes(2, "little")),
    "route quoted out differs": _route_patch(-11, (1).to_bytes(8, "little")),
    "route platform fee account set": _route_account(6, 9),
    "route destination override set": _route_account(4, 9),
    "route destination not owner's": _route_account(3, 9),
    "route authority not owner": _route_account(1, 9),
    "priority fee above cap": _cu_price(10_000_000),
    "duplicate compute price": _dup_cu_price,
    "compute budget heap frame": _cu_other,
    "account index out of range": _index_out_of_range,
    "unknown program": _unknown_program,
    "wSOL closed to a third party": _close_to_other,
    "wSOL never closed": _drop_close,
    "no syncNative before route": _drop_sync,
    "ATA create non-idempotent": _ata_create_non_idempotent,
    "ATA for another wallet": _ata_for_other_wallet,
    "SPL token transfer": _token_transfer,
}


@pytest.mark.parametrize("mutation", sorted(TX_MUTATIONS))
def test_tx_mutations_rejected(mutation):
    msg = mutate("buy_sol_bonk", TX_MUTATIONS[mutation])
    assert check("buy_sol_bonk", msg) is not None


@pytest.mark.parametrize("name", ["sell_bonk_sol", "sell_usdc_sol"])
def test_sell_with_sol_transfer_rejected(name):
    def add(msg, ixs):
        ixs.insert(2, ix(idx_of(msg, SYSTEM_PROGRAM), (2).to_bytes(4, "little") + (1).to_bytes(8, "little"),
                         [0, bytes(ixs[find(msg, ixs, JUPITER_V6_PROGRAM)].accounts)[3]]))
    assert "sell" in check(name, mutate(name, add))


def test_unresolved_lookup_table_rejected():
    q, msg = case("buy_sol_usdc")
    assert "unresolved" in swap_transaction_error(
        msg, None, owner=OWNER, input_mint=SOL_MINT, output_mint=USDC, amount=10_000_000,
        quote=q, slippage_bps=150, max_priority_fee_lamports=CAP)
    assert resolve_account_keys(msg, {}) is None


def test_wrong_owner_amount_and_signers_rejected():
    assert check("buy_sol_usdc", owner=str(Keypair().pubkey())) is not None
    assert check("buy_sol_usdc", amount=10_000_001) is not None
    assert check("buy_sol_usdc", slippage=100) is not None
    _, msg = case("buy_sol_usdc")
    h = msg.header
    two = rebuild(msg, header=MessageHeader(2, h.num_readonly_signed_accounts,
                                            h.num_readonly_unsigned_accounts))
    assert check("buy_sol_usdc", two) is not None


def test_validator_never_raises():
    assert swap_transaction_error(object(), [], owner=OWNER, input_mint=SOL_MINT, output_mint=USDC,
                                  amount=1, quote={}, slippage_bps=150,
                                  max_priority_fee_lamports=CAP) is not None


# -- executor lifecycle with the real policy -------------------------------------------

class World:
    """Fake Jupiter build + JSON-RPC; signing is real with a throwaway key."""

    def __init__(self, kp, build=None, alt_values=None):
        self.kp, self.build, self.alt_values = kp, build, alt_values
        self.swap_posts, self.rpc, self.sends = [], [], []

    def post(self, url, json=None, headers=None, timeout=None):
        if url == SWAP_URL:
            self.swap_posts.append(json)
            return Resp({"swapTransaction": base64.b64encode(self.build(json["quoteResponse"])).decode()})
        method, params = json["method"], json["params"]
        self.rpc.append(method)
        if method == "getMultipleAccounts":
            return Resp({"result": {"value": self.alt_values(params[0])}})
        if method == "sendTransaction":
            tx = VersionedTransaction.from_bytes(base64.b64decode(params[0]))
            assert tx.verify_with_results() == [True]
            self.sends.append(str(tx.signatures[0]))
            return Resp({"result": str(tx.signatures[0])})
        if method == "getSignatureStatuses":
            return Resp({"result": {"value": [None]}})
        raise AssertionError(method)


class FakeJup:
    swap_url = SWAP_URL

    def __init__(self, mutate_quote=None):
        self.quotes, self.mutate_quote = [], mutate_quote

    def _headers(self):
        return {}

    def quote(self, i, o, amount):
        self.quotes.append((i, o, amount))
        if (i, o, amount) == (SOL_MINT, USDC, 10**9):
            return {"inputMint": i, "outputMint": o, "inAmount": str(amount), "outAmount": "100000000"}
        q = direct_route({"inputMint": i, "outputMint": o, "inAmount": str(amount),
                          "outAmount": "777", "otherAmountThreshold": "700", "swapMode": "ExactIn"})
        if self.mutate_quote:
            self.mutate_quote(q)
        return q


def executor(world, jup):
    cfg = Config(live_trading=True, wallet_private_key="test-only-sentinel",
                 burner_wallet_pubkey=str(world.kp.pubkey()), rpc_url=RPC,
                 confirmation_timeout_seconds=0.0)
    ex = SwapExecutor(cfg, jup=jup, session=world)
    ex._load_keypair = lambda _K: world.kp
    return ex


def conformant_build(kp, extra=()):
    def build(q):
        return unsigned_tx_bytes(kp.pubkey(), conformant_instructions(
            kp.pubkey(), q["inputMint"], q["outputMint"], int(q["inAmount"]),
            int(q["outAmount"])) + list(extra))
    return build


def run_swap(world, jup):
    calls = []
    result = executor(world, jup).swap(
        SOL_MINT, USDC, 10_000_000, allocation_usd=1.0,
        before_broadcast=lambda sig: calls.append(sig) or True)
    return result, calls


def test_conformant_build_signs_and_broadcasts_once():
    kp = Keypair()
    w = World(kp, conformant_build(kp))
    result, calls = run_swap(w, FakeJup())
    assert len(calls) == 1 and w.sends == calls and result.tx_signature == calls[0]
    (body,) = w.swap_posts
    assert body["prioritizationFeeLamports"] == {"priorityLevelWithMaxLamports": {
        "priorityLevel": "medium", "maxLamports": 50_000}}


def test_quote_rejection_no_build_sign_or_send():
    kp = Keypair()
    w = World(kp, conformant_build(kp))
    result, calls = run_swap(w, FakeJup(lambda q: q.update(routePlan=q["routePlan"] * 2)))
    assert result.status == TransactionStatus.CONFIRMED_FAILURE and not result.ok
    assert result.tx_signature is None and "route policy" in result.error
    assert w.swap_posts == [] and calls == [] and w.sends == []


def test_presign_rejection_no_sign_or_send():
    from solders.system_program import TransferParams, transfer
    kp = Keypair()
    tip = transfer(TransferParams(from_pubkey=kp.pubkey(), to_pubkey=Keypair().pubkey(), lamports=1))
    w = World(kp, conformant_build(kp, [tip]))
    result, calls = run_swap(w, FakeJup())
    assert result.status == TransactionStatus.CONFIRMED_FAILURE and not result.ok
    assert result.tx_signature is None and "transaction policy" in result.error
    assert len(w.swap_posts) == 1 and calls == [] and w.sends == []
    assert "sendTransaction" not in w.rpc


def test_fee_over_cap_rejected_before_signing():
    kp = Keypair()

    def build(q):
        return unsigned_tx_bytes(kp.pubkey(), conformant_instructions(
            kp.pubkey(), q["inputMint"], q["outputMint"], int(q["inAmount"]),
            int(q["outAmount"]), cu_limit=1_400_000, cu_price=1_000_000))
    w = World(kp, build)
    result, calls = run_swap(w, FakeJup())
    assert "exceeds cap" in result.error and calls == [] and w.sends == []


def _real_alt_values(addrs):
    return [{"owner": DATA["lookup_tables"][a]["owner"], "data": [DATA["lookup_tables"][a]["data_base64"], "base64"]}
            if a in DATA["lookup_tables"] else None for a in addrs]


def test_executor_resolves_real_lookup_tables_read_only():
    """The production resolver (one getMultipleAccounts) + validator accept the
    real dry-run build; a missing or foreign-owned table fails closed."""
    _, msg = case("buy_sol_bonk")
    ex = executor(World(Keypair(), alt_values=_real_alt_values), FakeJup())
    keys = ex._resolve_account_keys(msg)
    assert keys == resolve_account_keys(msg, tables())
    assert ex.session.rpc == ["getMultipleAccounts"]
    assert executor(World(Keypair(), alt_values=lambda a: [None] * len(a)), FakeJup())._resolve_account_keys(msg) is None
    foreign = lambda a: [dict(v, owner=SYSTEM_PROGRAM) for v in _real_alt_values(a)]
    assert executor(World(Keypair(), alt_values=foreign), FakeJup())._resolve_account_keys(msg) is None


def test_unresolvable_lookup_table_refuses_before_signing():
    kp = Keypair()
    real_raw = base64.b64decode(DATA["cases"]["buy_sol_bonk"]["swap_response"]["swapTransaction"])
    w = World(kp, lambda q: real_raw, alt_values=lambda a: [None] * len(a))
    ex = executor(w, FakeJup())
    q = case("buy_sol_bonk")[0]
    jup = FakeJup()
    jup.quote = lambda i, o, amount: q if (i, o) != (SOL_MINT, USDC) else {
        "inputMint": i, "outputMint": o, "inAmount": str(amount), "outAmount": "100000000"}
    ex.jup = jup
    calls = []
    result = ex.swap(SOL_MINT, q["outputMint"], int(q["inAmount"]), allocation_usd=1.0,
                     before_broadcast=lambda s: calls.append(s) or True)
    assert "transaction policy" in result.error and "unresolved" in result.error
    assert calls == [] and w.sends == []
