"""P1 (buy path): durable buy intent + exact signature saved before broadcast,
position recorded exactly once from the verified fill.

Invariants (``TradingBot._live_entry`` / ``_resolve_buy_intents``):

* every armed swap needs the pre-broadcast callback (``requires_pre_broadcast_callback``
  is ``cfg.is_armed``); an armed buy without it is refused before quote/sign;
* a buy intent (mint, intended lamports, allocation, ``pending``) is durably
  saved BEFORE the entry swap is quoted or signed; the signed transaction's
  exact signature is durably saved on it BEFORE the broadcast (only
  ``save_state() is True`` allows it); a non-durable save sends nothing and
  never re-quotes / re-signs / resends;
* the position (``raw_tokens`` = ``raw_tokens_initial`` = verified raw fill)
  and the intent's ``settled`` status are written in ONE atomic save;
  ``confirmed_unrecorded`` is recovered exactly once from stored evidence;
* unknown outcomes are never resent; while any buy intent is open ALL new
  entries are blocked (policy) and its mint is frozen; resolution is by the
  saved signature (status + getTransaction/extract_fill) or blockhash expiry +
  none of the token in the wallet; a missing signature never proves non-execution.

Real here: ``TradingBot`` (``load_state``/``save_state``/``run``/``_tick``/
``_seek_entry`` incl. sizing, the pre-buy sell-route gate and the buy
lifecycle/resolver), ``SwapExecutor.swap`` -> ``_execute_live`` (SOL-exposure
gate, quote + direction validation, base64 decode, ``VersionedTransaction``,
the tx-version gate, REAL ed25519 signing with a throwaway keypair generated
in the test, signature derivation, ``send_raw_transaction``/``rpc_call``),
``_confirm_transaction``, ``extract_fill`` (incl. ``_native_sol_spent``),
``reconcile_wallet``, ``fetch_wallet_token_balances``. Fake: the HTTP session
(Jupiter build + JSON-RPC answered from the fixture and a programmable
wallet), Jupiter quotes, the price mark, the screener/strategy, the notifier.
No real wallet, key, RPC or network.

The fake RPC decodes every broadcast, verifies its signature against the
throwaway pubkey, and records whether that exact signature was ALREADY on an
open buy intent in the state file at the moment of broadcast.

Fixture provenance: ``tests/fixtures/synthetic_native_sol_buy.json`` is
SYNTHETIC (no real SOL -> token buy fixture exists in the repo). It is a
deterministic, hand-built jsonParsed ``getTransaction`` result in the shape of
the real native-SOL sell fixture: owner pays fee 15,000 + 2,039,280 rent for
the new token ATA; a temporary wSOL ATA is created, funded (system transfer +
syncNative), spent (13,333,333 lamports) and closed back to the owner; the
owner receives 4,200,000,000 raw (6 dp). It is NOT real on-chain data. Its
sha256 is pinned below.
"""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import types

import pytest
import requests
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

import bot.main as main_module
import bot.reconcile as reconcile_module
from bot.chain import SPL_TOKEN_PROGRAM, _native_delta, _native_sol_spent, extract_fill
from bot.config import Config
from bot.jupiter import SOL_MINT, USDC_MINT, SwapExecutor, TransactionStatus
from bot.state import save_state as REAL_SAVE
from tests.swap_tx_factory import conformant_instructions, direct_route, unsigned_tx_bytes
from tests.test_p1_state_durability import install as install_faulty_os

# Runs the REAL direct-route quote gate and pre-sign validator (see conftest).
pytestmark = pytest.mark.real_swap_policy

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "synthetic_native_sol_buy.json"
FIXTURE_SHA256 = "8d857dfaa4d0c86ab5019806dda03fbf7b1948b5bf72c1f69b4c85bbf3ca7bd6"
OWNER_PLACEHOLDER = "SynthOwner111111111111111111111111111111111"
MINT = "SynthBuyMint1111111111111111111111111111111"
MINT2 = "Synth2therMint11111111111111111111111111111"  # valid base58 (ATA derivation)
HELD = "SynthHeldMint111111111111111111111111111111"
RPC = "https://rpc.test"
SWAP_URL = "https://jupiter.test/swap"
SENTINEL = "test-only-sentinel"
SOL_IN, RAW_OUT, FEE, ATA_RENT = 13_333_333, 4_200_000_000, 15_000, 2_039_280
TOKENS = RAW_OUT / 10**6
PLANNED = 13_333_333          # $2 (bankroll 20 x 10%) at $150/SOL, floor-exact
EXPIRY = main_module.BUY_INTENT_EXPIRY_SECONDS
OPEN = main_module.OPEN_BUY_INTENT_STATUSES


class Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class BJup:
    """Quotes only: SOL/USD price, buy leg, sell-back leg; all ExactIn."""
    swap_url = SWAP_URL

    def __init__(self, sol_usd=150):
        self.quotes = []
        self.sol_usd = sol_usd

    def quote(self, i, o, amount):
        self.quotes.append((i, o, amount))
        if (i, o) == (SOL_MINT, USDC_MINT):
            out = self.sol_usd * 10**6 * amount // 10**9
            return {"inputMint": i, "outputMint": o, "inAmount": str(amount), "outAmount": str(out)}
        out = amount * 315 if i == SOL_MINT else amount // 315
        return direct_route({"inputMint": i, "outputMint": o, "inAmount": str(amount),
                             "outAmount": str(out), "otherAmountThreshold": str(out * 99 // 100),
                             "swapMode": "ExactIn"})

    def _headers(self):
        return {}

    def swap_quotes(self, mint=MINT):
        return [q for q in self.quotes if q == (SOL_MINT, mint, PLANNED)]


class Notes:
    enabled = True

    def __init__(self):
        self.buys, self.sells, self.sent = [], [], []

    def startup(self, *a):
        pass

    def buy(self, *a):
        self.buys.append(a)

    def sell(self, *a):
        self.sells.append(a)

    def halt(self, *a):
        pass

    def send(self, text):
        self.sent.append(text)


def load_fixture(owner, mint=MINT):
    text = FIXTURE.read_text().replace(OWNER_PLACEHOLDER, owner)
    if mint != MINT:
        text = text.replace(MINT, mint)
    return json.loads(text)["result"]


class BWorld:
    """Chain + RPC + Jupiter build boundary; signing is real."""

    def __init__(self, tmp_path, kp):
        self.kp = kp
        self.owner = str(kp.pubkey())
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.state = str(tmp_path / "state.json")
        self.lock = str(tmp_path / "lock")
        self.fill_tx = load_fixture(self.owner)
        self.txs, self.status, self.wallet = {}, {}, {}
        self.sends, self.durable_at_send, self.plan = [], [], []
        self.builds = 0
        self.extra_ixs = []          # appended to the conformant build (pre-sign policy tests)
        self.crash_build = False
        self.rpc = []
        self.marks, self.mark_calls = {}, []
        self.candidates = [MINT]
        self.now = 1_800_000_000.0

    def _open_disk_buy_signatures(self):
        try:
            with open(self.state) as f:
                state = json.load(f)
        except FileNotFoundError:
            return set()
        return {i.get("tx_signature") for i in state["portfolio"].get("buy_intents", [])
                if i.get("status") in OPEN and i.get("side") == "buy"}

    def land(self, sig, visible=True, mint=MINT):
        self.txs[sig] = self.fill_tx
        self.wallet[mint] = self.wallet.get(mint, 0.0) + TOKENS
        if visible:
            self.show(sig)

    def show(self, sig):
        self.status[sig] = {"confirmationStatus": "finalized", "err": None}

    def post(self, url, json=None, headers=None, timeout=None):
        if url == SWAP_URL:
            if self.crash_build:
                raise SystemExit("crash: died while building, before signing")
            self.builds += 1
            assert json["quoteResponse"]["inputMint"] == SOL_MINT
            q = json["quoteResponse"]
            # Policy-conformant SYNTHETIC build: the real pre-sign validator runs.
            unsigned = unsigned_tx_bytes(self.kp.pubkey(), conformant_instructions(
                self.kp.pubkey(), q["inputMint"], q["outputMint"],
                int(q["inAmount"]), int(q["outAmount"])) + list(self.extra_ixs))
            return Resp({"swapTransaction": base64.b64encode(unsigned).decode()})
        assert url == RPC, url
        method, params = json["method"], json["params"]
        self.rpc.append(method)
        if method == "sendTransaction":
            return self._send(params)
        if method == "getSignatureStatuses":
            return Resp({"result": {"value": [self.status.get(params[0][0])]}})
        if method == "getTransaction":
            sig = params[0]
            return Resp({"result": self.txs.get(sig) if sig in self.status else None})
        if method == "getTokenAccountsByOwner":
            assert params[0] == self.owner
            if params[1]["programId"] != SPL_TOKEN_PROGRAM:
                return Resp({"result": {"value": []}})
            return Resp({"result": {"value": [{"account": {"data": {"parsed": {"info": {
                "mint": m, "tokenAmount": {"uiAmount": ui, "amount": str(round(ui * 10**9)), "decimals": 9}}}}}} for m, ui in self.wallet.items() if ui]}})
        raise AssertionError(f"unexpected RPC {method}")

    def _send(self, params):
        tx = VersionedTransaction.from_bytes(base64.b64decode(params[0]))
        assert tx.verify_with_results() == [True]            # real, valid signature
        sig = str(tx.signatures[0])
        self.durable_at_send.append(sig in self._open_disk_buy_signatures())
        self.sends.append(sig)
        mode = self.plan.pop(0) if self.plan else "land"
        if mode == "land":
            self.land(sig)
        elif mode == "land_hidden":
            self.land(sig, visible=False)
        elif mode == "drop":
            pass
        elif mode == "fail_onchain":
            self.status[sig] = {"confirmationStatus": "finalized", "err": {"InstructionError": [5, 1]}}
        elif mode == "timeout_landed":
            self.land(sig)
            raise requests.exceptions.Timeout("send timed out")
        elif mode == "timeout_dropped":
            raise requests.exceptions.Timeout("send timed out")
        elif mode == "lost_landed":
            self.land(sig, visible=False)
            raise requests.exceptions.ConnectionError("connection reset; response lost")
        elif mode == "crash_landed":
            self.land(sig)
            raise SystemExit("crash during broadcast (it landed)")
        elif mode == "crash_dropped":
            raise SystemExit("crash during broadcast (it never left)")
        else:
            raise AssertionError(mode)
        return Resp({"result": sig})


def patch_world(monkeypatch, w):
    monkeypatch.setattr(reconcile_module, "get_wallet_pubkey", lambda cfg: w.owner)
    monkeypatch.setattr(main_module, "_price_usd",
                        lambda jup, mint, dec: w.mark_calls.append(mint) or w.marks.get(mint))
    monkeypatch.setattr(main_module, "time", types.SimpleNamespace(
        time=lambda: w.now, sleep=lambda s: None))
    monkeypatch.setattr(main_module, "save_state", REAL_SAVE)


@pytest.fixture
def bw(monkeypatch, tmp_path):
    w = BWorld(tmp_path, Keypair())                          # throwaway, test-generated
    patch_world(monkeypatch, w)
    return w


def boot(w, *, armed=True, jup=None):
    cfg = Config(live_trading=armed, wallet_private_key=SENTINEL if armed else "",
                 burner_wallet_pubkey=w.owner, rpc_url=RPC, bankroll_usd=20.0,
                 max_position_pct=10.0, confirmation_timeout_seconds=0.0,
                 confirmation_commitment="confirmed", poll_interval_seconds=0,
                 state_file=w.state, process_lock_file=w.lock)
    bot = main_module.TradingBot(cfg)
    bot.session = w
    bot.jup = jup or BJup()
    bot.executor = SwapExecutor(cfg, jup=bot.jup, session=w)
    bot.executor._load_keypair = lambda _Keypair: w.kp
    bot.notifier = Notes()
    bot.entry_scans = []
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: bot.entry_scans.append(1) or [
        types.SimpleNamespace(mint=m) for m in w.candidates])
    bot.screener = types.SimpleNamespace(screen=lambda mint, snapshot=None: types.SimpleNamespace(
        passed=True, price_usd=0.0005, symbol="SYN", decimals=6))
    bot._candidate_market_snapshot = lambda _mint: None
    return bot


def run(bot, ticks):
    done = []
    real = bot._tick

    def tick():
        real()
        done.append(1)
        if len(done) >= ticks:
            bot._running = False

    bot._tick = tick
    bot.run()
    return len(done)


def start(w, **kw):
    bot = boot(w, **kw)
    assert bot._save() is True
    return bot


def persisted(w):
    with open(w.state) as f:
        return json.load(f)


def buys_of(state):
    return state["portfolio"].get("buy_intents", [])


def position(state, mint=MINT):
    return next((p for p in state["portfolio"]["positions"] if p["mint"] == mint), None)


def recorded_once(w, sig):
    """Exactly-once, read from disk: one settled buy intent with this
    signature, one position with the exact raw fill, nothing open."""
    state = persisted(w)
    buys = buys_of(state)
    settled = [i for i in buys if i["status"] == "settled"]
    assert len(settled) == 1, buys
    (s,) = settled
    assert s["tx_signature"] == sig and w.sends.count(sig) == 1
    assert (s["lamports"], s["actual_in_amount"], s["actual_out_amount"]) == (PLANNED, SOL_IN, RAW_OUT)
    assert (s["fee_lamports"], s["fill_input_mint"], s["fill_output_mint"]) == (FEE, SOL_MINT, MINT)
    assert not [i for i in buys if i["status"] in OPEN]
    positions = state["portfolio"]["positions"]
    assert len(positions) == 1
    pos = position(state)
    assert pos["raw_tokens"] == pos["raw_tokens_initial"] == RAW_OUT
    assert pos["tokens"] == pytest.approx(TOKENS) and pos["token_decimals"] == 6
    assert pos["entry_signature"] == sig and pos["actual_in_amount"] == SOL_IN
    assert pos["size_usd"] == pytest.approx(2.0)
    assert state["portfolio"]["pending_intents"] == []          # sell intents untouched
    assert state["portfolio"]["realized_pnl"] == 0.0
    assert state["risk"]["realized_pnl_today"] == 0.0
    assert all(w.durable_at_send), "a signature was broadcast before it was durable"
    return state


def no_secrets(w):
    import base58
    text = open(w.state).read()
    secret = bytes(w.kp)
    assert SENTINEL not in text
    assert base58.b58encode(secret).decode() not in text
    assert base58.b58encode(secret[:32]).decode() not in text
    for intent in buys_of(json.loads(text)):
        assert not {"signed_tx", "raw_tx", "transaction", "keypair"} & set(intent)
        sig = intent.get("tx_signature")
        assert sig is None or (len(base58.b58decode(sig)) == 64 and sig != "1" * 64)


def crash_before_save(monkeypatch, k):
    n = [0]

    def save(*a, **kw):
        n[0] += 1
        if n[0] == k:
            raise SystemExit(f"crash before save {k}")
        return REAL_SAVE(*a, **kw)
    monkeypatch.setattr(main_module, "save_state", save)


def crash_after_save(monkeypatch, k):
    n = [0]

    def save(*a, **kw):
        ok = REAL_SAVE(*a, **kw)
        n[0] += 1
        if n[0] == k:
            raise SystemExit(f"crash after save {k}")
        return ok
    monkeypatch.setattr(main_module, "save_state", save)


def fail_save(monkeypatch, k):
    n = [0]

    def save(*a, **kw):
        n[0] += 1
        return False if n[0] == k else REAL_SAVE(*a, **kw)
    monkeypatch.setattr(main_module, "save_state", save)


def restore_save(monkeypatch):
    monkeypatch.setattr(main_module, "save_state", REAL_SAVE)


# -- fixture + extraction ------------------------------------------------------------

def test_fixture_is_pinned_and_labelled_synthetic():
    data = FIXTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FIXTURE_SHA256
    doc = json.loads(data)
    assert doc["_source"]["kind"] == "SYNTHETIC" and "NOT a real" in doc["_source"]["note"]


def test_buy_fill_is_exact_principal_not_fee_or_ata_rent():
    tx = load_fixture(OWNER_PLACEHOLDER)
    fill = extract_fill(tx, OWNER_PLACEHOLDER, SOL_MINT, MINT)
    assert (fill.actual_in_amount, fill.actual_out_amount) == (SOL_IN, RAW_OUT)
    assert (fill.in_decimals, fill.out_decimals, fill.fee_lamports) == (9, 6, FEE)
    assert _native_sol_spent(tx, OWNER_PLACEHOLDER) == SOL_IN
    # the old fee-adjusted lamport drop includes the new token ATA's rent: it
    # would exceed the intended lamports and make every first buy unverifiable
    assert _native_delta(tx, OWNER_PLACEHOLDER, FEE) == SOL_IN + ATA_RENT


def test_unexplained_owner_lamport_move_is_not_a_proven_principal():
    tx = load_fixture(OWNER_PLACEHOLDER)
    tx["transaction"]["message"]["instructions"].append({
        "program": "system", "programId": "11111111111111111111111111111111",
        "parsed": {"type": "transfer", "info": {
            "source": OWNER_PLACEHOLDER, "destination": "SomeoneElse", "lamports": 7}}})
    assert _native_sol_spent(tx, OWNER_PLACEHOLDER) is None
    tx = load_fixture(OWNER_PLACEHOLDER)
    tx["meta"]["postBalances"][0] -= 1               # balances no longer reconcile exactly
    assert _native_sol_spent(tx, OWNER_PLACEHOLDER) is None


# -- happy path, with the invariants observed at each save and at the broadcast -------

def test_intent_saved_before_quote_sign_and_signature_before_broadcast(bw, monkeypatch):
    bot = start(bw)
    writes = []

    def spy(*a, **kw):
        ok = REAL_SAVE(*a, **kw)
        writes.append((len(bot.jup.swap_quotes()), bw.builds, list(bw.sends),
                       [(i["status"], i.get("tx_signature")) for i in buys_of(persisted(bw))]))
        return ok
    monkeypatch.setattr(main_module, "save_state", spy)
    bot._tick()
    (sig,) = bw.sends
    # save 1: the intent, after the quote-only sell-route gate (1 buy-leg quote) and
    # BEFORE the swap's own quote, any build or signature
    assert writes[0] == (1, 0, [], [("pending", None)])
    # save 2: the exact signature, after signing and BEFORE the broadcast
    assert writes[1] == (2, 1, [], [("pending", sig)])
    assert bw.durable_at_send == [True]
    state = recorded_once(bw, sig)
    assert len(writes) == 4                       # + evidence, + position/settled (one write)
    assert writes[2][3] == [("confirmed_unrecorded", sig)]
    assert bot.notifier.buys and bot.portfolio.open_count == 1
    assert buys_of(state)[0]["side"] == "buy"
    no_secrets(bw)


def test_armed_buy_without_callback_refused(bw):
    ex = boot(bw).executor
    result = ex.swap(SOL_MINT, MINT, PLANNED, allocation_usd=2.0)
    assert result.status == TransactionStatus.CONFIRMED_FAILURE and result.tx_signature is None
    assert bw.builds == 0 and bw.sends == [] and ex.jup.swap_quotes() == []


# -- crashes / failures BEFORE the broadcast: nothing sent; recovery -----------------

def test_crash_before_intent_save_sends_nothing_then_buys_once(bw, monkeypatch):
    bot = start(bw)
    crash_before_save(monkeypatch, 1)
    with pytest.raises(SystemExit):
        bot._tick()
    restore_save(monkeypatch)
    assert bw.sends == [] and bw.builds == 0 and bot.jup.swap_quotes() == [(SOL_MINT, MINT, PLANNED)]
    assert buys_of(persisted(bw)) == []
    run(boot(bw), 2)
    (sig,) = bw.sends
    recorded_once(bw, sig)


@pytest.mark.parametrize("where", ["after_intent_save", "during_build"])
def test_crash_after_intent_before_sign_frozen_until_expiry_then_once(bw, monkeypatch, where):
    bot = start(bw)
    if where == "after_intent_save":
        crash_after_save(monkeypatch, 1)
    else:
        bw.crash_build = True
    with pytest.raises(SystemExit):
        bot._tick()
    restore_save(monkeypatch)
    bw.crash_build = False
    assert bw.sends == []
    assert [(i["status"], i["tx_signature"]) for i in buys_of(persisted(bw))] == [("pending", None)]
    # no signature is NOT evidence of non-execution: entries stay blocked until expiry
    bot2 = boot(bw)
    assert run(bot2, 3) == 3
    assert bw.sends == [] and bot2.entry_scans == [] and bw.builds == 0
    assert [i["status"] for i in buys_of(persisted(bw))] == ["pending"]
    bw.now += EXPIRY + 1                              # expired + none of the token held
    run(boot(bw), 3)
    (sig,) = bw.sends
    state = recorded_once(bw, sig)
    assert [i["status"] for i in buys_of(state)] == ["failed", "settled"]


def test_missing_signature_with_the_token_in_wallet_never_counts_as_not_executed(bw, monkeypatch):
    bot = start(bw)
    crash_after_save(monkeypatch, 1)
    with pytest.raises(SystemExit):
        bot._tick()
    restore_save(monkeypatch)
    bw.now += EXPIRY + 1
    bw.wallet[MINT] = TOKENS                          # the token is in the wallet: unexplained
    bot2 = boot(bw)
    assert run(bot2, 3) == 0                          # startup reconciliation blocks (untracked)
    (intent,) = buys_of(persisted(bw))
    assert intent["status"] == "pending" and "operator" in intent["resolution"]
    assert bw.sends == [] and persisted(bw)["portfolio"]["positions"] == []


@pytest.mark.parametrize("fault", ["return_false", "file_fsync", "dir_fsync"])
def test_signature_save_not_durable_no_broadcast_then_fresh_attempt(bw, monkeypatch, fault):
    bot = start(bw)
    if fault == "return_false":
        fail_save(monkeypatch, 2)
    else:
        install_faulty_os(monkeypatch, {2: fault}, path=bw.state)
    bot._tick()
    assert bw.sends == [] and bw.builds == 1 and len(bot.jup.swap_quotes()) == 2
    (intent,) = bot.portfolio.buy_intents
    old_sig = intent["tx_signature"]
    assert intent["status"] == "failed" and "not durably saved" in intent["error"]
    assert old_sig and len(old_sig) > 80               # the real derived signature, kept as audit
    assert bot.portfolio.positions == {}
    restore_save(monkeypatch)
    install_faulty_os(monkeypatch)
    bot._tick()                                        # normal lifecycle: a NEW intent
    (sig,) = bw.sends
    assert sig != old_sig and bw.builds == 2
    state = recorded_once(bw, sig)
    assert [(i["status"], i["tx_signature"]) for i in buys_of(state)] == [
        ("failed", old_sig), ("settled", sig)]


def test_signature_dir_fsync_crash_after_replace_resolves_by_expiry(bw, monkeypatch):
    """The signature write replaced the file (disk shows it) but the process
    died before the directory fsync. Nothing was broadcast; on restart the
    signature is looked up (unknown), entries stay blocked until expiry."""
    bot = start(bw)
    install_faulty_os(monkeypatch, {2: "dir_fsync"}, crash=True, path=bw.state)
    with pytest.raises(BaseException):
        bot._tick()
    install_faulty_os(monkeypatch)
    assert bw.sends == []
    (intent,) = buys_of(persisted(bw))
    disk_sig = intent["tx_signature"]
    assert intent["status"] == "pending" and disk_sig
    bot2 = boot(bw)
    run(bot2, 3)
    assert bw.sends == [] and bot2.entry_scans == [] and "getSignatureStatuses" in bw.rpc
    bw.now += EXPIRY + 1
    run(boot(bw), 3)
    (sig,) = bw.sends
    assert sig != disk_sig
    state = recorded_once(bw, sig)
    assert buys_of(state)[0]["status"] == "failed" and buys_of(state)[0]["tx_signature"] == disk_sig


def test_crash_after_signature_persisted_before_broadcast_expires_then_once(bw, monkeypatch):
    bot = start(bw)
    crash_after_save(monkeypatch, 2)
    with pytest.raises(SystemExit):
        bot._tick()
    restore_save(monkeypatch)
    assert bw.sends == []
    (intent,) = buys_of(persisted(bw))
    never_sent = intent["tx_signature"]
    assert intent["status"] == "pending" and never_sent
    bot2 = boot(bw)
    run(bot2, 3)                                       # unknown on chain, not expired: blocked
    assert bw.sends == [] and bot2.entry_scans == [] and bw.builds == 1
    assert persisted(bw)["portfolio"]["positions"] == []
    bw.now += EXPIRY + 1
    run(boot(bw), 3)
    (sig,) = bw.sends
    assert sig != never_sent
    state = recorded_once(bw, sig)
    assert [(i["status"], i["tx_signature"]) for i in buys_of(state)] == [
        ("failed", never_sent), ("settled", sig)]


# -- during / after the broadcast: never a resend ------------------------------------

def test_crash_during_broadcast_landed_recorded_once_no_resend(bw):
    bot = start(bw)
    bw.plan = ["crash_landed"]
    with pytest.raises(SystemExit):
        bot._tick()
    (sig,) = bw.sends
    (intent,) = buys_of(persisted(bw))
    assert (intent["status"], intent["tx_signature"]) == ("pending", sig)
    bot2 = boot(bw)
    assert run(bot2, 3) == 3                           # resolved BEFORE reconciliation
    assert bw.sends == [sig] and len(bot2.notifier.buys) == 1
    recorded_once(bw, sig)
    run(boot(bw), 2)
    assert bw.sends == [sig]
    recorded_once(bw, sig)


def test_crash_during_broadcast_never_left_no_resend_until_expiry(bw):
    bot = start(bw)
    bw.plan = ["crash_dropped"]
    with pytest.raises(SystemExit):
        bot._tick()
    (first,) = bw.sends
    run(boot(bw), 3)
    assert bw.sends == [first]
    bw.now += EXPIRY + 1
    run(boot(bw), 3)
    assert len(bw.sends) == 2 and bw.sends[0] == first
    state = recorded_once(bw, bw.sends[1])
    assert [i["status"] for i in buys_of(state)] == ["failed", "settled"]


@pytest.mark.parametrize("mode", ["timeout_landed", "lost_landed", "land_hidden"])
def test_unknown_outcome_that_landed_recorded_once_never_resent(bw, mode):
    bot = start(bw)
    bw.plan = [mode]
    bot._tick()
    (sig,) = bw.sends
    (intent,) = bot.portfolio.buy_intents
    assert (intent["status"], intent["tx_signature"]) == ("unresolved", sig)
    assert bot.portfolio.positions == {}
    scans = len(bot.entry_scans)
    if mode != "timeout_landed":
        for _ in range(2):                             # not visible yet: frozen, no resend
            bot._tick()
        assert bw.sends == [sig] and len(bot.entry_scans) == scans
        bw.show(sig)
    bot._tick()
    recorded_once(bw, sig)
    bot._tick()
    assert bw.sends == [sig] and len(bot.notifier.buys) == 1


@pytest.mark.parametrize("mode", ["timeout_dropped", "drop"])
def test_timeout_that_never_landed_no_resend_until_expiry(bw, mode):
    bot = start(bw)
    bw.plan = [mode]
    bot._tick()
    (first,) = bw.sends
    scans = len(bot.entry_scans)
    for _ in range(3):
        bot._tick()
    assert bw.sends == [first] and bw.builds == 1 and len(bot.entry_scans) == scans
    bw.now += EXPIRY + 1
    bot._tick()                                        # failed (expired, wallet empty) -> retry
    assert len(bw.sends) == 2
    state = recorded_once(bw, bw.sends[1])
    assert [(i["status"], i["tx_signature"]) for i in buys_of(state)][0] == ("failed", first)


def test_failed_on_chain_is_definitive_no_position_then_normal_retry(bw):
    bot = start(bw)
    bw.plan = ["fail_onchain"]
    bot._tick()
    (first,) = bw.sends
    (intent,) = bot.portfolio.buy_intents
    assert intent["status"] == "failed" and intent["tx_signature"] == first
    assert bot.portfolio.positions == {}
    bot._tick()
    state = recorded_once(bw, bw.sends[1])
    assert len(bw.sends) == 2


# -- entry-blocking policy -----------------------------------------------------------

def test_unresolved_buy_blocks_all_entries_freezes_mint_and_keeps_managing_positions(bw):
    bot = start(bw)
    bot.portfolio.open(HELD, "HLD", 1.0, 1.0, 1.0, token_decimals=6, entry_signature="e",
                       actual_in_amount=1, actual_out_amount=10**6, raw_tokens=10**6,
                       raw_tokens_initial=10**6)
    bw.wallet[HELD] = 1.0
    bw.candidates = [MINT, MINT2]
    bw.plan = ["drop"]
    bot._tick()
    (sig,) = bw.sends
    assert bot._buy_frozen(MINT) and not bot._buy_frozen(MINT2)
    quotes, scans, marks = len(bot.jup.quotes), len(bot.entry_scans), bw.mark_calls.count(HELD)
    for _ in range(3):
        bot._tick()
    # no new entry of ANY mint (not even MINT2): no scan, quote, build or send
    assert len(bot.entry_scans) == scans and len(bot.jup.quotes) == quotes
    assert bw.builds == 1 and bw.sends == [sig]
    assert bw.mark_calls.count(HELD) == marks + 3      # exits of other positions still managed
    assert set(bot.portfolio.positions) == {HELD}


def test_frozen_mint_is_never_bought_twice_even_if_the_entry_search_runs(bw):
    """Defence in depth below the all-entries gate: a direct entry search skips
    the frozen mint and buys another (one new intent, for MINT2 only)."""
    bot = start(bw)
    bw.plan = ["drop", "drop"]
    bot._tick()
    (first,) = bw.sends
    bw.candidates = [MINT, MINT2]
    bot._seek_entry()
    assert bot.jup.swap_quotes(MINT) == [(SOL_MINT, MINT, PLANNED)] * 2   # gate + first swap only
    assert [i["mint"] for i in bot.portfolio.buy_intents] == [MINT, MINT2]
    assert len(bw.sends) == 2


# -- fill verification ---------------------------------------------------------------

def test_fill_spending_more_than_intended_is_not_recorded(bw):
    jup = BJup(sol_usd=160)                            # plans 12,500,000 < the fill's 13,333,333
    bot = start(bw, jup=jup)
    bot._tick()
    (sig,) = bw.sends
    (intent,) = bot.portfolio.buy_intents
    assert intent["status"] == "unresolved" and "exceeds the intended" in intent["resolution"]
    assert bot.portfolio.positions == {}
    bot._tick()                                        # resolver: confirmed but does not verify
    assert bot.portfolio.positions == {} and bw.sends == [sig]
    assert "operator" in bot.portfolio.buy_intents[0]["resolution"]


def test_fill_of_another_mint_is_not_recorded(bw):
    bw.fill_tx = load_fixture(bw.owner, mint=MINT2)
    bot = start(bw)
    bot._tick()
    (intent,) = bot.portfolio.buy_intents
    assert intent["status"] == "unresolved" and bot.portfolio.positions == {}


# -- position-recording save fails / crashes: recovered exactly once ------------------

def test_position_recording_save_fails_recovered_once_from_stored_evidence(bw, monkeypatch):
    bot = start(bw)
    fail_save(monkeypatch, 4)
    bot._tick()
    (sig,) = bw.sends
    (intent,) = bot.portfolio.buy_intents
    assert intent["status"] == "confirmed_unrecorded" and bot.portfolio.positions == {}
    assert buys_of(persisted(bw))[0]["status"] == "confirmed_unrecorded"
    assert position(persisted(bw)) is None and bot.notifier.buys == []
    restore_save(monkeypatch)
    lookups = bw.rpc.count("getTransaction")
    bot._tick()
    assert bw.rpc.count("getTransaction") == lookups  # from stored evidence, no chain query
    recorded_once(bw, sig)
    bot._tick()
    recorded_once(bw, sig)


def test_crash_before_position_recording_recovered_once_at_startup(bw, monkeypatch):
    bot = start(bw)
    crash_before_save(monkeypatch, 4)
    with pytest.raises(SystemExit):
        bot._tick()
    restore_save(monkeypatch)
    (sig,) = bw.sends
    assert buys_of(persisted(bw))[0]["status"] == "confirmed_unrecorded"
    lookups = bw.rpc.count("getTransaction")
    run(boot(bw), 2)
    assert bw.rpc.count("getTransaction") == lookups
    recorded_once(bw, sig)


@pytest.mark.parametrize("crash", ["after", "before"])
def test_restart_after_every_save_boundary_exactly_one_broadcast_and_position(
        monkeypatch, tmp_path, crash):
    k = 1
    while True:
        w = BWorld(tmp_path / f"{crash}{k}", Keypair())
        patch_world(monkeypatch, w)
        bot = start(w)
        (crash_after_save if crash == "after" else crash_before_save)(monkeypatch, k)
        crashed = False
        try:
            bot._tick()
        except SystemExit:
            crashed = True
        restore_save(monkeypatch)
        if not crashed:
            break
        before = list(w.sends)
        run(boot(w), 3)                                # within the window: no blind resend
        if before:
            assert w.sends == before
        w.now += EXPIRY + 1
        run(boot(w), 3)
        assert len(w.sends) == 1, (crash, k, w.sends)   # exactly one broadcast in total
        state = recorded_once(w, w.sends[0])
        assert len(state["portfolio"]["positions"]) == 1
        run(boot(w), 2)
        assert len(w.sends) == 1
        recorded_once(w, w.sends[0])
        k += 1
    # saves: intent, signature (pre-broadcast), fill evidence, position + settled
    assert k - 1 == 4
    recorded_once(w, w.sends[0])
    no_secrets(w)


# -- unchanged: paper; legacy state --------------------------------------------------

def test_paper_entry_unchanged_no_intent_no_callback_no_send(bw):
    bot = start(bw, armed=False)
    seen = []
    real = bot.executor.swap
    bot.executor.swap = lambda *a, **kw: seen.append(kw) or real(*a, **kw)
    bot._tick()
    assert seen == [{"allocation_usd": pytest.approx(2.0)}]
    assert bot.portfolio.buy_intents == [] and MINT in bot.portfolio.positions
    pos = bot.portfolio.positions[MINT]
    assert pos.raw_tokens is None and pos.entry_price == 0.0005
    assert bw.builds == 0 and bw.sends == [] and "buy_intents" in persisted(bw)["portfolio"]


def test_legacy_state_without_buy_intents_loads_and_buys_once(bw):
    with open(bw.state, "w") as f:
        json.dump({"version": 1, "portfolio": {"realized_pnl": 0.0, "positions": [],
                                               "pending_intents": []},
                   "risk": {}}, f)
    bot = boot(bw)
    assert bot.portfolio.buy_intents == [] and bot._open_buy_intents() == []
    run(bot, 1)
    recorded_once(bw, bw.sends[0])


def test_open_buy_intent_halts_an_unarmed_session(bw):
    bot = start(bw)
    bw.plan = ["drop"]
    bot._tick()
    paper = boot(bw, armed=False)
    assert run(paper, 3) == 1 and paper._running is False   # needs an armed session
    assert bw.sends == [bw.sends[0]] and paper.portfolio.positions == {}


# -- direct-route policy rejections (real policy): definitive, nothing sent ---------

def test_quote_route_policy_rejection_never_builds_signs_or_sends(bw):
    """A multi-hop quote is refused before /swap: no build, no signature, no
    broadcast; the intent is definitively failed (never unresolved)."""
    class MultiHop(BJup):
        def quote(self, i, o, amount):
            q = super().quote(i, o, amount)
            if (i, o) == (SOL_MINT, MINT):
                q["routePlan"] = q["routePlan"] * 2
            return q
    bot = start(bw, jup=MultiHop())
    bot._tick()
    assert bw.builds == 0 and bw.sends == [] and "sendTransaction" not in bw.rpc
    (intent,) = buys_of(persisted(bw))
    assert intent["status"] == "failed" and intent["tx_signature"] is None
    assert intent["resolution"] == "refused before send (no transaction)"
    assert "route policy" in intent["error"]
    assert bot.portfolio.positions == {} and not bot._open_buy_intents()


def test_presign_policy_rejection_never_signs_or_sends(bw):
    """An extra SOL transfer in the built transaction is refused before
    signing: no signature persisted, nothing broadcast, intent failed."""
    from solders.system_program import TransferParams, transfer
    bw.extra_ixs = [transfer(TransferParams(
        from_pubkey=bw.kp.pubkey(), to_pubkey=Keypair().pubkey(), lamports=1_000))]
    bot = start(bw)
    bot._tick()
    assert bw.builds == 1 and bw.sends == [] and "sendTransaction" not in bw.rpc
    (intent,) = buys_of(persisted(bw))
    assert intent["status"] == "failed" and intent["tx_signature"] is None
    assert "signed_at" not in intent
    assert "transaction policy" in intent["error"]
    assert bot.portfolio.positions == {} and not bot._open_buy_intents()
