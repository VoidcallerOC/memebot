"""P1: the exact transaction signature is durably saved on the sell intent
BEFORE the transaction is broadcast.

Invariant: nothing is broadcast until the signed transaction's first
signature (base58) is on the right open intent in the state file and
``save_state`` returned True. A save that is not durable aborts the attempt:
no broadcast, no re-quote, no re-signing, no fallback send.

Real here: ``TradingBot`` (``load_state``/``save_state``/``run``/``_tick``/
exit management/``_settle_live_sell``/resolver), ``SwapExecutor.swap`` ->
``_execute_live`` (base64 decode, ``VersionedTransaction.from_bytes``, the
tx-version gate, REAL ed25519 signing with a throwaway keypair generated in
the test, signature derivation, ``chain.send_raw_transaction``/``rpc_call``),
confirmation and ``extract_fill`` on the pinned native-SOL fixture (its owner
rewritten to the throwaway pubkey). Fake: the HTTP session (Jupiter build +
JSON-RPC), the quote, the price mark, the notifier. No real wallet, no network.

The fake RPC decodes every broadcast, verifies its signature against the
throwaway pubkey, and checks that the state file ALREADY holds that exact
signature on an open intent at the moment of broadcast.
"""
from __future__ import annotations

import base64
import json
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
from bot.config import Config
from bot.jupiter import SOL_MINT, SwapExecutor
from bot.chain import TOKEN_2022_PROGRAM
from tests.test_p1_native_sol_settlement_e2e import (
    ENTRY, FEE, FIXTURE, FULL, IN_RAW, MINT, OUT_LAMPORTS, OWNER, RAW_DOUBLE, RPC, SENTINEL,
    SL_MARK, SL_PNL, SWAP_URL, TP_MARK, Jup, Notes, Resp, intents_of, persisted, position,
)
from tests.swap_tx_factory import conformant_unsigned_tx, direct_route
from tests.test_p1_state_durability import install as install_faulty_os

# Runs the REAL direct-route quote gate and pre-sign validator (see conftest).
pytestmark = pytest.mark.real_swap_policy


class DirectJup(Jup):
    """The shared fake's quotes as single-step 100% direct routes."""

    def quote(self, i, o, amount):
        q = super().quote(i, o, amount)
        return direct_route(q) if isinstance(q, dict) and "outAmount" in q else q

EXPIRY = main_module.SELL_INTENT_EXPIRY_SECONDS
TP50_PNL = FULL * (TP_MARK - ENTRY)


class RWorld:
    """Chain + RPC + Jupiter build boundary; signing is real."""

    def __init__(self, tmp_path, kp):
        self.kp = kp
        self.owner = str(kp.pubkey())
        self.state = str(tmp_path / "state.json")
        self.lock = str(tmp_path / "lock")
        self.fill_tx = json.loads(FIXTURE.read_text().replace(OWNER, self.owner))["result"]
        self.txs, self.status, self.wallet = {}, {}, {}
        self.sends = []               # signatures broadcast, decoded from the raw bytes
        self.durable_at_send = []     # was that exact signature on disk at broadcast?
        self.plan = []                # per broadcast: what the network does
        self.builds = 0
        self.crash_build = False
        self.rpc = []
        self.marks = {}
        self.now = 1_800_000_000.0

    def _open_disk_signatures(self):
        try:
            with open(self.state) as f:
                state = json.load(f)
        except FileNotFoundError:
            return set()
        return {i.get("tx_signature") for i in state["portfolio"]["pending_intents"]
                if i.get("status") in main_module.OPEN_SELL_INTENT_STATUSES}

    def land(self, sig, visible=True):
        self.txs[sig] = self.fill_tx
        self.wallet[MINT] = round(self.wallet[MINT] - FULL, 12)
        if visible:
            self.show(sig)

    def show(self, sig):
        self.status[sig] = {"confirmationStatus": "finalized", "err": None}

    def post(self, url, json=None, headers=None, timeout=None):
        if url == SWAP_URL:
            if self.crash_build:
                raise SystemExit("crash: died while building, before signing")
            self.builds += 1
            q = json["quoteResponse"]
            # Policy-conformant SYNTHETIC build: the real pre-sign validator runs.
            unsigned = conformant_unsigned_tx(
                self.kp.pubkey(), q["inputMint"], q["outputMint"],
                # (legacy-fixture callers may pass quotes without amounts)
                int(q.get("inAmount") or 1), int(q.get("outAmount") or 1))
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
            if params[1]["programId"] != TOKEN_2022_PROGRAM:
                return Resp({"result": {"value": []}})
            return Resp({"result": {"value": [{"account": {"data": {"parsed": {"info": {
                "mint": m, "tokenAmount": {"uiAmount": ui, "amount": str(round(ui * 10**9)), "decimals": 9}}}}}} for m, ui in self.wallet.items() if ui]}})
        raise AssertionError(f"unexpected RPC {method}")

    def _send(self, params):
        assert params[1]["encoding"] == "base64"
        tx = VersionedTransaction.from_bytes(base64.b64decode(params[0]))
        assert tx.verify_with_results() == [True]            # real, valid signature
        sig = str(tx.signatures[0])
        self.durable_at_send.append(sig in self._open_disk_signatures())
        self.sends.append(sig)
        mode = self.plan.pop(0) if self.plan else "land"
        if mode == "land":
            self.land(sig)
        elif mode == "land_hidden":
            self.land(sig, visible=False)
        elif mode == "drop":
            pass                                              # accepted, never executes
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
        elif mode == "wrong_sig":
            self.land(sig, visible=False)
            return Resp({"result": "SomeOtherSignature111111111111111111111111111111111111111111111"})
        else:
            raise AssertionError(mode)
        return Resp({"result": sig})


@pytest.fixture
def rw(monkeypatch, tmp_path):
    w = RWorld(tmp_path, Keypair())                          # throwaway, test-generated
    monkeypatch.setattr(reconcile_module, "get_wallet_pubkey", lambda cfg: w.owner)
    monkeypatch.setattr(main_module, "_price_usd", lambda jup, mint, dec: w.marks.get(mint))
    monkeypatch.setattr(main_module, "time", types.SimpleNamespace(
        time=lambda: w.now, sleep=lambda s: None))
    return w


def boot(w):
    cfg = Config(live_trading=True, wallet_private_key=SENTINEL, burner_wallet_pubkey=w.owner,
                 rpc_url=RPC, bankroll_usd=20.0, max_position_pct=10.0,
                 confirmation_timeout_seconds=0.0, confirmation_commitment="confirmed",
                 poll_interval_seconds=0, state_file=w.state, process_lock_file=w.lock)
    bot = main_module.TradingBot(cfg)
    bot.session = w
    bot.jup = DirectJup()
    bot.executor = SwapExecutor(cfg, jup=bot.jup, session=w)
    bot.executor._load_keypair = lambda _Keypair: w.kp
    bot.notifier = Notes()
    bot.entry_scans = []
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: bot.entry_scans.append(1) or [])
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


def start(w, raw=IN_RAW, mark=SL_MARK):
    bot = boot(w)
    tokens = raw / 10**8
    bot.portfolio.open(MINT, "XSCP", ENTRY, tokens * ENTRY, tokens, token_decimals=8,
                       entry_signature="entry", actual_in_amount=1, actual_out_amount=raw,
                       fee_lamports=5000, raw_tokens=raw, raw_tokens_initial=raw)
    w.wallet[MINT] = tokens
    assert bot._save() is True
    w.marks[MINT] = mark
    return bot


def booked_once(w, sig, *, pnl=SL_PNL, remaining_raw=None, rung=None):
    """Exactly-once, read from disk: one settled intent with this signature."""
    state = persisted(w)
    mine = intents_of(state)
    settled = [i for i in mine if i["status"] == "settled"]
    assert len(settled) == 1, mine
    (s,) = settled
    assert s["tx_signature"] == sig and w.sends.count(sig) == 1
    assert (s["raw_amount"], s["actual_in_amount"], s["booked_raw"]) == (IN_RAW, IN_RAW, IN_RAW)
    assert (s["actual_out_amount"], s["fee_lamports"]) == (OUT_LAMPORTS, FEE)
    assert s["booked_pnl"] == pytest.approx(pnl) and s.get("rung") == rung
    assert not [i for i in mine if i["status"] in main_module.OPEN_SELL_INTENT_STATUSES]
    assert state["portfolio"]["realized_pnl"] == pytest.approx(pnl)
    assert state["risk"]["realized_pnl_today"] == pytest.approx(pnl)
    pos = position(state)
    if remaining_raw is None:
        assert pos is None
    else:
        assert pos["raw_tokens"] == remaining_raw
        assert pos["ladder_filled"] == ([rung] if rung is not None else [])
    assert all(w.durable_at_send), "a signature was broadcast before it was durable"
    return state


def no_secrets(w):
    text = open(w.state).read()
    secret = bytes(w.kp)
    assert SENTINEL not in text
    import base58
    assert base58.b58encode(secret).decode() not in text
    assert base58.b58encode(secret[:32]).decode() not in text
    assert json.dumps(list(secret)) not in text and str(list(secret)) not in text
    for intent in intents_of(json.loads(text)):
        assert not {"signed_tx", "raw_tx", "transaction", "keypair"} & set(intent)
        sig = intent.get("tx_signature")
        assert sig is None or (len(base58.b58decode(sig)) == 64 and sig != "1" * 64)


def crash_before_save(monkeypatch, k):
    real = main_module.save_state
    n = [0]

    def save(*a, **kw):
        n[0] += 1
        if n[0] == k:
            raise SystemExit(f"crash before save {k}")
        return real(*a, **kw)
    monkeypatch.setattr(main_module, "save_state", save)
    return real


def crash_after_save(monkeypatch, k):
    real = main_module.save_state
    n = [0]

    def save(*a, **kw):
        ok = real(*a, **kw)
        n[0] += 1
        if n[0] == k:
            raise SystemExit(f"crash after save {k}")
        return ok
    monkeypatch.setattr(main_module, "save_state", save)
    return real


def fail_save(monkeypatch, k):
    real = main_module.save_state
    n = [0]

    def save(*a, **kw):
        n[0] += 1
        return False if n[0] == k else real(*a, **kw)
    monkeypatch.setattr(main_module, "save_state", save)
    return real


# -- the happy paths, with the invariant observed at the broadcast -------------------

def test_on_time_signature_saved_before_broadcast_equals_broadcast_bytes(rw, monkeypatch):
    bot = start(rw)
    writes = []
    real = main_module.save_state

    def spy(*a, **kw):
        ok = real(*a, **kw)
        writes.append((list(rw.sends), [i.get("tx_signature") for i in intents_of(persisted(rw))]))
        return ok
    monkeypatch.setattr(main_module, "save_state", spy)
    bot._tick()
    (sig,) = rw.sends
    # save 1: intent without signature; save 2: the exact signature, BEFORE broadcast
    assert writes[0] == ([], [None]) and writes[1] == ([], [sig])
    assert rw.durable_at_send == [True]
    assert bot.jup.quotes == [(MINT, SOL_MINT, IN_RAW)] and rw.builds == 1
    booked_once(rw, sig)
    no_secrets(rw)


def test_delayed_path_resolves_by_the_saved_signature(rw):
    bot = start(rw)
    rw.plan = ["land_hidden"]
    bot._tick()
    (sig,) = rw.sends
    (intent,) = intents_of(persisted(rw))
    assert (intent["status"], intent["tx_signature"]) == ("unresolved", sig)
    rw.show(sig)
    for _ in range(3):
        bot._tick()
    booked_once(rw, sig)
    assert bot.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]


def test_take_profit_rung_booked_once_with_saved_signature(rw):
    bot = start(rw, raw=RAW_DOUBLE, mark=TP_MARK)
    bot._tick()
    (sig,) = rw.sends
    booked_once(rw, sig, pnl=TP50_PNL, remaining_raw=IN_RAW, rung=50.0)


# -- crashes / failures before the broadcast: nothing sent, safe recovery ------------

@pytest.mark.parametrize("where", ["before_signing", "after_signing_before_persist"])
def test_crash_before_signature_durable_sends_nothing_and_recovers_once(rw, monkeypatch, where):
    bot = start(rw)
    if where == "before_signing":
        rw.crash_build = True
    else:
        crash_before_save(monkeypatch, 2)              # signed; dies before the signature save
    with pytest.raises(SystemExit):
        bot._tick()
    rw.crash_build = False
    monkeypatch.setattr(main_module, "save_state", _real_save())
    assert rw.sends == []
    (intent,) = intents_of(persisted(rw))
    assert (intent["status"], intent["tx_signature"]) == ("pending", None)
    # a missing signature is NOT evidence of non-execution: frozen until expiry
    bot2 = boot(rw)
    run(bot2, 3)
    assert rw.sends == [] and bot2.entry_scans == []
    assert [i["status"] for i in intents_of(persisted(rw))] == ["pending"]
    rw.now += EXPIRY + 1                               # expiry + unchanged wallet: not executed
    run(boot(rw), 4)
    (sig,) = rw.sends
    state = booked_once(rw, sig)
    assert [i["status"] for i in intents_of(state)] == ["failed", "settled"]


def _real_save():
    from bot.state import save_state
    return save_state


def test_presave_failure_no_broadcast_then_new_attempt_keeps_audit(rw, monkeypatch):
    bot = start(rw)
    fail_save(monkeypatch, 2)                          # the signature save is not durable
    bot._tick()
    assert rw.sends == [] and rw.builds == 1 and bot.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]
    (intent,) = bot.portfolio.pending_intents
    old_sig = intent["tx_signature"]
    assert intent["status"] == "blocked" and "not durably saved" in intent["error"]
    assert old_sig and len(old_sig) > 80               # the real derived signature, kept as audit
    monkeypatch.setattr(main_module, "save_state", _real_save())
    bot._tick()                                        # cleared -> exit re-evaluated -> new attempt
    (sig,) = rw.sends
    assert sig != old_sig and rw.builds == 2           # fresh build, fresh signature
    state = booked_once(rw, sig)
    old = intents_of(state)[0]
    assert (old["status"], old["tx_signature"]) == ("failed", old_sig)
    no_secrets(rw)


@pytest.mark.parametrize("crash", [False, True])
def test_presave_dir_fsync_after_replace_disk_has_signature_nothing_sent(rw, monkeypatch, crash):
    """The signature write replaced the file (disk shows it) but is not
    durable. Nothing is broadcast. On restart that signature is NOT proof of
    anything: resolved via status lookup + expiry + unchanged wallet."""
    bot = start(rw)
    install_faulty_os(monkeypatch, {2: "dir_fsync"}, crash=crash, path=rw.state)
    if crash:
        with pytest.raises(BaseException):
            bot._tick()
    else:
        bot._tick()
    assert rw.sends == []
    (intent,) = intents_of(persisted(rw))
    disk_sig = intent["tx_signature"]
    assert disk_sig and intent["status"] in ({"pending"} if crash else {"blocked"})
    install_faulty_os(monkeypatch)
    if crash:
        bot2 = boot(rw)
        run(bot2, 3)                                   # signature unknown on chain, not expired
        assert rw.sends == [] and "getSignatureStatuses" in rw.rpc
        assert [i["status"] for i in intents_of(persisted(rw))] == ["pending"]
        rw.now += EXPIRY + 1
    run(boot(rw), 4)
    (sig,) = rw.sends
    assert sig != disk_sig
    state = booked_once(rw, sig)
    assert intents_of(state)[0]["tx_signature"] == disk_sig
    assert intents_of(state)[0]["status"] == "failed"


def test_crash_after_signature_persisted_before_broadcast_reconciles_by_expiry(rw, monkeypatch):
    bot = start(rw)
    crash_after_save(monkeypatch, 2)
    with pytest.raises(SystemExit):
        bot._tick()
    monkeypatch.setattr(main_module, "save_state", _real_save())
    assert rw.sends == []
    (intent,) = intents_of(persisted(rw))
    never_sent = intent["tx_signature"]
    assert intent["status"] == "pending" and never_sent
    bot2 = boot(rw)
    run(bot2, 3)                                       # not on chain, not expired: frozen
    assert rw.sends == [] and bot2.notifier.sells == []
    assert persisted(rw)["portfolio"]["realized_pnl"] == 0.0     # no false booking
    rw.now += EXPIRY + 1
    run(boot(rw), 4)
    (sig,) = rw.sends
    state = booked_once(rw, sig)
    assert sig != never_sent
    assert [(i["status"], i["tx_signature"]) for i in intents_of(state)] == [
        ("failed", never_sent), ("settled", sig)]


# -- crashes / errors during or after the broadcast: never a blind resend ------------

@pytest.mark.parametrize("kind", ["stop_loss", "take_profit"])
def test_crash_during_broadcast_landed_books_once_by_saved_signature(rw, kind):
    raw, mark = (IN_RAW, SL_MARK) if kind == "stop_loss" else (RAW_DOUBLE, TP_MARK)
    bot = start(rw, raw=raw, mark=mark)
    rw.plan = ["crash_landed"]
    with pytest.raises(SystemExit):
        bot._tick()
    (sig,) = rw.sends
    (intent,) = intents_of(persisted(rw))
    assert (intent["status"], intent["tx_signature"]) == ("pending", sig)
    bot2 = boot(rw)
    assert run(bot2, 3) == 3
    assert rw.sends == [sig] and len(bot2.notifier.sells) == 1
    if kind == "stop_loss":
        booked_once(rw, sig)
    else:
        booked_once(rw, sig, pnl=TP50_PNL, remaining_raw=IN_RAW, rung=50.0)
    run(boot(rw), 2)
    assert rw.sends == [sig]


def test_crash_during_broadcast_that_never_left_resolves_by_expiry_then_once(rw):
    bot = start(rw)
    rw.plan = ["crash_dropped"]
    with pytest.raises(SystemExit):
        bot._tick()
    (first,) = rw.sends
    run(boot(rw), 3)
    assert rw.sends == [first]                         # no blind resend
    rw.now += EXPIRY + 1
    run(boot(rw), 4)
    assert len(rw.sends) == 2
    booked_once(rw, rw.sends[1])
    assert intents_of(persisted(rw))[0]["tx_signature"] == first


@pytest.mark.parametrize("mode", ["timeout_landed", "lost_landed"])
def test_send_error_after_landing_no_resend_books_once(rw, mode):
    bot = start(rw)
    rw.plan = [mode]
    bot._tick()
    (sig,) = rw.sends
    (intent,) = bot.portfolio.pending_intents
    assert (intent["status"], intent["tx_signature"]) == ("unresolved", sig)
    for _ in range(3):
        bot._tick()
    assert rw.sends == [sig]
    if mode == "lost_landed":
        rw.show(sig)                                   # status becomes visible later
        for _ in range(3):
            bot._tick()
    booked_once(rw, sig)
    assert len(bot.notifier.sells) == 1


def test_send_timeout_that_never_landed_no_resend_until_expiry(rw):
    bot = start(rw)
    rw.plan = ["timeout_dropped"]
    bot._tick()
    (first,) = rw.sends
    for _ in range(4):
        bot._tick()
    assert rw.sends == [first] and bot.entry_scans == []
    rw.now += EXPIRY + 1
    for _ in range(4):
        bot._tick()
    assert len(rw.sends) == 2
    booked_once(rw, rw.sends[1])


def test_uncertain_outcome_stays_frozen_no_second_sell(rw):
    """Landed (wallet dropped) but never visible: past expiry it is neither
    failed nor booked; the mint stays frozen and entries stay blocked."""
    bot = start(rw)
    rw.plan = ["land_hidden"]
    bot._tick()
    (sig,) = rw.sends
    rw.now += EXPIRY + 1
    bot2 = boot(rw)
    run(bot2, 4)
    state = persisted(rw)
    assert rw.sends == [sig] and bot2.entry_scans == []
    (intent,) = intents_of(state)
    assert intent["status"] in main_module.OPEN_SELL_INTENT_STATUSES and intent["tx_signature"] == sig
    assert state["portfolio"]["realized_pnl"] == 0.0 and position(state)["raw_tokens"] == IN_RAW


def test_rpc_reporting_a_different_signature_never_replaces_the_saved_one(rw):
    bot = start(rw)
    rw.plan = ["wrong_sig"]
    bot._tick()
    (sig,) = rw.sends
    (intent,) = intents_of(persisted(rw))
    assert intent["tx_signature"] == sig and intent["status"] == "unresolved"
    for _ in range(3):
        bot._tick()
    assert rw.sends == [sig]
    rw.show(sig)
    for _ in range(3):
        bot._tick()
    booked_once(rw, sig)


# -- restart from disk after every save boundary of the settling tick ---------------

@pytest.mark.parametrize("path", ["on_time", "delayed"])
def test_crash_after_every_save_converges_exactly_once(rw, monkeypatch, tmp_path, path):
    k = 1
    while True:
        d = tmp_path / f"{path}-{k}"
        d.mkdir()
        rw.__init__(d, rw.kp)
        bot = start(rw)
        if path == "delayed":
            rw.plan = ["land_hidden"]
            bot._tick()
            rw.show(rw.sends[0])
        real = crash_after_save(monkeypatch, k)
        try:
            bot._tick()
            crashed = False
        except SystemExit:
            crashed = True
        monkeypatch.setattr(main_module, "save_state", real)
        if not crashed:
            break
        before = list(rw.sends)
        assert persisted(rw)["portfolio"]["realized_pnl"] in (0.0, pytest.approx(SL_PNL))
        bot2 = boot(rw)
        run(bot2, 3)                                   # within the window: no blind resend
        assert rw.sends == before
        rw.now += EXPIRY + 1
        run(boot(rw), 4)
        assert len(rw.sends) == 1, (k, rw.sends)      # exactly one broadcast in total
        booked_once(rw, rw.sends[0])
        run(boot(rw), 2)
        assert len(rw.sends) == 1
        k += 1
    # on-time: pending, signature (pre-broadcast), evidence, booking; delayed: evidence, booking
    assert k - 1 == {"on_time": 4, "delayed": 2}[path]
    booked_once(rw, rw.sends[0])
    no_secrets(rw)
