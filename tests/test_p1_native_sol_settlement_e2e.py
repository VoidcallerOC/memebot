"""P1: real native-SOL fill, end to end through executor, parser and resolver,
with crash-safe exactly-once settlement.

Fixture: ``tests/fixtures/native_sol_sell_fWueMsPH.json`` (unmodified public
``getTransaction`` result, sha256 pinned below). Owner HckQ93... sells
274,425 raw XsCPL9... (Token-2022, 8 dp) into a temporary wSOL account that is
closed in the same transaction: 8,656,067 lamports native SOL out, fee 10,143.

What is real here: ``TradingBot.__init__`` (production ``load_state``),
``TradingBot.run``/``_tick``/``_manage_positions``/``_sell``/
``_settle_live_sell``/``_resolve_sell_intents``, ``SwapExecutor.swap`` ->
``_execute_live`` -> ``_confirm_transaction`` -> ``_extract_confirmed_fill``,
``chain.rpc_call``/``get_signature_status``/``get_confirmed_transaction``/
``extract_fill``/``fetch_wallet_token_balances``, ``reconcile_wallet``,
``Portfolio``/``RiskManager`` accounting and ``save_state`` (atomic write).

What is fake (offline only): the HTTP session (answers JSON-RPC from the
fixture and a programmable wallet), the Jupiter quote/build, the signing
constructor (a spy: nothing is signed), ``send_raw_transaction`` (a stub: nothing
is sent), the key loader (returns the fixture owner pubkey, no key material),
the price mark and the notifier. ``extract_fill`` is never stubbed and no
intent is pre-set to ``settled``.

Sizing on main (no exact raw fields): the intent's ``raw_amount`` comes from
main's own ``int(min(original_tokens * fraction, tokens) * 10**8)``; positions
of 0.00274425 tokens (fraction 1.0) and 0.0054885 tokens (fraction 0.5) both
give exactly 274,425 (verified in ``test_main_sizing_maps_to_fixture_raw``).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import pathlib
import sys
import types

import pytest
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
import solders.transaction as real_tx
from solders.transaction import VersionedTransaction

import bot.jupiter as jupiter_module
import bot.main as main_module
import bot.reconcile as reconcile_module
from bot.chain import TOKEN_2022_PROGRAM, extract_fill, get_confirmed_transaction
from bot.config import Config
from bot.jupiter import SOL_MINT, SwapExecutor

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "native_sol_sell_fWueMsPH.json"
FIXTURE_SHA256 = "730382ed56c73866e74cc80bbadfa75732c9ab943d3f4d8d971d61ec14be75b1"
SIG = "fWueMsPHe5MiGePaSXQiS1B9VEMjA7MH65QieuMKNme2PWTTecoQgHUnKvUyRTCXgrA4rfSygZzCGPvpwhmqo6f"
SIG2 = "SecondSaleSignature1111111111111111111111111111111111111111111111111111111111111111"
OWNER = "HckQ93Xqjjo8mwt5pNPWvyCTZXQZ858rvzmm7ZRrZg9t"
MINT = "XsCPL9dNWBMvFtTmwcCA5v3xWPSMEBCszbQdiLLq6aN"
MINT2 = "SecondMint111111111111111111111111111111111"
IN_RAW, OUT_LAMPORTS, FEE = 274_425, 8_656_067, 10_143
RPC = "https://rpc.test"
SWAP_URL = "https://jupiter.test/swap"
SENTINEL = "test-only-sentinel"

RAW_DOUBLE = 2 * IN_RAW     # 548,850 raw bought; the first TP rung (0.5) sells 274,425
FULL = IN_RAW / 10**8       # float tokens are DERIVED from the exact raw holding
DOUBLE = RAW_DOUBLE / 10**8
ENTRY = 700.0
SL_MARK = 350.0             # -50%: stop-loss, fraction 1.0
TP_MARK = 1120.0            # +60%: only the 50% rung, fraction 0.5
HOLD_MARK = 700.0           # no exit action
SL_PNL = FULL * (SL_MARK - ENTRY)


def fixture_result(mint=MINT):
    text = FIXTURE.read_text()
    data = json.loads(text.replace(MINT, mint)) if mint != MINT else json.loads(text)
    assert data["_source"]["signature"] == SIG
    return data["result"]


# -- offline world: HTTP/RPC boundary, signing spy, send stub -----------------

class Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class World:
    """Chain + wallet + process boundary shared by every 'process lifetime'."""

    def __init__(self, tmp_path):
        self.state = str(tmp_path / "state.json")
        self.lock = str(tmp_path / "lock")
        self.txs = {SIG: fixture_result()}   # sig -> getTransaction result (None = null)
        self.status = {}                      # sig -> signature status dict (absent = null)
        self.wallet = {}                      # mint -> ui amount (Token-2022 accounts)
        self.sig_queue = []                   # signatures returned by the send stub, in order
        self.on_send = {}                     # sig -> callable(world): what the chain does
        self.sends, self.signs, self.rpc = [], [], []
        self.marks = {}
        self.now = 1_800_000_000.0
        self.crash_after_send = False

    # requests.Session stand-in: the only network surface the bot touches
    def post(self, url, json=None, headers=None, timeout=None):
        if url == SWAP_URL:
            return Resp({"swapTransaction": V0_B64})
        assert url == RPC, url
        method, params = json["method"], json["params"]
        self.rpc.append(method)
        if method == "getSignatureStatuses":
            return Resp({"result": {"value": [self.status.get(params[0][0])]}})
        if method == "getTransaction":
            assert params[1]["encoding"] == "jsonParsed"
            assert params[1]["maxSupportedTransactionVersion"] == 0
            sig = params[0]
            tx = self.txs.get(sig) if sig in self.status else None
            return Resp({"result": copy.deepcopy(tx)})
        if method == "getTokenAccountsByOwner":
            assert params[0] == OWNER
            if params[1]["programId"] != TOKEN_2022_PROGRAM:
                return Resp({"result": {"value": []}})
            accounts = [{"account": {"data": {"parsed": {"info": {
                "mint": m, "tokenAmount": {"uiAmount": ui, "amount": str(round(ui * 10**9)), "decimals": 9}}}}}}
                for m, ui in self.wallet.items() if ui]
            return Resp({"result": {"value": accounts}})
        raise AssertionError(f"unexpected RPC method {method}")  # e.g. sendTransaction

    def send(self, session, rpc_url, raw):
        assert raw == b"spy-signed"
        sig = self.sig_queue.pop(0)
        self.sends.append(sig)
        effect = self.on_send.get(sig)
        if effect:
            effect(self)
        if self.crash_after_send:
            raise SystemExit("crash: process died after send, before the result returned")
        return sig


def lands(mint, sold_ui, visible=True):
    """The sale executes on chain; optionally its status is already visible."""
    def effect(w):
        w.wallet[mint] = round(w.wallet.get(mint, 0.0) - sold_ui, 12)
        if visible:
            w.status[SIG if mint == MINT else SIG2] = {"confirmationStatus": "finalized", "err": None}
    return effect


def show(w, sig=SIG):
    w.status[sig] = {"confirmationStatus": "finalized", "err": None}


_PAYER = Pubkey.from_string(OWNER)
V0_B64 = base64.b64encode(bytes(VersionedTransaction.populate(
    MessageV0.try_compile(_PAYER, [Instruction(
        Pubkey.from_string("Stake11111111111111111111111111111111111111"), b"\x01",
        [AccountMeta(_PAYER, True, True)])], [], Hash.default()),
    [Signature.default()]))).decode()


class Jup:
    swap_url = SWAP_URL

    def __init__(self):
        self.quotes = []

    def quote(self, i, o, amount):
        self.quotes.append((i, o, amount))
        return {"inputMint": i, "outputMint": o, "inAmount": str(amount),
                "outAmount": str(OUT_LAMPORTS), "otherAmountThreshold": str(OUT_LAMPORTS * 99 // 100),
                "swapMode": "ExactIn", "slippageBps": 100}

    def _headers(self):
        return {}


class Notes:
    enabled = True

    def __init__(self):
        self.sells, self.halts, self.sent = [], [], []

    def startup(self, *a):
        pass

    def sell(self, *a):
        self.sells.append(a)

    def halt(self, *a):
        self.halts.append(a)

    def send(self, text):
        self.sent.append(text)

    def buy(self, *a):
        raise AssertionError("no buys in these tests")


@pytest.fixture
def world(monkeypatch, tmp_path):
    w = World(tmp_path)

    class SpySigned:
        def __init__(self, signature):
            # A signed transaction is known on chain by its first signature:
            # the one the send stub will return for it.
            self.signatures = [signature]

        def __bytes__(self):
            return b"spy-signed"

    class SpyVersionedTransaction:
        """Real decoding; the signing constructor is a spy (no crypto)."""
        from_bytes = staticmethod(VersionedTransaction.from_bytes)

        def __new__(cls, message, signers):
            w.signs.append(type(message).__name__)
            return SpySigned(w.sig_queue[0] if w.sig_queue else "UnqueuedSpySignature")

    spy = types.ModuleType("solders.transaction")
    spy.VersionedTransaction = SpyVersionedTransaction
    spy.Legacy = real_tx.Legacy
    monkeypatch.setitem(sys.modules, "solders.transaction", spy)
    monkeypatch.setattr(jupiter_module, "send_raw_transaction", w.send)
    monkeypatch.setattr(reconcile_module, "get_wallet_pubkey", lambda cfg: OWNER)
    monkeypatch.setattr(main_module, "_price_usd", lambda jup, mint, dec: w.marks.get(mint))
    monkeypatch.setattr(main_module, "time", types.SimpleNamespace(
        time=lambda: w.now, sleep=lambda s: None))
    return w


def config(w):
    return Config(live_trading=True, wallet_private_key=SENTINEL, burner_wallet_pubkey=OWNER,
                  rpc_url=RPC, bankroll_usd=20.0, max_position_pct=10.0,
                  confirmation_timeout_seconds=0.0, confirmation_commitment="confirmed",
                  poll_interval_seconds=0, state_file=w.state, process_lock_file=w.lock)


def boot(w):
    """A new process: production constructor (loads the state file)."""
    cfg = config(w)
    bot = main_module.TradingBot(cfg)
    bot.session = w
    bot.jup = Jup()
    bot.executor = SwapExecutor(cfg, jup=bot.jup, session=w)
    bot.executor._load_keypair = lambda _kp: types.SimpleNamespace(pubkey=lambda: OWNER)
    bot.notifier = Notes()
    bot.entry_scans = []
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: bot.entry_scans.append(1) or [])
    return bot


def run(bot, ticks):
    """Production ``run()`` (startup resolve + reconcile, then ticks)."""
    done = []
    real_tick = bot._tick

    def tick():
        real_tick()
        done.append(1)
        if len(done) >= ticks:
            bot._running = False

    bot._tick = tick
    bot.run()
    return len(done)


def open_position(bot, w, mint=MINT, raw=IN_RAW, entry=ENTRY):
    """A live position exactly as the armed entry path records it: the raw
    holding is the verified entry fill's integer raw output (never derived
    from a float); ``tokens`` is derived from it."""
    tokens = raw / 10**8
    bot.portfolio.open(mint, mint[:4], entry, tokens * entry, tokens, token_decimals=8,
                       entry_signature="entry-" + mint[:4], actual_in_amount=1,
                       actual_out_amount=raw, fee_lamports=5000,
                       raw_tokens=raw, raw_tokens_initial=raw)
    w.wallet[mint] = tokens
    bot._save()


def persisted(w):
    with open(w.state) as f:
        return json.load(f)


def intents_of(state, mint=MINT):
    return [i for i in state["portfolio"]["pending_intents"] if i.get("mint") == mint]


def position(state, mint=MINT):
    return next((p for p in state["portfolio"]["positions"] if p["mint"] == mint), None)


def assert_booked_once(w, *, pnl=SL_PNL, remaining_raw=None, mint=MINT, sig=SIG, sends=1,
                       risk_pnl=None):
    """Exactly-once invariants read from the persisted state file only."""
    state = persisted(w)
    mine = intents_of(state, mint)
    settled = [i for i in mine if i["status"] == "settled"]
    assert len(settled) == 1, mine
    intent = settled[0]
    assert intent["tx_signature"] == sig
    assert intent["raw_amount"] == IN_RAW and intent["booked_raw"] == IN_RAW
    assert intent["booked_pnl"] == pytest.approx(pnl)
    assert (intent["actual_in_amount"], intent["actual_out_amount"], intent["fee_lamports"]) == (
        IN_RAW, OUT_LAMPORTS, FEE)
    assert not [i for i in mine if i["status"] in main_module.OPEN_SELL_INTENT_STATUSES]
    assert state["portfolio"]["realized_pnl"] == pytest.approx(pnl)
    assert state["risk"]["realized_pnl_today"] == pytest.approx(pnl if risk_pnl is None else risk_pnl)
    pos = position(state, mint)
    if remaining_raw is None:
        assert pos is None
    else:
        assert pos["raw_tokens"] == remaining_raw                      # exact integer
        assert pos["raw_tokens"] == pos["raw_tokens_initial"] - IN_RAW  # decremented once
        assert pos["tokens"] == remaining_raw / 10**8                   # derived from raw
        assert pos["realized_pnl"] == pytest.approx(pnl)
    assert w.sends.count(sig) == sends
    return state


def assert_never_terminal_unbooked(w, mint=MINT, risk_before=0.0):
    """Read from disk: every settled intent's accounting is persisted with it,
    and nothing is booked without a settled intent (raw holding, position,
    portfolio P&L and daily-risk P&L all agree with the settled intents)."""
    state = persisted(w)
    settled = [i for i in intents_of(state, mint) if i["status"] == "settled"]
    pos = position(state, mint)
    booked_raw = sum(i.get("booked_raw") or 0 for i in settled)
    booked_pnl = sum(i.get("booked_pnl") or 0.0 for i in settled)
    assert all("booked_at" in i for i in settled), ("settled without booking marker", state)
    if pos is not None:
        assert pos["raw_tokens"] == pos["raw_tokens_initial"] - booked_raw, (
            "raw holding disagrees with settled intents (terminal-but-unbooked)", state)
        assert pos["tokens"] == pos["raw_tokens"] / 10**8
    else:
        assert settled, ("position gone without a settled sell", state)
    assert state["portfolio"]["realized_pnl"] == pytest.approx(booked_pnl)
    assert state["risk"]["realized_pnl_today"] == pytest.approx(risk_before + booked_pnl)


# -- fixture and real parser path ------------------------------------------------

def test_fixture_pinned_and_parsed_through_real_rpc_path(world):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    show(world)
    tx = get_confirmed_transaction(world, RPC, SIG, "confirmed")
    assert world.rpc == ["getTransaction"]
    fill = extract_fill(tx, OWNER, MINT, SOL_MINT)
    assert (fill.actual_in_amount, fill.in_decimals) == (IN_RAW, 8)
    assert (fill.actual_out_amount, fill.out_decimals) == (OUT_LAMPORTS, 9)
    assert fill.fee_lamports == FEE and fill.complete
    assert tx["meta"]["err"] is None and tx["version"] == 0


def test_exact_sizing_maps_to_fixture_raw():
    def pos(raw):
        return types.SimpleNamespace(token_decimals=8, raw_tokens=raw, raw_tokens_initial=raw)
    assert main_module.exact_raw_sell_amount(pos(IN_RAW), 1.0) == (IN_RAW, "")
    assert main_module.exact_raw_sell_amount(pos(RAW_DOUBLE), 0.5) == (IN_RAW, "")
    assert main_module.exact_raw_sell_amount(pos(IN_RAW - 1), 1.0) == (IN_RAW - 1, "")


# -- on-time path (_settle_live_sell) ------------------------------------------

def test_on_time_stop_loss_settles_once_and_restart_does_not_rebook(world):
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL)
    world.marks[MINT] = SL_MARK
    bot._tick()
    assert bot.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]
    assert world.signs == ["MessageV0"] and world.sends == [SIG]
    assert world.rpc[:2] == ["getSignatureStatuses", "getTransaction"]
    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.realized_pnl == pytest.approx(SL_PNL)
    assert len(bot.notifier.sells) == 1
    assert_booked_once(world)

    world.rpc.clear()
    bot2 = boot(world)                       # restart through production load
    assert run(bot2, 3) == 3                 # startup reconcile clean, three ticks
    assert "getTokenAccountsByOwner" in world.rpc
    assert bot2.notifier.sells == []
    assert_booked_once(world)


def test_on_time_partial_take_profit_settles_once_restart_reconciles_clean(world):
    bot = boot(world)
    open_position(bot, world, raw=RAW_DOUBLE)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL)
    world.marks[MINT] = TP_MARK
    bot._tick()
    tp_pnl = FULL * (TP_MARK - ENTRY)
    state = assert_booked_once(world, pnl=tp_pnl, remaining_raw=IN_RAW)
    (intent,) = intents_of(state)
    # submitted == intended == fill-validated == decremented, all exact ints
    assert bot.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]
    assert intent["raw_amount"] == intent["actual_in_amount"] == intent["booked_raw"] == IN_RAW
    assert RAW_DOUBLE - position(state)["raw_tokens"] == IN_RAW
    assert position(state)["ladder_filled"] == [50.0]

    bot2 = boot(world)
    assert run(bot2, 3) == 3                 # reconcile: tracked 0.00274425 == wallet
    assert bot2.notifier.sells == [] and world.sends == [SIG]
    assert_booked_once(world, pnl=tp_pnl, remaining_raw=IN_RAW)


# -- delayed path (_resolve_sell_intents) -----------------------------------------

def test_delayed_timeout_then_resolver_settles_once(world):
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL, visible=False)
    world.marks[MINT] = SL_MARK
    bot._tick()                               # TIMEOUT: unknown outcome, nothing booked
    state = persisted(world)
    assert [i["status"] for i in intents_of(state)] == ["unresolved"]
    assert intents_of(state)[0]["tx_signature"] == SIG
    assert position(state)["tokens"] == FULL and state["portfolio"]["realized_pnl"] == 0.0

    show(world)
    bot._tick()                               # resolver: real getTransaction + extract_fill
    assert bot.notifier.sells and len(bot.notifier.sells) == 1
    assert_booked_once(world)
    bot._tick()                               # reconciliation tick, then a normal tick
    bot._tick()
    assert len(bot.notifier.sells) == 1 and world.sends == [SIG]

    bot2 = boot(world)
    assert run(bot2, 3) == 3
    assert bot2.notifier.sells == []
    assert_booked_once(world)


def test_restart_with_unresolved_intent_books_once_at_startup(world):
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL, visible=False)
    world.marks[MINT] = SL_MARK
    bot._tick()
    del bot                                   # process ends with the intent unresolved

    show(world)
    bot2 = boot(world)
    assert run(bot2, 2) == 2                  # startup resolver books, reconcile clean
    assert len(bot2.notifier.sells) == 1
    assert_booked_once(world)
    bot3 = boot(world)
    assert run(bot3, 2) == 2
    assert bot3.notifier.sells == []
    assert_booked_once(world)


# -- mismatched / unprovable fills stay unresolved -----------------------------

@pytest.mark.parametrize("path", ["on_time", "delayed"])
@pytest.mark.parametrize("variant", ["raw_mismatch", "no_transaction", "malformed_meta"])
def test_bad_evidence_stays_unresolved_frozen_without_pnl(world, path, variant):
    raw = IN_RAW - 1 if variant == "raw_mismatch" else IN_RAW   # intent raw 274,424 vs fill 274,425
    tokens = raw / 10**8
    if variant == "no_transaction":
        world.txs[SIG] = None                  # confirmed signature, getTransaction null
    if variant == "malformed_meta":
        world.txs[SIG] = dict(fixture_result(), meta="corrupted")  # documented edit: unparseable meta
    bot = boot(world)
    open_position(bot, world, raw=raw)
    bot.risk.record_realized_pnl(-0.5)          # earlier loss today; must survive untouched
    bot._save()
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, tokens, visible=path == "on_time")
    world.marks[MINT] = SL_MARK
    bot._tick()
    if path == "delayed":
        show(world)
    for _ in range(3):
        world.now += 10
        bot._tick()
    state = persisted(world)
    mine = intents_of(state)
    assert [i["status"] for i in mine] == ["unresolved"]
    assert position(state)["tokens"] == tokens
    assert position(state)["raw_tokens"] == raw == position(state)["raw_tokens_initial"]
    assert state["portfolio"]["realized_pnl"] == 0.0
    assert state["risk"]["realized_pnl_today"] == pytest.approx(-0.5)
    assert bot.notifier.sells == [] and world.sends == [SIG]   # frozen: never re-sold
    assert bot._sell_frozen(MINT)

    bot2 = boot(world)
    run(bot2, 2)                               # startup: still unprovable -> stays open
    state = persisted(world)
    assert [i["status"] for i in intents_of(state)] == ["unresolved"]
    assert "operator action required" in (intents_of(state)[0].get("resolution") or "")
    assert state["portfolio"]["realized_pnl"] == 0.0 and world.sends == [SIG]
    assert bot2.notifier.sells == []


# -- daily-loss halt and other positions ----------------------------------------

@pytest.mark.parametrize("path", ["on_time", "delayed"])
def test_daily_loss_halt_tripped_by_settlement_is_never_reset(world, path):
    bot = boot(world)
    open_position(bot, world)
    bot.risk.record_realized_pnl(-1.5)          # limit is $2.00; this sale trips it
    bot._save()
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL, visible=path == "on_time")
    world.marks[MINT] = SL_MARK
    bot._tick()
    if path == "delayed":
        show(world)
        bot._tick()
    assert len(bot.notifier.halts) == 1
    state = persisted(world)
    assert state["risk"]["halted_today"] is True
    assert state["risk"]["realized_pnl_today"] == pytest.approx(-1.5 + SL_PNL)
    assert state["portfolio"]["realized_pnl"] == pytest.approx(SL_PNL)

    bot2 = boot(world)
    assert bot2.risk.trading_halted()
    run(bot2, 3)
    assert bot2.entry_scans == [] and bot2.notifier.halts == [] and bot2.notifier.sells == []
    state = persisted(world)
    assert state["risk"]["halted_today"] is True
    assert state["risk"]["realized_pnl_today"] == pytest.approx(-1.5 + SL_PNL)
    assert world.sends == [SIG]


def test_other_positions_still_get_exits_while_one_mint_is_frozen(world):
    world.txs[SIG2] = fixture_result(MINT2)    # documented edit: fixture with the mint renamed
    bot = boot(world)
    open_position(bot, world)
    open_position(bot, world, mint=MINT2, raw=RAW_DOUBLE)
    world.sig_queue = [SIG, SIG2]
    world.on_send[SIG2] = lands(MINT2, FULL)    # MINT's sale never lands: outcome stays unknown
    world.marks[MINT] = SL_MARK
    world.marks[MINT2] = HOLD_MARK
    bot._tick()                                 # MINT: TIMEOUT -> unresolved, frozen
    bot._tick()                                 # reconciliation tick (wallet unchanged: clean)
    world.marks[MINT2] = TP_MARK
    world.now += 10
    bot._tick()                                 # MINT2 take-profit still fires and settles
    assert world.sends == [SIG, SIG2]
    assert bot.entry_scans == []                # entries blocked while MINT is unresolved
    tp_pnl = FULL * (TP_MARK - ENTRY)
    assert_booked_once(world, mint=MINT2, sig=SIG2, pnl=tp_pnl, remaining_raw=IN_RAW)
    state = persisted(world)
    assert [i["status"] for i in intents_of(state)] == ["unresolved"]
    assert position(state)["tokens"] == FULL


# -- crash windows: die right after each durable save -----------------------------

def _crash_after_save(monkeypatch, k):
    real = main_module.save_state
    count = [0]

    def save(*a, **kw):
        ok = real(*a, **kw)         # durable (fsync + os.replace) before the crash
        count[0] += 1
        if count[0] == k:
            raise SystemExit(f"crash after save {k}")
        return ok

    monkeypatch.setattr(main_module, "save_state", save)
    return count


def _settle_with_crash(world, monkeypatch, path, k, raw=IN_RAW, mark=SL_MARK):
    """One lifetime: open, sell, and die right after the k-th save of the
    settling tick. Returns True if the crash point was reached."""
    real = main_module.save_state
    bot = boot(world)
    open_position(bot, world, raw=raw)
    world.sig_queue = [SIG, SIG]
    world.on_send[SIG] = lands(MINT, FULL, visible=path == "on_time")
    world.marks[MINT] = mark
    if path == "delayed":
        bot._tick()                     # TIMEOUT: unresolved intent persisted
        state = persisted(world)        # no decrement while the outcome is unknown
        assert position(state)["raw_tokens"] == raw and state["portfolio"]["realized_pnl"] == 0.0
        show(world)
    _crash_after_save(monkeypatch, k)
    try:
        bot._tick()
        crashed = False
    except SystemExit:
        crashed = True
    finally:
        monkeypatch.setattr(main_module, "save_state", real)
    return crashed


@pytest.mark.parametrize("kind", ["stop_loss_full", "take_profit_partial"])
@pytest.mark.parametrize("path", ["on_time", "delayed"])
def test_crash_after_every_save_never_loses_or_double_books(world, monkeypatch, tmp_path, path, kind):
    """Sweep: the process dies right after the k-th durable save of the settling
    tick, for every k. A fresh process then loads the file (production
    constructor) and runs startup + ticks. From persisted state alone: the sale
    is booked exactly once, never terminal-but-unbooked, never re-sold."""
    raw, mark = (IN_RAW, SL_MARK) if kind == "stop_loss_full" else (RAW_DOUBLE, TP_MARK)
    expect = (dict(pnl=SL_PNL) if kind == "stop_loss_full"
              else dict(pnl=FULL * (TP_MARK - ENTRY), remaining_raw=IN_RAW))
    k = 1
    while True:
        (tmp_path / f"{path}-{kind}-{k}").mkdir()
        world.__init__(tmp_path / f"{path}-{kind}-{k}")  # fresh chain/wallet/state per crash point
        if not _settle_with_crash(world, monkeypatch, path, k, raw=raw, mark=mark):
            break
        assert_never_terminal_unbooked(world)
        if path == "on_time" and k == 1:
            # the pre-send intent is durable with its exact raw amount before
            # anything was quoted, signed or sent
            (intent,) = intents_of(persisted(world))
            assert (intent["status"], intent["raw_amount"]) == ("pending", IN_RAW)
            assert world.signs == [] and world.sends == []
        world.now += main_module.SELL_INTENT_EXPIRY_SECONDS + 1  # past any expiry window
        run(boot(world), 4)
        assert_never_terminal_unbooked(world)
        if path == "on_time" and k == 1 and kind == "take_profit_partial":
            # Died before anything was sent: the unsent intent resolves as
            # failed and the TP rung, never consumed, is retried and sold once.
            state = assert_booked_once(world, **expect)
            assert [i["status"] for i in intents_of(state)] == ["failed", "settled"]
            assert position(state)["ladder_filled"] == [50.0]
        else:
            assert_booked_once(world, **expect)
        k += 1
    # saves in the settling tick: on-time = pending, signature (pre-broadcast),
    # confirmed_unbooked, settled;
    # delayed = confirmed_unbooked, settled (main: pending, settled, booked / settled, booked)
    assert k - 1 == {"on_time": 4, "delayed": 2}[path]
    assert_booked_once(world, **expect)                # the uncrashed run as well


def test_crash_after_send_resolves_by_the_signature_saved_before_broadcast(world):
    """Formerly a gap: the process died inside the swap after the transaction
    was sent, before its signature reached the state file, leaving a 'pending'
    intent with no signature (frozen, operator action). The exact signature is
    now durably saved BEFORE broadcast, so a restart resolves the sale by that
    signature and books it exactly once, without resending."""
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL)
    world.crash_after_send = True
    world.marks[MINT] = SL_MARK
    with pytest.raises(SystemExit):
        bot._tick()
    world.crash_after_send = False
    (intent,) = intents_of(persisted(world))
    assert (intent["status"], intent["tx_signature"]) == ("pending", SIG)
    bot2 = boot(world)
    assert run(bot2, 3) == 3
    assert world.sends == [SIG] and len(bot2.notifier.sells) == 1
    assert_booked_once(world)

# -- the settlement state machine directly ------------------------------------------

def _unbooked_intent(**over):
    intent = {"mint": MINT, "fraction": 1.0, "reason": "stop_loss", "requested_price": SL_MARK,
              "status": "confirmed_unbooked", "tx_signature": SIG, "created_at": 1.0,
              "raw_amount": IN_RAW, "actual_in_amount": IN_RAW,
              "actual_out_amount": OUT_LAMPORTS, "fee_lamports": FEE}
    intent.update(over)
    return intent


def test_persisted_confirmed_unbooked_is_booked_once_at_startup_without_chain_lookup(world):
    bot = boot(world)
    open_position(bot, world)
    bot.portfolio.pending_intents.append(_unbooked_intent())
    bot._save()
    world.wallet[MINT] = 0.0                     # the sale landed before the crash
    world.rpc.clear()
    bot2 = boot(world)
    assert run(bot2, 2) == 2
    assert "getTransaction" not in world.rpc and "getSignatureStatuses" not in world.rpc
    assert len(bot2.notifier.sells) == 1
    assert_booked_once(world, sends=0)
    bot3 = boot(world)
    run(bot3, 2)
    assert bot3.notifier.sells == []
    assert_booked_once(world, sends=0)


def test_booking_is_idempotent_in_process(world):
    bot = boot(world)
    open_position(bot, world)
    intent = _unbooked_intent()
    bot.portfolio.pending_intents.append(intent)
    assert bot._book_confirmed_sell(intent) is True
    assert bot._book_confirmed_sell(intent) is False
    intent["status"] = "confirmed_unbooked"       # even a stale status cannot rebook
    assert bot._book_confirmed_sell(intent) is False
    assert bot.portfolio.realized_pnl == pytest.approx(SL_PNL)
    assert len(bot.notifier.sells) == 1


@pytest.mark.parametrize("bad, expected", [
    ({"actual_in_amount": IN_RAW - 1}, "does not match"),
    ({"actual_in_amount": None}, "does not match"),
    ({"actual_in_amount": True, "raw_amount": True}, "raw_amount"),
    ({"actual_out_amount": 0}, "output amount"),
    ({"actual_out_amount": None}, "output amount"),
    ({"tx_signature": None}, "signature"),
    ({"raw_amount": None}, "raw_amount"),
    ({"mint": MINT2}, "position no longer tracked"),
])
def test_unprovable_stored_evidence_stays_open_without_pnl(world, bad, expected):
    bot = boot(world)
    open_position(bot, world)
    intent = _unbooked_intent(**bad)
    bot.portfolio.pending_intents.append(intent)
    bot._save()
    bot._resolve_sell_intents()
    state = persisted(world)
    stored = state["portfolio"]["pending_intents"][0]
    assert stored["status"] == "confirmed_unbooked"
    assert expected in stored["resolution"] and "operator action required" in stored["resolution"]
    assert state["portfolio"]["realized_pnl"] == 0.0 and position(state)["tokens"] == FULL
    assert bot.notifier.sells == []
    assert bot._sell_frozen(intent["mint"])


def test_accounting_failure_rolls_back_memory_and_keeps_intent_open(world):
    bot = boot(world)
    open_position(bot, world)
    bot.risk.record_realized_pnl(-0.25)
    intent = _unbooked_intent()
    bot.portfolio.pending_intents.append(intent)
    bot._save()
    before = persisted(world)

    def boom(pnl):
        raise RuntimeError("risk store failed")

    bot.risk.record_realized_pnl = boom          # instance attribute shadows the method
    with pytest.raises(RuntimeError):
        bot._book_confirmed_sell(intent)
    del bot.risk.record_realized_pnl
    # memory restored exactly; nothing half-booked can be saved later
    assert bot.portfolio.positions[MINT].tokens == FULL
    assert bot.portfolio.realized_pnl == 0.0
    assert bot.risk.realized_pnl_today == pytest.approx(-0.25)
    assert intent["status"] == "confirmed_unbooked"
    assert persisted(world) == before
    assert bot._book_confirmed_sell(intent) is True
    state = persisted(world)
    assert state["risk"]["realized_pnl_today"] == pytest.approx(-0.25 + SL_PNL)
    assert state["portfolio"]["realized_pnl"] == pytest.approx(SL_PNL)
