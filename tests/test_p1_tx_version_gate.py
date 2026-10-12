"""P1: the live executor signs only legacy or v0 transactions.

solders 0.29.0 (pinned in requirements/ci-py3.*.lock) decodes three message
kinds: ``Message`` (legacy, ``version() == Legacy.Legacy``), ``MessageV0``
(``version() == 0``) and ``MessageV1`` (SIMD-0385, ``version() == 1``). On
main the executor signed whatever Jupiter returned; a v1 transaction would be
signed and sent, and ``get_confirmed_transaction``
(``maxSupportedTransactionVersion: 0``) could then never fetch its fill.

All transactions here are real solders objects built with ``populate`` and
default (all-zero) signatures: nothing is cryptographically signed, the
signing constructor is replaced by a spy, and ``send_raw_transaction`` is a
stub. No network.
"""
from __future__ import annotations

import base64
import sys
import types

import pytest
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.message import Message, MessageV0, MessageV1
from solders.pubkey import Pubkey
from solders.signature import Signature
import solders.transaction as real_tx
from solders.transaction import Legacy, VersionedTransaction

import bot.jupiter as jupiter_module
import bot.main as main_module
from bot.config import Config
from bot.jupiter import SOL_MINT, SwapExecutor, TransactionStatus
from bot.portfolio import Portfolio
from bot.risk import RiskManager

# getattr so the module still imports for mutation runs against main
unsupported_tx_version = getattr(jupiter_module, "unsupported_tx_version", None)

PAYER = Pubkey.from_string("11111111111111111111111111111112")
PROGRAM = Pubkey.from_string("Stake11111111111111111111111111111111111111")
OTHER = Pubkey.from_string("Vote111111111111111111111111111111111111111")
BLOCKHASH = Hash.default()
MINT = "VersionGateMint111111111111111111111111111"


def _ix():
    return Instruction(PROGRAM, b"\x01", [AccountMeta(PAYER, True, True), AccountMeta(OTHER, False, True)])


def unsigned(message):
    return VersionedTransaction.populate(message, [Signature.default()])


def legacy_tx():
    return unsigned(Message.new_with_blockhash([_ix()], PAYER, BLOCKHASH))


def v0_tx():
    return unsigned(MessageV0.try_compile(PAYER, [_ix()], [], BLOCKHASH))


def v1_tx():
    return unsigned(MessageV1.try_compile(PAYER, [_ix()], BLOCKHASH))


def b64(tx):
    return base64.b64encode(bytes(tx)).decode()


# -- the solders API facts the gate relies on ------------------------------

def test_solders_api_distinguishes_versions():
    for tx, msg_type, version in ((legacy_tx(), Message, Legacy.Legacy),
                                  (v0_tx(), MessageV0, 0), (v1_tx(), MessageV1, 1)):
        decoded = VersionedTransaction.from_bytes(bytes(tx))
        assert type(decoded.message) is msg_type
        assert decoded.version() == version
    assert not isinstance(v1_tx().version(), Legacy)


# -- the pure gate --------------------------------------------------------------

def test_gate_allows_legacy_and_v0_only():
    assert unsupported_tx_version(legacy_tx()) is None
    assert unsupported_tx_version(v0_tx()) is None
    assert unsupported_tx_version(v1_tx()) == "refused before signing: unsupported transaction version 1"


class Shaped:
    def __init__(self, message, version):
        self.message, self._version = message, version

    def version(self):
        if isinstance(self._version, Exception):
            raise self._version
        return self._version


@pytest.mark.parametrize("obj", [
    Shaped(legacy_tx().message, 0),               # legacy message claiming v0
    Shaped(v0_tx().message, Legacy.Legacy),       # v0 message claiming legacy
    Shaped(v0_tx().message, False),               # bool is not version 0
    Shaped(v0_tx().message, 1),
    Shaped(v1_tx().message, 0),                   # v1 message claiming v0
    Shaped(object(), 0),
    Shaped(v0_tx().message, RuntimeError("boom")),
    object(),
    None,
])
def test_gate_refuses_anything_else_and_never_raises(obj):
    reason = unsupported_tx_version(obj)
    assert reason is not None and reason.startswith("refused before signing: unsupported transaction version")


# -- the executor boundary ------------------------------------------------------

class Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, swap_b64):
        self.swap_b64 = swap_b64

    def post(self, url, json=None, headers=None, timeout=None):
        assert url.endswith("/swap"), url  # only the /swap build request is expected
        return Resp({"swapTransaction": self.swap_b64})


def cfg():
    return Config(live_trading=True, wallet_private_key="test-only-sentinel",
                  burner_wallet_pubkey=str(PAYER), bankroll_usd=20.0, max_position_pct=10.0,
                  confirmation_timeout_seconds=0.0, process_lock_file=".test-lock",
                  state_file=".test-state")


@pytest.fixture
def boundary(monkeypatch):
    events = []

    class SpySigned:
        signatures = ["sig-test"]   # what the signed tx is known by on chain

        def __bytes__(self):
            return b"spy-signed"

    class SpyVersionedTransaction:
        """Real decoding; the signing constructor is a spy (no crypto)."""
        from_bytes = staticmethod(VersionedTransaction.from_bytes)

        def __new__(cls, message, signers):
            events.append(("sign", type(message).__name__))
            return SpySigned()

    spy_module = types.ModuleType("solders.transaction")
    spy_module.VersionedTransaction = SpyVersionedTransaction
    spy_module.Legacy = real_tx.Legacy
    monkeypatch.setitem(sys.modules, "solders.transaction", spy_module)
    monkeypatch.setattr(jupiter_module, "send_raw_transaction",
                        lambda session, rpc, raw: events.append(("send", raw)) or "sig-test")

    def make(swap_b64):
        ex = SwapExecutor(cfg(), session=Session(swap_b64))
        ex._load_keypair = lambda _kp: types.SimpleNamespace(pubkey=lambda: PAYER)
        ex._confirm_transaction = lambda client, sig: TransactionStatus.CONFIRMED_SUCCESS
        ex._extract_confirmed_fill = lambda *a: None
        return ex

    return types.SimpleNamespace(events=events, make=make)


def run(ex):
    # an armed sell must carry a pre-broadcast persistence callback (True: durable)
    quote = {"inputMint": MINT, "outputMint": SOL_MINT, "outAmount": "10"}
    return ex._execute_live(quote, 5, 10, input_mint=MINT, output_mint=SOL_MINT,
                            before_broadcast=lambda sig: True)


@pytest.mark.parametrize("build, kind", [(legacy_tx, "Message"), (v0_tx, "MessageV0")])
def test_supported_versions_reach_sign_and_send(boundary, build, kind):
    result = run(boundary.make(b64(build())))
    assert boundary.events[0] == ("sign", kind)
    assert boundary.events[1:] == [("send", b"spy-signed")]
    assert result.ok and result.tx_signature == "sig-test"
    assert result.status == TransactionStatus.CONFIRMED_SUCCESS


def test_v1_refused_before_signing(boundary):
    result = run(boundary.make(b64(v1_tx())))
    assert boundary.events == []                      # never signed, never sent
    assert not result.ok
    assert result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature is None
    assert result.error == "refused before signing: unsupported transaction version 1"


@pytest.mark.parametrize("payload", [
    "", "!!!not-base64!!!", base64.b64encode(b"\x00").decode(),
    base64.b64encode(bytes(range(200))).decode(),
    base64.b64encode(bytes(v0_tx())[:-5]).decode(),   # truncated v0
])
def test_malformed_transaction_fails_closed_before_signing(boundary, payload):
    result = run(boundary.make(payload))
    assert boundary.events == []
    assert result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature is None
    assert result.error == "refused before signing: undecodable swap transaction"


# -- #32 resolver treats a version refusal as a cleared, retryable exit --------

def test_version_refusal_clears_sell_intent_for_retry(boundary, monkeypatch):
    class Jup:
        quote_url, swap_url = "https://jupiter.test/quote", "https://jupiter.test/swap"

        def quote(self, i, o, amount):
            return {"inputMint": i, "outputMint": o, "inAmount": str(amount), "outAmount": "5000000",
                    "otherAmountThreshold": "4925000", "swapMode": "ExactIn", "slippageBps": 150}

        def _headers(self):
            return {}

    ex = boundary.make(b64(v1_tx()))
    ex.jup = Jup()
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg, bot.executor, bot.session = ex.cfg, ex, object()
    bot.portfolio, bot.risk = Portfolio(), RiskManager(ex.cfg)
    bot._save = lambda: None
    bot._reconciliation_required = False
    bot.notifier = types.SimpleNamespace(enabled=False)
    for name in ("get_signature_status", "get_confirmed_transaction", "fetch_wallet_token_balances"):
        monkeypatch.setattr(main_module, name,
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no chain lookup")))
    bot.portfolio.open(MINT, "V", 1.0, 2.0, 2.0, token_decimals=6,
                       raw_tokens=2_000_000, raw_tokens_initial=2_000_000)

    assert bot._settle_live_sell(bot.portfolio.positions[MINT], 1.0, 0.5, "stop_loss") is False
    intent = bot.portfolio.pending_intents[-1]
    assert (intent["status"], intent["outcome"], intent["tx_signature"]) == (
        "unresolved", "confirmed_failure", None)
    assert bot._reconciliation_required is False
    assert boundary.events == []

    bot._resolve_sell_intents()
    assert intent["status"] == "failed"
    assert intent["resolution"] == "rejected before send (no transaction)"
    assert bot.portfolio.positions[MINT].tokens == 2.0  # position kept; exit retried next tick
    assert not bot._sell_frozen(MINT)
