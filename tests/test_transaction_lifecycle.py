"""Phase 2 transaction lifecycle tests; no real wallet or transaction is used."""
from __future__ import annotations

import sys
import types

import pytest

from bot.config import Config
from bot.jupiter import JupiterClient, SwapExecutor, SwapResult, TransactionStatus
import bot.main as main_module
from bot.portfolio import Portfolio

SENTINEL = "test-only-sentinel"


def cfg(**overrides):
    values = dict(
        live_trading=True,
        wallet_private_key=SENTINEL,
        burner_wallet_pubkey="PUBKEY",
        confirmation_timeout_seconds=0.0,
        confirmation_poll_interval_seconds=0.01,
    )
    values.update(overrides)
    return Config(**values)


class StatusClient:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def get_signature_statuses(self, _signatures):
        if self.error:
            raise self.error
        return types.SimpleNamespace(value=[self.value])


def test_confirmation_success_requires_confirmed_status():
    executor = SwapExecutor(cfg())
    client = StatusClient({"err": None, "confirmationStatus": "confirmed"})
    assert executor._confirm_transaction(client, "sig") == TransactionStatus.CONFIRMED_SUCCESS


def test_processed_status_does_not_count_as_confirmed():
    executor = SwapExecutor(cfg(confirmation_timeout_seconds=0.0))
    client = StatusClient({"err": None, "confirmationStatus": "processed"})
    assert executor._confirm_transaction(client, "sig") == TransactionStatus.TIMEOUT


def test_confirmed_failure_is_distinguished():
    executor = SwapExecutor(cfg())
    client = StatusClient({"err": {"InstructionError": [0, "Custom"]}, "confirmationStatus": "confirmed"})
    assert executor._confirm_transaction(client, "sig") == TransactionStatus.CONFIRMED_FAILURE


def test_status_rpc_error_is_unknown():
    executor = SwapExecutor(cfg())
    assert executor._confirm_transaction(StatusClient(error=RuntimeError("status RPC unavailable")), "sig") == TransactionStatus.UNKNOWN


def test_confirmation_timeout_is_explicit():
    executor = SwapExecutor(cfg(confirmation_timeout_seconds=0.0))
    assert executor._confirm_transaction(StatusClient(None), "sig") == TransactionStatus.TIMEOUT


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"swapTransaction": "AA=="}


class FakeSession:
    def post(self, *_args, **_kwargs):
        return FakeResponse()


def install_fake_live_modules(monkeypatch, status):
    class FakeKeypair:
        def pubkey(self):
            return "PUBKEY"

    class RawTx:
        message = object()

        @classmethod
        def from_bytes(cls, _raw):
            return cls()

    class SignedTx:
        def __init__(self, *_args):
            pass

        def __bytes__(self):
            return b"signed"

    class FakeClient:
        def __init__(self, _rpc):
            pass

        def send_raw_transaction(self, _data):
            return types.SimpleNamespace(value="sig")

        def get_signature_statuses(self, _signatures):
            if isinstance(status, Exception):
                raise status
            return types.SimpleNamespace(value=[status])

    Versioned = type(
        "VersionedTransaction",
        (),
        {"from_bytes": RawTx.from_bytes, "__init__": SignedTx.__init__, "__bytes__": SignedTx.__bytes__},
    )
    modules = {
        "solders": types.ModuleType("solders"),
        "solders.keypair": types.ModuleType("solders.keypair"),
        "solders.transaction": types.ModuleType("solders.transaction"),
        "solana": types.ModuleType("solana"),
        "solana.rpc": types.ModuleType("solana.rpc"),
        "solana.rpc.api": types.ModuleType("solana.rpc.api"),
    }
    modules["solders.keypair"].Keypair = FakeKeypair
    modules["solders.transaction"].VersionedTransaction = Versioned
    modules["solana.rpc.api"].Client = FakeClient
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def run_live_execution(monkeypatch, status):
    install_fake_live_modules(monkeypatch, status)
    executor = SwapExecutor(cfg(), jup=JupiterClient(cfg()), session=FakeSession())
    executor._load_keypair = lambda _keypair: types.SimpleNamespace(pubkey=lambda: "PUBKEY")
    return executor._execute_live({"outAmount": "10"}, 5, 10)


def test_submission_signature_but_confirmation_failure_is_not_success(monkeypatch):
    result = run_live_execution(monkeypatch, {"err": {"failed": True}, "confirmationStatus": "confirmed"})
    assert not result.ok
    assert result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature == "sig"


def test_submission_signature_and_confirmed_success_is_explicit(monkeypatch):
    result = run_live_execution(monkeypatch, {"err": None, "confirmationStatus": "confirmed"})
    assert result.ok
    assert result.status == TransactionStatus.CONFIRMED_SUCCESS
    assert result.actual_out_amount is None


def test_submission_signature_but_timeout_is_not_success(monkeypatch):
    result = run_live_execution(monkeypatch, None)
    assert not result.ok
    assert result.status == TransactionStatus.TIMEOUT


def test_submission_status_rpc_error_is_unknown(monkeypatch):
    result = run_live_execution(monkeypatch, RuntimeError("status RPC unavailable"))
    assert not result.ok
    assert result.status == TransactionStatus.UNKNOWN


def test_unknown_result_requires_reconciliation_before_next_entry(monkeypatch):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg()
    bot._running = True
    bot._reconciliation_required = True
    bot.notifier = types.SimpleNamespace(enabled=False)
    bot.portfolio = Portfolio()
    bot.session = object()
    reconciles = []
    monkeypatch.setattr(main_module, "reconcile_wallet", lambda *args: reconciles.append(True) or None)
    bot._tick()
    assert bot._running is False
    assert reconciles == [True]


def test_unconfirmed_result_never_creates_portfolio_position():
    portfolio = Portfolio()
    result = SwapResult(ok=False, simulated=False, in_amount=10, out_amount=20, status=TransactionStatus.UNKNOWN)
    assert not result.ok
    assert portfolio.positions == {}


@pytest.mark.parametrize("status", [TransactionStatus.CONFIRMED_FAILURE, TransactionStatus.TIMEOUT, TransactionStatus.UNKNOWN])
def test_live_failed_or_unconfirmed_result_does_not_open_position(status):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg()
    bot.portfolio = Portfolio()
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint="MINT")])
    bot.screener = types.SimpleNamespace(screen=lambda _mint: types.SimpleNamespace(passed=True, price_usd=1.0, symbol="TEST"))
    bot.risk = types.SimpleNamespace(position_size_usd=lambda: 5.0)
    bot.executor = types.SimpleNamespace(swap=lambda *args, **kwargs: SwapResult(ok=False, simulated=False, in_amount=1, out_amount=1, status=status))
    bot.notifier = types.SimpleNamespace(buy=lambda *args: None)
    bot._usd_to_lamports = lambda _usd: 1
    bot._reconciliation_required = False
    bot._seek_entry()
    assert bot.portfolio.positions == {}
