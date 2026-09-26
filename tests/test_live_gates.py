"""Phase 1 live-safety gate tests; no wallet or transaction is used."""
from __future__ import annotations

import sys
import types

from bot.config import Config
from bot.jupiter import JupiterClient, SwapExecutor
import bot.main as main_module


SENTINEL = "test-only-sentinel"


def armed_cfg(**overrides):
    values = dict(
        live_trading=True,
        wallet_private_key=SENTINEL,
        burner_wallet_pubkey="PUBKEY",
        bankroll_usd=20.0,
        process_lock_file=".test-lock",
        state_file=".test-state",
    )
    values.update(overrides)
    return Config(**values)


def test_live_startup_blocks_when_preflight_fails(monkeypatch):
    cfg = armed_cfg()
    calls = []

    class FakeLock:
        def __init__(self, path):
            calls.append(("lock-init", path))

        def acquire(self):
            calls.append("lock-acquire")
            return True

        def release(self):
            calls.append("lock-release")

    class FailedPreflight:
        eligible = False
        reason_codes = ["WALLET_BALANCE_UNVERIFIED"]

    monkeypatch.setattr(main_module, "load_config", lambda: cfg)
    monkeypatch.setattr(main_module, "ProcessLock", FakeLock)
    monkeypatch.setattr(main_module, "run_preflight", lambda *a, **kw: FailedPreflight())
    monkeypatch.setattr(
        main_module, "TradingBot", lambda _cfg: (_ for _ in ()).throw(AssertionError("bot must not start"))
    )

    assert main_module.main() == 4
    assert calls == [("lock-init", ".test-lock"), "lock-acquire", "lock-release"]


def test_live_allocation_above_twenty_is_rejected_before_execution(monkeypatch):
    cfg = armed_cfg(bankroll_usd=100.0)
    jup = JupiterClient(cfg)
    monkeypatch.setattr(jup, "quote", lambda *_args: {"outAmount": "123"})
    executor = SwapExecutor(cfg, jup=jup)
    monkeypatch.setattr(executor, "_execute_live", lambda *_args: (_ for _ in ()).throw(AssertionError("must not execute")))

    result = executor.swap("in", "out", 1, allocation_usd=100.0)
    assert not result.ok
    assert "20.00" in result.error


def test_live_signer_mismatch_blocks_before_swap_construction(monkeypatch):
    cfg = armed_cfg(burner_wallet_pubkey="EXPECTED")
    jup = JupiterClient(cfg)
    executor = SwapExecutor(cfg, jup=jup)

    class FakeKeypair:
        def pubkey(self):
            return "ACTUAL"

    class FakeVersionedTransaction:
        @classmethod
        def from_bytes(cls, _raw):
            raise AssertionError("transaction must not be constructed")

    class FakeClient:
        def __init__(self, _rpc):
            raise AssertionError("RPC client must not be created")

    modules = {
        "solders": types.ModuleType("solders"),
        "solders.keypair": types.ModuleType("solders.keypair"),
        "solders.transaction": types.ModuleType("solders.transaction"),
        "solana": types.ModuleType("solana"),
        "solana.rpc": types.ModuleType("solana.rpc"),
        "solana.rpc.api": types.ModuleType("solana.rpc.api"),
    }
    modules["solders.keypair"].Keypair = FakeKeypair
    modules["solders.transaction"].VersionedTransaction = FakeVersionedTransaction
    modules["solana.rpc.api"].Client = FakeClient
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(executor, "_load_keypair", lambda _keypair: FakeKeypair())
    result = executor._execute_live({"outAmount": "123"}, 1, 123)
    assert not result.ok
    assert result.error == "signer does not match BURNER_WALLET_PUBKEY"


def test_missing_burner_address_blocks_signer(monkeypatch):
    cfg = armed_cfg(burner_wallet_pubkey="")
    executor = SwapExecutor(cfg, jup=JupiterClient(cfg))

    class FakeKeypair:
        def pubkey(self):
            return "ACTUAL"

    monkeypatch.setattr(executor, "_load_keypair", lambda _keypair: FakeKeypair())
    monkeypatch.setitem(sys.modules, "solders", types.ModuleType("solders"))
    keypair_module = types.ModuleType("solders.keypair")
    tx_module = types.ModuleType("solders.transaction")
    solana_module = types.ModuleType("solana")
    rpc_module = types.ModuleType("solana.rpc")
    api_module = types.ModuleType("solana.rpc.api")
    keypair_module.Keypair = FakeKeypair
    tx_module.VersionedTransaction = object
    api_module.Client = object
    for name, module in {
        "solders.keypair": keypair_module,
        "solders.transaction": tx_module,
        "solana": solana_module,
        "solana.rpc": rpc_module,
        "solana.rpc.api": api_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    result = executor._execute_live({"outAmount": "123"}, 1, 123)
    assert not result.ok
    assert result.error == "signer does not match BURNER_WALLET_PUBKEY"


def test_live_startup_blocks_on_reconciliation_failure(monkeypatch):
    cfg = armed_cfg()
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg
    bot._running = True
    bot.notifier = types.SimpleNamespace(enabled=False)
    bot.portfolio = object()
    bot.session = object()
    ticked = []
    bot._tick = lambda: ticked.append(True)
    bot._banner = lambda: None
    monkeypatch.setattr(main_module, "reconcile_wallet", lambda *args: None)

    bot.run()
    assert ticked == []


def test_live_startup_blocks_on_reconciliation_mismatch(monkeypatch):
    cfg = armed_cfg()
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg
    bot._running = True
    bot.notifier = types.SimpleNamespace(enabled=False)
    bot.portfolio = object()
    bot.session = object()
    ticked = []
    bot._tick = lambda: ticked.append(True)
    bot._banner = lambda: None
    mismatch = types.SimpleNamespace(clean=False)
    monkeypatch.setattr(main_module, "reconcile_wallet", lambda *args: mismatch)

    bot.run()
    assert ticked == []


def test_dry_run_does_not_require_live_preflight_or_reconciliation(monkeypatch):
    cfg = Config(live_trading=False, wallet_private_key="", process_lock_file=".test-lock")
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg
    bot._running = False
    bot.notifier = types.SimpleNamespace(enabled=False)
    bot.portfolio = types.SimpleNamespace(open_count=0, realized_pnl=0.0)
    bot._banner = lambda: None
    bot._save = lambda: None
    bot._summary = lambda: None
    monkeypatch.setattr(main_module, "reconcile_wallet", lambda *args: (_ for _ in ()).throw(AssertionError("dry run must skip reconciliation")))

    bot.run()
