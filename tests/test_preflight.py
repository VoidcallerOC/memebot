"""Read-only preflight tests; all wallet/RPC/Jupiter behavior is mocked."""
import json

from bot.config import Config
from bot.preflight import MAX_EXPERIMENT_USD, run_preflight
from bot.process_lock import ProcessLock, process_lock_available
from bot.reconcile import fetch_sol_balance
from bot.risk import RiskManager

TEST_ONLY_SENTINEL = "test-only-sentinel"


class FakeJupiter:
    quote_url = "https://jupiter.test/quote"
    swap_url = "https://jupiter.test/swap"


class FakeExecutor:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def swap(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("preflight must never call swap")


class FakeSafety:
    def __init__(self, *args, **kwargs):
        pass

    def screen(self, mint):
        raise AssertionError("preflight must not screen or trade a token")


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"result": {"value": 1_500_000_000}}


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


def cfg(**overrides):
    values = dict(
        live_trading=True,
        wallet_private_key=TEST_ONLY_SENTINEL,
        burner_wallet_pubkey="PUBKEY",
        bankroll_usd=20.0,
        max_position_pct=2.0,
        stop_loss_pct=15.0,
        require_liquidity_locked=True,
        require_mint_revoked=True,
        require_freeze_revoked=True,
        state_file="state.json",
    )
    values.update(overrides)
    return Config(**values)


def test_sol_balance_helper_only_reads_and_converts_lamports():
    session = FakeSession()
    balance = fetch_sol_balance(cfg(rpc_url="https://rpc.test"), "PUBKEY", session)
    assert balance == 1.5
    assert session.calls[0][1]["json"]["method"] == "getBalance"


def run(tmp_path, config=None, **overrides):
    defaults = dict(
        derive_pubkey=lambda _cfg: "PUBKEY",
        read_sol_balance=lambda _cfg, _pubkey, _session: 1.25,
        jupiter_factory=lambda _cfg, **kwargs: FakeJupiter(),
        swap_executor_factory=lambda _cfg, **kwargs: FakeExecutor(),
        risk_factory=RiskManager,
        safety_factory=lambda _cfg, **kwargs: FakeSafety(),
        conflict_checker=lambda: False,
        journal_path=str(tmp_path / "journal.log"),
    )
    defaults.update(overrides)
    return run_preflight(config or cfg(), **defaults)


def test_bankroll_zero_rejects(tmp_path):
    result = run(tmp_path, cfg(bankroll_usd=0))
    assert not result.eligible
    assert "EXPERIMENT_CEILING_EXCEEDED" in result.reason_codes


def test_bankroll_twenty_is_allowed_by_ceiling(tmp_path):
    result = run(tmp_path, cfg(bankroll_usd=20))
    assert result.experiment_ceiling_usd == 20.0
    assert result.prerequisites[5].passed  # experiment_ceiling


def test_bankroll_over_twenty_rejects(tmp_path):
    result = run(tmp_path, cfg(bankroll_usd=20.01))
    assert not result.eligible
    assert "EXPERIMENT_CEILING_EXCEEDED" in result.reason_codes


def test_proposed_allocation_twenty_is_allowed(tmp_path):
    result = run(tmp_path, proposed_allocation_usd=20.0)
    assert result.prerequisites[5].passed


def test_proposed_allocation_over_twenty_rejects(tmp_path):
    result = run(tmp_path, proposed_allocation_usd=20.01)
    assert not result.eligible
    assert "EXPERIMENT_CEILING_EXCEEDED" in result.reason_codes


def test_missing_burner_public_key_rejects(tmp_path):
    result = run(tmp_path, cfg(burner_wallet_pubkey=""))
    assert "SIGNER_VERIFICATION_BLOCKED" in result.reason_codes


def test_missing_signer_rejects(tmp_path):
    result = run(tmp_path, derive_pubkey=lambda _cfg: None)
    assert "SIGNER_VERIFICATION_BLOCKED" in result.reason_codes


def test_signer_public_key_mismatch_rejects(tmp_path):
    result = run(tmp_path, derive_pubkey=lambda _cfg: "OTHER")
    assert "BURNER_SIGNER_MISMATCH" in result.reason_codes


def test_missing_balance_verification_rejects(tmp_path):
    result = run(tmp_path, read_sol_balance=lambda *_args: None)
    assert "WALLET_BALANCE_UNVERIFIED" in result.reason_codes


def test_unavailable_rpc_rejects(tmp_path):
    result = run(tmp_path, read_sol_balance=lambda *_args: None)
    balance = next(p for p in result.prerequisites if p.name == "wallet_balance")
    assert not balance.passed


def test_unavailable_reconciliation_rejects(tmp_path):
    result = run(tmp_path, reconciliation_available=False)
    assert "RECONCILIATION_UNAVAILABLE" in result.reason_codes


def test_unavailable_jupiter_rejects(tmp_path):
    def broken_jupiter(*_args, **_kwargs):
        raise RuntimeError("unavailable")

    result = run(tmp_path, jupiter_factory=broken_jupiter)
    assert "JUPITER_UNAVAILABLE" in result.reason_codes


def test_disabled_risk_controls_reject(tmp_path):
    result = run(tmp_path, cfg(max_position_pct=0))
    assert "RISK_CONTROLS_UNAVAILABLE" in result.reason_codes


def test_disabled_safety_controls_reject(tmp_path):
    result = run(tmp_path, cfg(require_mint_revoked=False))
    assert "SAFETY_CONTROLS_UNAVAILABLE" in result.reason_codes


def test_unwritable_journal_rejects(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocker")
    result = run(tmp_path, journal_path=str(blocker / "journal.log"))
    assert "JOURNAL_UNAVAILABLE" in result.reason_codes


def test_conflicting_process_rejects(tmp_path):
    result = run(tmp_path, conflict_checker=lambda: True)
    assert "CONFLICTING_PROCESS" in result.reason_codes


def test_no_conflicting_process_passes_with_default_detector(tmp_path):
    lock_path = tmp_path / "memebot.lock"
    result = run(tmp_path, cfg(process_lock_file=str(lock_path)), conflict_checker=None)
    conflict = next(p for p in result.prerequisites if p.name == "conflicting_process")
    assert conflict.passed


def test_held_process_lock_fails_preflight(tmp_path):
    lock_path = tmp_path / "memebot.lock"
    holder = ProcessLock(str(lock_path))
    assert holder.acquire()
    try:
        result = run(tmp_path, cfg(process_lock_file=str(lock_path)), conflict_checker=None)
        conflict = next(p for p in result.prerequisites if p.name == "conflicting_process")
        assert not conflict.passed
        assert "CONFLICTING_PROCESS" in result.reason_codes
    finally:
        holder.release()


def test_stale_lock_file_is_not_treated_as_conflict(tmp_path):
    lock_path = tmp_path / "stale.lock"
    lock_path.write_text("stale state that is not used for liveness")
    assert process_lock_available(str(lock_path))


def test_lock_state_failure_fails_closed(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocker")
    assert not process_lock_available(str(blocker / "memebot.lock"))
    result = run(
        tmp_path,
        cfg(process_lock_file=str(blocker / "memebot.lock")),
        conflict_checker=None,
    )
    assert "CONFLICTING_PROCESS" in result.reason_codes


def test_process_lock_diagnostics_contain_no_wallet_secret(tmp_path, caplog):
    lock_path = tmp_path / "memebot.lock"
    holder = ProcessLock(str(lock_path))
    assert holder.acquire()
    try:
        result = run(tmp_path, cfg(process_lock_file=str(lock_path)), conflict_checker=None)
        rendered = json.dumps(result.to_dict()) + caplog.text
        assert TEST_ONLY_SENTINEL not in rendered
    finally:
        holder.release()


def test_failed_preflight_never_submits_transaction(tmp_path):
    FakeExecutor.calls = 0
    result = run(tmp_path, cfg(bankroll_usd=100))
    assert not result.eligible
    assert FakeExecutor.calls == 0


def test_secret_never_appears_in_output(tmp_path):
    result = run(tmp_path)
    rendered = json.dumps(result.to_dict())
    assert TEST_ONLY_SENTINEL not in rendered
    assert "PUBKEY" in rendered


def test_live_mode_is_not_changed_by_preflight(tmp_path):
    config = cfg(live_trading=False)
    run(tmp_path, config)
    assert config.live_trading is False


def test_ceiling_is_fixed_and_cannot_be_increased_by_strategy(tmp_path):
    result = run(tmp_path, cfg(bankroll_usd=20), proposed_allocation_usd=20.01)
    assert result.experiment_ceiling_usd == MAX_EXPERIMENT_USD == 20.0
    assert not result.eligible


def test_current_baseline_is_blocked_without_live_wallet_config(tmp_path):
    result = run(
        tmp_path,
        cfg(live_trading=False, wallet_private_key="", burner_wallet_pubkey="", bankroll_usd=100),
        derive_pubkey=lambda _cfg: None,
    )
    assert not result.eligible
    assert "LIVE_TRADING_DISABLED" in result.reason_codes
    assert "SIGNER_VERIFICATION_BLOCKED" in result.reason_codes
    assert "EXPERIMENT_CEILING_EXCEEDED" in result.reason_codes
