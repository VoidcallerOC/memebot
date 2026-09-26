"""Fail-closed, read-only eligibility preflight.

The preflight is deliberately separate from trading execution.  It may inspect
configuration, instantiate existing safety/risk/execution components, derive a
public key locally, and issue read-only RPC balance requests.  It never quotes,
signs, submits, funds, or changes live-trading configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .config import MAX_EXPERIMENT_USD, Config, load_config
from .jupiter import JupiterClient, SwapExecutor
from .process_lock import process_lock_available
from .reconcile import fetch_sol_balance, get_wallet_pubkey
from .risk import RiskManager
from .safety import SafetyScreener

@dataclass(frozen=True)
class Prerequisite:
    name: str
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    eligible: bool
    reason_codes: list[str]
    wallet_address: Optional[str]
    derived_signer: Optional[str]
    observed_sol_balance: Optional[float]
    experiment_ceiling_usd: float
    prerequisites: list[Prerequisite]
    checked_at: str

    def to_dict(self) -> dict[str, Any]:
        # Deliberately serialize only public identities and safe check details.
        return {
            "eligible": self.eligible,
            "reason_codes": list(self.reason_codes),
            "wallet_address": self.wallet_address,
            "derived_signer": self.derived_signer,
            "observed_sol_balance": self.observed_sol_balance,
            "experiment_ceiling_usd": self.experiment_ceiling_usd,
            "prerequisites": [asdict(item) for item in self.prerequisites],
            "checked_at": self.checked_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def _check(name: str, passed: bool, reason: str = "", **details: Any) -> Prerequisite:
    return Prerequisite(name=name, passed=passed, reason=reason, details=details)


def _journal_is_writable(path: str) -> bool:
    """Verify temporary journal creation in the journal directory.

    The probe is deleted immediately and never contains credentials.  It does
    not overwrite the configured journal or state file.
    """
    directory = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    try:
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=".preflight-journal-", suffix=".tmp", delete=True,
        ) as fh:
            fh.write("preflight probe\n")
            fh.flush()
        return True
    except (OSError, ValueError):
        return False


def run_preflight(
    cfg: Config,
    *,
    session: Any = None,
    derive_pubkey: Callable[[Config], Optional[str]] = get_wallet_pubkey,
    read_sol_balance: Callable[..., Optional[float]] = fetch_sol_balance,
    jupiter_factory: Callable[..., Any] = JupiterClient,
    swap_executor_factory: Callable[..., Any] = SwapExecutor,
    risk_factory: Callable[[Config], Any] = RiskManager,
    safety_factory: Callable[..., Any] = SafetyScreener,
    reconciliation_available: bool = True,
    journal_path: Optional[str] = None,
    conflict_checker: Optional[Callable[[], bool]] = None,
    proposed_allocation_usd: Optional[float] = None,
    process_lock_held: bool = False,
) -> PreflightResult:
    """Run all checks without changing ``cfg`` or performing a transaction.

    ``conflict_checker`` follows the supported-detector convention: it returns
    True when a conflicting live process is present.  If omitted, the shared
    OS-level process lock is probed; lock-access failures fail closed.
    """
    checks: list[Prerequisite] = []
    reasons: list[str] = []
    wallet_address = cfg.burner_wallet_pubkey or None
    derived_signer: Optional[str] = None
    observed_sol_balance: Optional[float] = None

    def add(item: Prerequisite, code: Optional[str] = None) -> None:
        checks.append(item)
        if not item.passed and code:
            reasons.append(code)

    add(_check(
        "live_mode", cfg.live_trading,
        "LIVE_TRADING must be explicitly enabled; preflight never changes it",
    ), "LIVE_TRADING_DISABLED")

    credential_available = bool(cfg.wallet_private_key)
    add(_check(
        "wallet_credential", credential_available,
        "WALLET_PRIVATE_KEY is unavailable" if not credential_available else "",
    ), "SIGNER_VERIFICATION_BLOCKED")

    burner_available = bool(cfg.burner_wallet_pubkey)
    add(_check(
        "burner_public_identity", burner_available,
        "BURNER_WALLET_PUBKEY is unavailable" if not burner_available else "",
        wallet_address=wallet_address,
    ), "SIGNER_VERIFICATION_BLOCKED")

    if credential_available:
        try:
            derived_signer = derive_pubkey(cfg)
        except Exception:
            derived_signer = None
    signer_available = bool(derived_signer)
    add(_check(
        "signer_identity", signer_available,
        "signer public identity could not be derived" if not signer_available else "",
        derived_signer=derived_signer,
    ), "SIGNER_VERIFICATION_BLOCKED")

    exact_match = signer_available and burner_available and derived_signer == wallet_address
    match_reason = ""
    if not signer_available or not burner_available:
        match_reason = "signer and burner public identities are required"
    elif not exact_match:
        match_reason = "derived signer does not match BURNER_WALLET_PUBKEY"
    add(_check(
        "signer_public_key_match", exact_match,
        match_reason,
        wallet_address=wallet_address,
        derived_signer=derived_signer,
    ), "BURNER_SIGNER_MISMATCH" if signer_available and burner_available and not exact_match else "SIGNER_VERIFICATION_BLOCKED")

    allocation = cfg.bankroll_usd if proposed_allocation_usd is None else proposed_allocation_usd
    ceiling_ok = (
        isinstance(cfg.bankroll_usd, (int, float)) and cfg.bankroll_usd > 0
        and cfg.bankroll_usd <= MAX_EXPERIMENT_USD
        and isinstance(allocation, (int, float)) and allocation > 0
        and allocation <= MAX_EXPERIMENT_USD
    )
    add(_check(
        "experiment_ceiling", ceiling_ok,
        "bankroll and proposed allocation must be in (0, $20.00]" if not ceiling_ok else "",
        bankroll_usd=cfg.bankroll_usd,
        proposed_allocation_usd=allocation,
    ), "EXPERIMENT_CEILING_EXCEEDED")

    if exact_match:
        try:
            observed_sol_balance = read_sol_balance(cfg, wallet_address, session)
        except Exception:
            observed_sol_balance = None
    balance_ok = (
        isinstance(observed_sol_balance, (int, float))
        and not isinstance(observed_sol_balance, bool)
        and observed_sol_balance >= 0
    )
    add(_check(
        "wallet_balance", balance_ok,
        "native SOL balance could not be independently read" if not balance_ok else "",
        wallet_address=wallet_address,
        observed_sol_balance=observed_sol_balance,
    ), "WALLET_BALANCE_UNVERIFIED")

    try:
        risk = risk_factory(cfg)
        risk_ok = all(callable(getattr(risk, name, None)) for name in (
            "can_open_new_position", "position_size_usd", "evaluate_exit",
        )) and cfg.max_position_pct > 0 and cfg.stop_loss_pct > 0
    except Exception:
        risk_ok = False
    add(_check(
        "risk_controls", risk_ok,
        "existing risk controls are unavailable or disabled" if not risk_ok else "",
    ), "RISK_CONTROLS_UNAVAILABLE")

    safety_ok = False
    try:
        safety = safety_factory(cfg, session=session) if session is not None else safety_factory(cfg)
        safety_ok = all(callable(getattr(safety, name, None)) for name in ("screen",)) and all(
            getattr(cfg, name, False) for name in (
                "require_liquidity_locked", "require_mint_revoked", "require_freeze_revoked",
            )
        )
    except Exception:
        safety_ok = False
    add(_check(
        "safety_controls", safety_ok,
        "existing safety enforcement is unavailable or disabled" if not safety_ok else "",
    ), "SAFETY_CONTROLS_UNAVAILABLE")

    reconciliation_ok = reconciliation_available and callable(derive_pubkey) and callable(read_sol_balance)
    add(_check(
        "reconciliation", reconciliation_ok,
        "existing reconciliation implementation is unavailable" if not reconciliation_ok else "",
    ), "RECONCILIATION_UNAVAILABLE")

    try:
        jupiter = jupiter_factory(cfg, session=session) if session is not None else jupiter_factory(cfg)
        jupiter_ok = bool(getattr(jupiter, "quote_url", "")) and bool(getattr(jupiter, "swap_url", ""))
    except Exception:
        jupiter_ok = False
    add(_check(
        "jupiter_readiness", jupiter_ok,
        "Jupiter integration could not be initialized" if not jupiter_ok else "",
    ), "JUPITER_UNAVAILABLE")

    try:
        executor = swap_executor_factory(cfg, jup=jupiter, session=session) if session is not None else swap_executor_factory(cfg, jup=jupiter)
        accounting_ok = callable(getattr(executor, "swap", None))
    except Exception:
        accounting_ok = False
    add(_check(
        "actual_fill_accounting", accounting_ok,
        "existing execution/accounting implementation is unavailable" if not accounting_ok else "",
    ), "ACCOUNTING_UNAVAILABLE")

    journal_target = journal_path or cfg.state_file
    journal_ok = _journal_is_writable(journal_target)
    add(_check(
        "journal", journal_ok,
        "journal directory is not writable" if not journal_ok else "",
    ), "JOURNAL_UNAVAILABLE")

    if process_lock_held:
        conflict_ok = True
        conflict_reason = ""
    elif conflict_checker is None:
        try:
            conflict_ok = process_lock_available(cfg.process_lock_file)
            conflict_reason = (
                "conflicting process detected or process lock unavailable"
                if not conflict_ok else ""
            )
        except Exception:
            conflict_ok = False
            conflict_reason = "conflicting-process detector failed"
    else:
        try:
            conflict_present = bool(conflict_checker())
            conflict_ok = not conflict_present
            conflict_reason = "conflicting live process detected" if conflict_present else ""
        except Exception:
            conflict_ok = False
            conflict_reason = "conflicting-process detector failed"
    add(_check("conflicting_process", conflict_ok, conflict_reason), "CONFLICTING_PROCESS")

    return PreflightResult(
        eligible=not reasons and all(item.passed for item in checks),
        reason_codes=list(dict.fromkeys(reasons)),
        wallet_address=wallet_address,
        derived_signer=derived_signer,
        observed_sol_balance=observed_sol_balance,
        experiment_ceiling_usd=MAX_EXPERIMENT_USD,
        prerequisites=checks,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only safety preflight")
    parser.add_argument("--allocation-usd", type=float, default=None)
    args = parser.parse_args(argv)
    cfg = load_config()
    result = run_preflight(cfg, proposed_allocation_usd=args.allocation_usd)
    print(result.to_json())
    return 0 if result.eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
