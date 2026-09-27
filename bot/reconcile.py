"""Wallet reconciliation — does the bot's tracked state match reality?

A persisted position file can drift from the actual on-chain wallet: a trade
landed that the bot didn't record (or vice-versa), a position was sold
manually, tokens were airdropped, or a crash happened mid-write. Trading on a
wrong picture of your holdings is how you accidentally sell tokens you don't
have or double-buy. On startup (when armed for live trading) the bot compares
what it *thinks* it holds against what the chain says, and warns loudly on any
drift.

The pure comparison logic (:func:`compare_holdings`) is separated from the RPC
fetch so it can be tested without a network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
except ImportError:  # reconciliation is best-effort
    requests = None  # type: ignore[assignment]

from .chain import (
    SPL_TOKEN_PROGRAM,
    TOKEN_2022_PROGRAM,
    fetch_wallet_token_balances,
)
from .config import Config

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 10

__all__ = [
    "ReconcileResult",
    "SPL_TOKEN_PROGRAM",
    "TOKEN_2022_PROGRAM",
    "compare_holdings",
    "fetch_onchain_balances",
    "fetch_sol_balance",
    "fetch_wallet_token_balances",
    "get_wallet_pubkey",
    "reconcile_wallet",
]


@dataclass
class ReconcileResult:
    phantom: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    drifted: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.phantom or self.untracked or self.drifted)

    def summary(self) -> str:
        if self.clean:
            return "wallet reconciliation clean — tracked state matches chain"
        parts = []
        if self.phantom:
            parts.append(f"{len(self.phantom)} phantom (tracked, not on chain)")
        if self.untracked:
            parts.append(f"{len(self.untracked)} untracked (on chain, not tracked)")
        if self.drifted:
            parts.append(f"{len(self.drifted)} drifted (amount mismatch)")
        return "wallet reconciliation MISMATCH: " + ", ".join(parts)


def compare_holdings(
    tracked: dict[str, float],
    onchain: dict[str, float],
    rel_tol: float = 0.02,
    dust: float = 1e-9,
) -> ReconcileResult:
    result = ReconcileResult()
    all_mints = set(tracked) | set(onchain)
    for mint in sorted(all_mints):
        t = tracked.get(mint, 0.0)
        o = onchain.get(mint, 0.0)
        t_zero = t <= dust
        o_zero = o <= dust
        if not t_zero and o_zero:
            result.phantom.append(mint)
        elif t_zero and not o_zero:
            result.untracked.append(mint)
        elif not t_zero and not o_zero:
            denom = max(abs(t), abs(o))
            if denom > 0 and abs(t - o) / denom > rel_tol:
                result.drifted.append((mint, t, o))
    return result


def get_wallet_pubkey(cfg: Config) -> Optional[str]:
    if not cfg.wallet_private_key:
        return None
    try:
        from solders.keypair import Keypair
    except ImportError:
        log.debug("solders not installed; cannot derive wallet pubkey")
        return None
    try:
        key = cfg.wallet_private_key.strip()
        if key.startswith("["):
            import json
            kp = Keypair.from_bytes(bytes(json.loads(key)))
        else:
            import base58
            kp = Keypair.from_bytes(base58.b58decode(key))
        return str(kp.pubkey())
    except Exception as exc:
        log.warning("could not derive wallet pubkey: %s", exc)
        return None


def fetch_onchain_balances(
    cfg: Config, pubkey: str, session=None
) -> Optional[dict[str, float]]:
    snapshot = fetch_wallet_token_balances(cfg, pubkey, session)
    if snapshot is None:
        return None
    return snapshot.amounts


def fetch_sol_balance(cfg: Config, pubkey: str, session=None) -> Optional[float]:
    if requests is None:
        return None
    session = session or requests.Session()
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "getBalance", "params": [pubkey],
    }
    try:
        resp = session.post(cfg.rpc_url, json=payload, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        value = (resp.json().get("result") or {}).get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) / 1_000_000_000.0
    except Exception as exc:
        log.warning("failed to fetch SOL balance: %s", exc)
        return None


def reconcile_wallet(cfg: Config, portfolio, session=None, notifier=None) -> Optional[ReconcileResult]:
    pubkey = get_wallet_pubkey(cfg)
    if pubkey is None:
        log.debug("reconciliation skipped (no wallet key available)")
        return None
    onchain = fetch_onchain_balances(cfg, pubkey, session)
    if onchain is None:
        log.warning("reconciliation skipped — could not read on-chain balances")
        return None

    tracked = {mint: pos.tokens for mint, pos in portfolio.positions.items()}
    result = compare_holdings(tracked, onchain)

    if result.clean:
        log.info(result.summary())
    else:
        log.warning(result.summary())
        for mint in result.phantom:
            log.warning("  PHANTOM: tracking %s but wallet holds ~0 "
                        "(was it sold elsewhere?)", mint[:8])
        for mint in result.untracked:
            log.warning("  UNTRACKED: wallet holds %s but bot isn't tracking it", mint[:8])
        for mint, t, o in result.drifted:
            log.warning("  DRIFT: %s tracked=%.4f on-chain=%.4f", mint[:8], t, o)
        if notifier is not None and getattr(notifier, "enabled", False):
            notifier.send("⚠️ " + result.summary())
    return result
