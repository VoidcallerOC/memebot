"""Read-only Solana JSON-RPC helpers used by confirmation, fill extraction,
and wallet reconciliation.

These calls never require ``solana.rpc.api.Client``. Live signing still uses
``solders``; broadcast and confirmation go through JSON-RPC so the bot can
run without the Solana Python client installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Config

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 12
SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
SOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000


@dataclass
class WalletTokenSnapshot:
    amounts: dict[str, float] = field(default_factory=dict)
    commitment: str = "confirmed"
    programs: tuple[str, ...] = ()


@dataclass
class FillRecord:
    actual_in_amount: Optional[int] = None
    actual_out_amount: Optional[int] = None
    fee_lamports: Optional[int] = None
    in_decimals: Optional[int] = None
    out_decimals: Optional[int] = None
    input_mint: Optional[str] = None
    output_mint: Optional[str] = None

    @property
    def complete(self) -> bool:
        return (
            self.actual_in_amount is not None
            and self.actual_out_amount is not None
            and self.actual_in_amount > 0
            and self.actual_out_amount > 0
        )


def rpc_call(session, rpc_url: str, method: str, params: list) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    resp = session.post(rpc_url, json=payload, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(body["error"])
    return body.get("result") if isinstance(body, dict) else None


def send_raw_transaction(session, rpc_url: str, raw_tx: bytes) -> str:
    import base64

    encoded = base64.b64encode(raw_tx).decode("ascii")
    result = rpc_call(
        session,
        rpc_url,
        "sendTransaction",
        [encoded, {"encoding": "base64", "skipPreflight": False}],
    )
    if not result:
        raise RuntimeError("sendTransaction returned empty signature")
    return str(result)


def get_signature_status(session, rpc_url: str, signature: str) -> Any:
    result = rpc_call(
        session,
        rpc_url,
        "getSignatureStatuses",
        [[signature], {"searchTransactionHistory": True}],
    )
    value = (result or {}).get("value") if isinstance(result, dict) else result
    if isinstance(value, list):
        return value[0] if value else None
    return None


def get_confirmed_transaction(
    session, rpc_url: str, signature: str, commitment: str
) -> Optional[dict]:
    result = rpc_call(
        session,
        rpc_url,
        "getTransaction",
        [
            signature,
            {
                "encoding": "jsonParsed",
                "commitment": commitment,
                "maxSupportedTransactionVersion": 0,
            },
        ],
    )
    return result if isinstance(result, dict) else None


def _token_units(entry: dict) -> tuple[Optional[int], Optional[int]]:
    amount_info = entry.get("uiTokenAmount") or entry.get("tokenAmount") or {}
    raw = amount_info.get("amount")
    decimals = amount_info.get("decimals")
    try:
        return (int(raw), int(decimals))
    except (TypeError, ValueError):
        return (None, None)


def _index_token_balances(balances: list, owner: str) -> dict[str, tuple[int, int]]:
    indexed: dict[str, tuple[int, int]] = {}
    for entry in balances or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("owner") != owner:
            continue
        mint = entry.get("mint")
        raw, decimals = _token_units(entry)
        if not mint or raw is None or decimals is None:
            continue
        prev = indexed.get(mint)
        if prev:
            indexed[mint] = (prev[0] + raw, decimals)
        else:
            indexed[mint] = (raw, decimals)
    return indexed


def _account_keys(tx: dict) -> list[str]:
    message = ((tx.get("transaction") or {}).get("message") or {})
    keys = message.get("accountKeys") or []
    out: list[str] = []
    for key in keys:
        if isinstance(key, str):
            out.append(key)
        elif isinstance(key, dict):
            out.append(str(key.get("pubkey") or key.get("publicKey") or ""))
        else:
            out.append(str(key))
    return out


def _native_delta(tx: dict, owner: str, fee_lamports: int) -> Optional[int]:
    meta = tx.get("meta") or {}
    pre = meta.get("preBalances") or []
    post = meta.get("postBalances") or []
    keys = _account_keys(tx)
    try:
        idx = keys.index(owner)
    except ValueError:
        return None
    if idx >= len(pre) or idx >= len(post):
        return None
    try:
        spent = int(pre[idx]) - int(post[idx])
    except (TypeError, ValueError):
        return None
    if spent <= 0:
        return None
    principal = spent - max(fee_lamports, 0)
    return principal if principal > 0 else spent


def extract_fill(
    tx: dict,
    owner: str,
    input_mint: str,
    output_mint: str,
) -> FillRecord:
    """Deterministically read wallet token/SOL deltas from a parsed transaction."""
    meta = tx.get("meta") or {}
    fee = meta.get("fee")
    try:
        fee_lamports = int(fee) if fee is not None else None
    except (TypeError, ValueError):
        fee_lamports = None

    pre = _index_token_balances(meta.get("preTokenBalances") or [], owner)
    post = _index_token_balances(meta.get("postTokenBalances") or [], owner)
    record = FillRecord(
        fee_lamports=fee_lamports,
        input_mint=input_mint,
        output_mint=output_mint,
    )

    if output_mint:
        pre_out = pre.get(output_mint, (0, post.get(output_mint, (0, 0))[1]))
        post_out = post.get(output_mint)
        if post_out is not None:
            delta = post_out[0] - pre_out[0]
            if delta > 0:
                record.actual_out_amount = delta
                record.out_decimals = post_out[1]

    if input_mint and input_mint != SOL_MINT:
        pre_in = pre.get(input_mint)
        post_in = post.get(input_mint, (0, pre_in[1] if pre_in else 0))
        if pre_in is not None:
            delta = pre_in[0] - post_in[0]
            if delta > 0:
                record.actual_in_amount = delta
                record.in_decimals = pre_in[1]
    elif input_mint == SOL_MINT:
        wsol_pre = pre.get(SOL_MINT)
        wsol_post = post.get(SOL_MINT, (0, 9))
        if wsol_pre is not None:
            delta = wsol_pre[0] - wsol_post[0]
            if delta > 0:
                record.actual_in_amount = delta
                record.in_decimals = 9
        if record.actual_in_amount is None:
            native = _native_delta(tx, owner, fee_lamports or 0)
            if native:
                record.actual_in_amount = native
                record.in_decimals = 9

    return record


def _accounts_to_amounts(accounts: list) -> dict[str, float]:
    balances: dict[str, float] = {}
    for acct in accounts or []:
        try:
            info = acct["account"]["data"]["parsed"]["info"]
            mint = info["mint"]
            ui = float(info["tokenAmount"]["uiAmount"] or 0)
        except (KeyError, TypeError, ValueError):
            continue
        balances[mint] = balances.get(mint, 0.0) + ui
    return balances


def fetch_wallet_token_balances(
    cfg: Config,
    pubkey: str,
    session=None,
) -> Optional[WalletTokenSnapshot]:
    """Read SPL Token + Token-2022 balances at the configured commitment."""
    if session is None:
        try:
            import requests
        except ImportError:
            return None
        session = requests.Session()

    commitment = cfg.confirmation_commitment or "confirmed"
    programs = (SPL_TOKEN_PROGRAM, TOKEN_2022_PROGRAM)
    merged: dict[str, float] = {}
    for program_id in programs:
        try:
            result = rpc_call(
                session,
                cfg.rpc_url,
                "getTokenAccountsByOwner",
                [
                    pubkey,
                    {"programId": program_id},
                    {"encoding": "jsonParsed", "commitment": commitment},
                ],
            )
        except Exception as exc:
            log.warning("token account fetch failed for %s: %s", program_id[:8], exc)
            return None
        accounts = (result or {}).get("value") if isinstance(result, dict) else None
        if accounts is None:
            return None
        for mint, amount in _accounts_to_amounts(accounts).items():
            merged[mint] = merged.get(mint, 0.0) + amount
    return WalletTokenSnapshot(
        amounts=merged,
        commitment=commitment,
        programs=programs,
    )
