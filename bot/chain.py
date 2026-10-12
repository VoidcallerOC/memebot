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


_WSOL_INIT_TYPES = frozenset({"initializeAccount", "initializeAccount2", "initializeAccount3"})
_ATA_CREATE_TYPES = frozenset({"create", "createIdempotent"})
_TOKEN_TRANSFER_TYPES = frozenset({"transfer", "transferChecked"})


def _parsed_instructions(tx: dict) -> Optional[list[dict]]:
    """Top-level then inner parsed instructions, or None if inner data is absent."""
    meta = tx.get("meta")
    inner = meta.get("innerInstructions") if isinstance(meta, dict) else None
    if not isinstance(inner, list):
        return None
    message = ((tx.get("transaction") or {}).get("message") or {})
    out: list[dict] = []
    for ix in message.get("instructions") or []:
        if isinstance(ix, dict):
            out.append(ix)
    for group in inner:
        if not isinstance(group, dict):
            return None
        for ix in group.get("instructions") or []:
            if isinstance(ix, dict):
                out.append(ix)
    return out


def _ix_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _native_sol_proceeds(tx: dict, owner: str) -> Optional[int]:
    """Lamports of SOL a token -> native SOL swap delivered to ``owner``.

    With ``wrapAndUnwrapSol`` Jupiter routes wSOL into an owner wSOL token
    account (often a temporary ATA created and closed in the same
    transaction, so absent from pre/postTokenBalances) and closes it to the
    owner. Proceeds are the parsed wSOL ``transfer``/``transferChecked``
    amounts into the owner's wSOL accounts minus amounts out of them.

    They are accepted only if the authoritative lamport balances agree
    EXACTLY. With S = the owner + the owner's wSOL accounts:

        delta_lamports(S) + fee(if owner pays) + rent(owner -> new non-wSOL
        accounts via createAccount) == wSOL in - wSOL out

    (createAccount / closeAccount between the owner and its own wSOL
    accounts are internal to S and cancel). Any other lamport movement into
    or out of S (an unrelated transfer, a tip, an unexpected refund) breaks
    the equality and returns None. Missing meta / innerInstructions, a
    failed transaction, an unknown owner, or no positive proceeds also
    return None. Never raises.
    """
    try:
        meta = tx.get("meta")
        if not isinstance(meta, dict) or meta.get("err") is not None:
            return None
        keys = _account_keys(tx)
        pre = meta.get("preBalances")
        post = meta.get("postBalances")
        if (
            not isinstance(pre, list) or not isinstance(post, list)
            or len(keys) != len(pre) or len(keys) != len(post)
            or owner not in keys or keys.count(owner) != 1
        ):
            return None
        fee = _ix_int(meta.get("fee"))
        if fee is None:
            return None
        ixs = _parsed_instructions(tx)
        if ixs is None:
            return None

        # The owner's wSOL token accounts: persistent ones from token balances
        # plus ones initialised / ATA-created for the owner in this tx.
        wsol: set[str] = set()
        for entry in (meta.get("preTokenBalances") or []) + (meta.get("postTokenBalances") or []):
            if not isinstance(entry, dict) or entry.get("mint") != SOL_MINT:
                continue
            idx = entry.get("accountIndex")
            if not isinstance(idx, int) or not 0 <= idx < len(keys):
                return None
            if entry.get("owner") == owner:
                wsol.add(keys[idx])
        for ix in ixs:
            parsed = ix.get("parsed")
            if not isinstance(parsed, dict):
                continue
            info = parsed.get("info") or {}
            kind = parsed.get("type")
            if ix.get("program") == "spl-token" and kind in _WSOL_INIT_TYPES:
                if info.get("mint") == SOL_MINT and info.get("owner") == owner:
                    wsol.add(info.get("account"))
            elif ix.get("program") == "spl-associated-token-account" and kind in _ATA_CREATE_TYPES:
                if info.get("mint") == SOL_MINT and info.get("wallet") == owner:
                    wsol.add(info.get("account"))
        wsol.discard(None)
        if not wsol or owner in wsol:
            return None
        group = wsol | {owner}

        wsol_in = wsol_out = rent_out = 0
        for ix in ixs:
            parsed = ix.get("parsed")
            if not isinstance(parsed, dict):
                continue
            info = parsed.get("info") or {}
            if not isinstance(info, dict):
                return None
            kind = parsed.get("type")
            program = ix.get("program")
            if program == "spl-token" and kind in _TOKEN_TRANSFER_TYPES:
                src, dst = info.get("source"), info.get("destination")
                if src not in wsol and dst not in wsol:
                    continue
                if kind == "transferChecked":
                    if info.get("mint") != SOL_MINT:
                        return None
                    amount = _ix_int((info.get("tokenAmount") or {}).get("amount"))
                else:
                    amount = _ix_int(info.get("amount"))
                if amount is None:
                    return None
                if dst in wsol and src not in wsol:
                    wsol_in += amount
                elif src in wsol and dst not in wsol:
                    wsol_out += amount
            elif program == "spl-token" and kind == "closeAccount":
                acct, dst = info.get("account"), info.get("destination")
                if acct in wsol and dst != owner:
                    return None  # our SOL refunded elsewhere
                if acct not in wsol and dst in group:
                    return None  # unexplained refund into the owner
            elif program == "spl-token" and kind == "syncNative":
                if info.get("account") in wsol:
                    return None  # wrapping inside a sell is not a sell proceed
            elif program == "system":
                if kind == "createAccount":
                    src, new = info.get("source"), info.get("newAccount")
                    lamports = _ix_int(info.get("lamports"))
                    if lamports is None:
                        return None
                    if src == owner and new in wsol:
                        continue  # rent into our own wSOL account: internal
                    if src == owner:
                        rent_out += lamports  # rent for another new account
                    elif new in group:
                        return None
                elif info.get("source") in group or info.get("destination") in group or \
                        info.get("newAccount") in group or info.get("account") in group:
                    return None  # any other system lamport move touching S
        proceeds = wsol_in - wsol_out
        if proceeds <= 0:
            return None
        delta = sum(int(post[i]) - int(pre[i]) for i, k in enumerate(keys) if k in group)
        fee_paid = fee if keys[0] == owner else 0
        if delta + fee_paid + rent_out != proceeds:
            return None
        return proceeds
    except (TypeError, ValueError, AttributeError):
        return None


def _native_sol_spent(tx: dict, owner: str) -> Optional[int]:
    """Lamports of SOL a native SOL -> token swap took from ``owner``.

    The buy-side mirror of ``_native_sol_proceeds``. With ``wrapAndUnwrapSol``
    Jupiter wraps SOL into an owner wSOL token account (system ``transfer``
    owner -> wSOL + ``syncNative``; often a temporary account created and
    closed in the same transaction) and the route takes wSOL out of it. The
    swap principal is the parsed wSOL ``transfer``/``transferChecked`` amount
    out of the owner's wSOL accounts minus amounts into them. It excludes the
    transaction fee and rent for new non-wSOL accounts (e.g. the output
    token's ATA), which the plain lamport delta (``_native_delta``) includes.

    Accepted only if the authoritative lamport balances agree EXACTLY. With
    S = the owner + the owner's wSOL accounts:

        -delta_lamports(S) == fee(if owner pays) + rent(owner -> new non-wSOL
        accounts via createAccount) + (wSOL out - wSOL in)

    Wrapping (owner -> own wSOL ``transfer``/``createAccount``), ``syncNative``
    and closing an owner wSOL account back to the owner are internal to S.
    Any other lamport movement into or out of S breaks the equality or is
    rejected and returns None, as do missing meta / innerInstructions, a
    failed transaction, an unknown owner or no positive principal. Never
    raises.
    """
    try:
        meta = tx.get("meta")
        if not isinstance(meta, dict) or meta.get("err") is not None:
            return None
        keys = _account_keys(tx)
        pre = meta.get("preBalances")
        post = meta.get("postBalances")
        if (
            not isinstance(pre, list) or not isinstance(post, list)
            or len(keys) != len(pre) or len(keys) != len(post)
            or owner not in keys or keys.count(owner) != 1
        ):
            return None
        fee = _ix_int(meta.get("fee"))
        if fee is None:
            return None
        ixs = _parsed_instructions(tx)
        if ixs is None:
            return None

        wsol: set[str] = set()
        for entry in (meta.get("preTokenBalances") or []) + (meta.get("postTokenBalances") or []):
            if not isinstance(entry, dict) or entry.get("mint") != SOL_MINT:
                continue
            idx = entry.get("accountIndex")
            if not isinstance(idx, int) or not 0 <= idx < len(keys):
                return None
            if entry.get("owner") == owner:
                wsol.add(keys[idx])
        for ix in ixs:
            parsed = ix.get("parsed")
            if not isinstance(parsed, dict):
                continue
            info = parsed.get("info") or {}
            kind = parsed.get("type")
            if ix.get("program") == "spl-token" and kind in _WSOL_INIT_TYPES:
                if info.get("mint") == SOL_MINT and info.get("owner") == owner:
                    wsol.add(info.get("account"))
            elif ix.get("program") == "spl-associated-token-account" and kind in _ATA_CREATE_TYPES:
                if info.get("mint") == SOL_MINT and info.get("wallet") == owner:
                    wsol.add(info.get("account"))
        wsol.discard(None)
        if not wsol or owner in wsol:
            return None
        group = wsol | {owner}

        wsol_in = wsol_out = rent_out = 0
        for ix in ixs:
            parsed = ix.get("parsed")
            if not isinstance(parsed, dict):
                continue
            info = parsed.get("info") or {}
            if not isinstance(info, dict):
                return None
            kind = parsed.get("type")
            program = ix.get("program")
            if program == "spl-token" and kind in _TOKEN_TRANSFER_TYPES:
                src, dst = info.get("source"), info.get("destination")
                if src not in wsol and dst not in wsol:
                    continue
                if kind == "transferChecked":
                    if info.get("mint") != SOL_MINT:
                        return None
                    amount = _ix_int((info.get("tokenAmount") or {}).get("amount"))
                else:
                    amount = _ix_int(info.get("amount"))
                if amount is None:
                    return None
                if src in wsol and dst not in wsol:
                    wsol_out += amount
                elif dst in wsol and src not in wsol:
                    wsol_in += amount
            elif program == "spl-token" and kind == "closeAccount":
                acct, dst = info.get("account"), info.get("destination")
                if acct in wsol and dst != owner:
                    return None  # our SOL sent elsewhere
                if acct not in wsol and dst in group:
                    return None  # unexplained refund into the owner
            elif program == "system":
                if kind == "createAccount":
                    src, new = info.get("source"), info.get("newAccount")
                    lamports = _ix_int(info.get("lamports"))
                    if lamports is None:
                        return None
                    if src == owner and new in wsol:
                        continue  # rent into our own wSOL account: internal
                    if src == owner:
                        rent_out += lamports  # rent for another new account (token ATA)
                    elif new in group:
                        return None
                elif kind == "transfer" and info.get("source") == owner \
                        and info.get("destination") in wsol:
                    continue  # wrapping into our own wSOL account: internal
                elif info.get("source") in group or info.get("destination") in group or \
                        info.get("newAccount") in group or info.get("account") in group:
                    return None  # any other system lamport move touching S
        spent = wsol_out - wsol_in
        if spent <= 0:
            return None
        delta = sum(int(post[i]) - int(pre[i]) for i, k in enumerate(keys) if k in group)
        fee_paid = fee if keys[0] == owner else 0
        if -delta != fee_paid + rent_out + spent:
            return None
        return spent
    except (TypeError, ValueError, AttributeError):
        return None


def extract_fill(
    tx: dict,
    owner: str,
    input_mint: str,
    output_mint: str,
) -> FillRecord:
    """Deterministically read wallet token/SOL deltas from a parsed transaction.

    Amounts are raw integer units (lamports for SOL). A failed transaction
    (``meta.err`` set) or missing meta yields no amounts. For SOL output the
    owner's persistent wSOL token-balance delta is used when positive;
    otherwise native SOL proceeds are derived by ``_native_sol_proceeds``
    (temporary wSOL account closed in the same transaction) or left None.
    For SOL input the same order applies on the input side: the persistent
    wSOL delta, else the exact principal from ``_native_sol_spent``, else
    the fee-adjusted lamport drop (``_native_delta``, may include rent).
    """
    meta = tx.get("meta")
    if not isinstance(meta, dict):
        return FillRecord(input_mint=input_mint, output_mint=output_mint)
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
    if meta.get("err") is not None:
        return record  # failed transaction: nothing was filled

    if output_mint:
        pre_out = pre.get(output_mint, (0, post.get(output_mint, (0, 0))[1]))
        post_out = post.get(output_mint)
        if post_out is not None:
            delta = post_out[0] - pre_out[0]
            if delta > 0:
                record.actual_out_amount = delta
                record.out_decimals = post_out[1]
        if record.actual_out_amount is None and output_mint == SOL_MINT:
            native = _native_sol_proceeds(tx, owner)
            if native is not None:
                record.actual_out_amount = native
                record.out_decimals = 9

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
            # Exact swap principal (excludes fee and new-account rent) when
            # the parsed instructions and lamport balances prove it.
            spent = _native_sol_spent(tx, owner)
            if spent is not None:
                record.actual_in_amount = spent
                record.in_decimals = 9
        if record.actual_in_amount is None:
            # Fallback: owner lamport drop minus fee. It can over-state the
            # principal by new-account rent, never under-state it.
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
