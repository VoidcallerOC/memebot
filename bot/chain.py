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
        if not isinstance(ix, dict):
            return None  # malformed instruction: nothing is provable
        out.append(ix)
    for group in inner:
        if not isinstance(group, dict):
            return None
        for ix in group.get("instructions") or []:
            if not isinstance(ix, dict):
                return None
            out.append(ix)
    return out


# Jupiter v6 aggregator program: the swap route instruction of every swap the
# bot builds (Jupiter /swap, wrapAndUnwrapSol).
JUPITER_V6_PROGRAM = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"


def _swap_route_inner_ids(tx: dict) -> Optional[set[int]]:
    """``id()`` of every inner instruction executed under THE swap route.

    The route is the single top-level instruction invoking the Jupiter v6
    program; its CPIs are the inner group whose ``index`` is that
    instruction's position. Zero or several top-level route instructions,
    or a missing / duplicated inner group, return None (not provable). Only
    wSOL transfers inside this set are attributable to the swap.
    """
    message = ((tx.get("transaction") or {}).get("message") or {})
    tops = message.get("instructions")
    inner = (tx.get("meta") or {}).get("innerInstructions")
    if not isinstance(tops, list) or not isinstance(inner, list):
        return None
    routes = [i for i, ix in enumerate(tops)
              if isinstance(ix, dict) and ix.get("programId") == JUPITER_V6_PROGRAM]
    if len(routes) != 1:
        return None
    groups = [g for g in inner if isinstance(g, dict) and g.get("index") == routes[0]]
    if len(groups) != 1 or not isinstance(groups[0].get("instructions"), list):
        return None
    return {id(ix) for ix in groups[0]["instructions"] if isinstance(ix, dict)}


SPL_TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
_TOKEN_PROGRAM_IDS = {"spl-token": SPL_TOKEN_PROGRAM_ID, "spl-token-2022": TOKEN_2022_PROGRAM}
_MALFORMED = object()


def _ix_program_id(ix: dict, keys: list[str]) -> Optional[str]:
    """``programId`` (jsonParsed) or ``keys[programIdIndex]`` (compiled), else None."""
    pid = ix.get("programId")
    if isinstance(pid, str) and pid:
        return pid
    idx = ix.get("programIdIndex")
    if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(keys):
        return keys[idx]
    return None


def _ix_account_refs(ix: dict, keys: list[str]):
    """Every account an instruction references, normalised to pubkeys:
    ``accounts`` entries as strings, ``{"pubkey": ...}`` dicts or indices into
    ``keys``, plus string values of ``parsed.info``. ``_MALFORMED`` if any
    reference cannot be read (non-list ``accounts``, bad index, odd entry)."""
    refs: set[str] = set()
    accounts = ix.get("accounts", [])
    if not isinstance(accounts, list):
        return _MALFORMED
    for ref in accounts:
        if isinstance(ref, str):
            refs.add(ref)
        elif isinstance(ref, dict) and isinstance(ref.get("pubkey"), str):
            refs.add(ref["pubkey"])
        elif isinstance(ref, int) and not isinstance(ref, bool) and 0 <= ref < len(keys):
            refs.add(keys[ref])
        else:
            return _MALFORMED
    parsed = ix.get("parsed")
    if isinstance(parsed, dict):
        info = parsed.get("info")
        if isinstance(info, dict):
            refs.update(v for v in info.values() if isinstance(v, str))
        elif info is not None:
            return _MALFORMED
    return refs


def _opaque_token_touch(ix: object, keys: list[str], targets: set[str], decoded_labels) -> bool:
    """True if ``ix`` must fail the fill closed: it is (or cannot be ruled out
    as) a token-program instruction touching ``targets`` that the caller does
    NOT decode.

    Rule (raw ``data`` is never decoded):
    * Cleanly decoded = ``parsed`` is a dict with a dict ``info``, the
      ``program`` label is in ``decoded_labels`` and the program id (if
      present) is that label's token program. Those are left to the caller.
    * Otherwise, token program = label ``spl-token``/``spl-token-2022`` OR
      program id (``programId`` / ``programIdIndex``) SPL Token / Token-2022.
      Unknown program = no readable program id and no string label.
    * A token-program or unknown-program instruction whose account refs (see
      ``_ix_account_refs``) intersect ``targets``, or whose refs cannot be read
      at all, fails closed. One that touches none of ``targets`` is ignored, as
      is any instruction of a known non-token program.
    * A non-dict instruction fails closed.
    """
    if not isinstance(ix, dict):
        return True
    label = ix.get("program")
    pid = _ix_program_id(ix, keys)
    parsed = ix.get("parsed")
    if (isinstance(parsed, dict) and isinstance(parsed.get("info"), dict)
            and label in decoded_labels and pid in (None, _TOKEN_PROGRAM_IDS.get(label))):
        return False
    is_token = label in _TOKEN_PROGRAM_IDS or pid in _TOKEN_PROGRAM_IDS.values()
    unknown = pid is None and not isinstance(label, str)
    if not (is_token or unknown):
        return False
    refs = _ix_account_refs(ix, keys)
    return refs is _MALFORMED or bool(refs & targets)


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
    amounts into the owner's wSOL accounts minus amounts out of them, counted
    ONLY when executed as CPIs of the single Jupiter route instruction; any
    wSOL transfer touching the owner's wSOL accounts outside it returns None.

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
        route_ids = _swap_route_inner_ids(tx)
        if route_ids is None:
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
        n_in = n_out = 0  # owner wSOL transfers by direction (see below)
        for ix in ixs:
            if _opaque_token_touch(ix, keys, wsol, {"spl-token"}):
                return None  # undecoded token instruction on our wSOL account
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
                if id(ix) not in route_ids:
                    # A wSOL transfer touching our accounts that is not a CPI
                    # of the swap route is not provably part of the swap
                    # (it would still reconcile the balance equation and skew
                    # the amount): fail closed.
                    return None
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
                    n_in += 1
                elif src in wsol and dst not in wsol:
                    wsol_out += amount
                    n_out += 1
                else:
                    return None  # owner wSOL -> owner wSOL: not part of a sell
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
        # Route membership alone proves nothing: a sell has EXACTLY ONE wSOL
        # transfer touching the owner's wSOL accounts, pool -> owner, and no
        # owner wSOL outflow anywhere. Anything else is not provably the
        # swap's proceeds.
        if n_in != 1 or n_out != 0:
            return None
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
    out of the owner's wSOL accounts minus amounts into them, counted ONLY as
    CPIs of the single Jupiter route instruction (any other wSOL transfer
    touching the owner's wSOL accounts returns None). It excludes the
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
        route_ids = _swap_route_inner_ids(tx)
        if route_ids is None:
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
        n_in = n_out = 0  # owner wSOL transfers by direction (see below)
        for ix in ixs:
            if _opaque_token_touch(ix, keys, wsol, {"spl-token"}):
                return None  # undecoded token instruction on our wSOL account
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
                if id(ix) not in route_ids:
                    # A wSOL transfer touching our accounts that is not a CPI
                    # of the swap route is not provably part of the swap
                    # (it would still reconcile the balance equation and skew
                    # the amount): fail closed.
                    return None
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
                    n_out += 1
                elif dst in wsol and src not in wsol:
                    wsol_in += amount
                    n_in += 1
                else:
                    return None  # owner wSOL -> owner wSOL: not part of a buy
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
        # Route membership alone proves nothing: a buy has EXACTLY ONE wSOL
        # transfer touching the owner's wSOL accounts, owner -> pool, and no
        # owner wSOL inflow anywhere. Anything else is not provably the
        # swap's principal.
        if n_out != 1 or n_in != 0:
            return None
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


_TOKEN_PROGRAMS = frozenset({"spl-token", "spl-token-2022"})
# Token-program instructions that may touch an owner output-token account
# without moving its balance (account set-up during the swap).
_TOKEN_SETUP_TYPES = _WSOL_INIT_TYPES | frozenset({"initializeImmutableOwner", "getAccountDataSize"})


def _route_token_received(tx: dict, owner: str, mint: str) -> Optional[int]:
    """Raw units of ``mint`` a swap provably delivered to ``owner``, else None.

    The bought token gets the same strictness as wSOL. Across the WHOLE
    transaction (top-level and every inner group), the owner's ``mint`` token
    accounts (from pre/postTokenBalances, initializeAccount* and ATA create)
    must receive EXACTLY ONE ``transfer``/``transferChecked`` (SPL Token or
    Token-2022), executed as a CPI of the single top-level Jupiter route,
    from a non-owner account. Any other token-program instruction touching
    those accounts (an outflow, owner -> owner, mintTo, burn, closeAccount,
    ``transferCheckedWithFee``...) returns None. The owner's aggregate
    pre/post token delta for ``mint`` must EQUAL the transfer amount, so a
    Token-2022 transfer fee (received < sent) or any unseen movement fails
    closed. Never raises.
    """
    try:
        meta = tx.get("meta")
        if not isinstance(meta, dict) or meta.get("err") is not None or not mint:
            return None
        keys = _account_keys(tx)
        ixs = _parsed_instructions(tx)
        route_ids = _swap_route_inner_ids(tx)
        if ixs is None or route_ids is None:
            return None
        owned: set[str] = set()
        for entry in (meta.get("preTokenBalances") or []) + (meta.get("postTokenBalances") or []):
            if not isinstance(entry, dict) or entry.get("mint") != mint:
                continue
            if entry.get("owner") == owner:
                idx = entry.get("accountIndex")
                if not isinstance(idx, int) or not 0 <= idx < len(keys):
                    return None
                owned.add(keys[idx])
        for ix in ixs:
            parsed = ix.get("parsed")
            if not isinstance(parsed, dict):
                continue
            info = parsed.get("info") or {}
            kind = parsed.get("type")
            if ix.get("program") in _TOKEN_PROGRAMS and kind in _WSOL_INIT_TYPES:
                if info.get("mint") == mint and info.get("owner") == owner:
                    owned.add(info.get("account"))
            elif ix.get("program") == "spl-associated-token-account" and kind in _ATA_CREATE_TYPES:
                if info.get("mint") == mint and info.get("wallet") == owner:
                    owned.add(info.get("account"))
        owned.discard(None)
        if not owned or owner in owned:
            return None

        received: Optional[int] = None
        for ix in ixs:
            if _opaque_token_touch(ix, keys, owned, _TOKEN_PROGRAMS):
                return None  # undecoded/unlabelled token instruction on our account
            parsed = ix.get("parsed")
            if not isinstance(parsed, dict) or ix.get("program") not in _TOKEN_PROGRAMS:
                continue
            info = parsed.get("info")
            if not isinstance(info, dict):
                return None
            touched = {v for v in info.values() if isinstance(v, str)} & owned
            if not touched:
                continue
            kind = parsed.get("type")
            if kind in _TOKEN_SETUP_TYPES:
                continue
            if kind not in _TOKEN_TRANSFER_TYPES:
                return None  # mint/burn/close/fee-transfer/... on our account
            src, dst = info.get("source"), info.get("destination")
            if src in owned or dst not in owned:
                return None  # outflow or owner -> owner
            if id(ix) not in route_ids or received is not None:
                return None  # not a route CPI, or a second inflow
            if kind == "transferChecked":
                if info.get("mint") != mint:
                    return None
                amount = _ix_int((info.get("tokenAmount") or {}).get("amount"))
            else:
                amount = _ix_int(info.get("amount"))
            if not amount:
                return None
            received = amount
        if received is None:
            return None
        pre = _index_token_balances(meta.get("preTokenBalances") or [], owner).get(mint, (0, 0))
        post = _index_token_balances(meta.get("postTokenBalances") or [], owner).get(mint)
        if post is None or post[0] - pre[0] != received:
            return None
        return received
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
    (``meta.err`` set) or missing meta yields no amounts. For SOL output,
    ``actual_out_amount`` is set only from the proceeds proven by
    ``_native_sol_proceeds`` (temporary or persistent wSOL account); a
    non-zero owner wSOL token-balance delta must equal that exactly, else
    the fill is incomplete.
    For SOL input, ``actual_in_amount`` is set only from the exact principal
    proven by ``_native_sol_spent``; otherwise it is None (incomplete fill).
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

    if output_mint == SOL_MINT:
        # SOL out is accepted ONLY when ``_native_sol_proceeds`` proves it:
        # exactly one pool -> owner wSOL transfer as a CPI of the single
        # Jupiter route, no owner wSOL outflow, and every lamport of the owner
        # + its wSOL accounts reconciled. This covers a persistent (never
        # closed) wSOL account too: its lamports rise with its token balance.
        # The owner's wSOL token-balance delta is NOT a fallback (an unrelated
        # transfer into a persistent wSOL account inflates it); when non-zero
        # it must equal the proven amount EXACTLY, else the fill is unproven.
        native = _native_sol_proceeds(tx, owner)
        if native is not None:
            post_w = post.get(SOL_MINT)
            pre_w = pre.get(SOL_MINT)
            wsol_delta = (post_w[0] if post_w else 0) - (pre_w[0] if pre_w else 0)
            if wsol_delta in (0, native):
                record.actual_out_amount = native
                record.out_decimals = 9
    elif output_mint:
        # Bought token: only the amount ``_route_token_received`` proves
        # (exactly one route-CPI inflow from a non-owner account, nothing
        # else touching the owner's accounts of that mint, aggregate delta
        # equal). Never the bare balance delta: an unrelated incoming
        # transfer of the same mint would inflate it.
        got = _route_token_received(tx, owner, output_mint)
        post_out = post.get(output_mint)
        if got is not None and post_out is not None:
            record.actual_out_amount = got
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
        # SOL in is accepted ONLY when ``_native_sol_spent`` proves the exact
        # principal (parsed wSOL movements reconciled against every lamport
        # balance of the owner + its wSOL accounts). No balance-delta
        # fallback: an owner/wSOL net delta can be offset by other SOL moving
        # in the same transaction (an incoming transfer UNDERSTATES the
        # spend, rent OVERSTATES it). Unproven -> actual_in stays None, the
        # fill is incomplete, and a live buy stays unresolved.
        spent = _native_sol_spent(tx, owner)
        if spent is not None:
            record.actual_in_amount = spent
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
