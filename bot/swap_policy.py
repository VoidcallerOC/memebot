"""Direct-route swap policy: quote gate and pre-signing transaction validator.

Two fail-closed checks that sit between Jupiter and the signer:

* ``route_quote_error`` — before ``/swap`` is called: the quote must be a
  single-hop, 100%-allocation, SOL<->token direct route with no platform fee,
  echoing the requested mints, amount and slippage.
* ``swap_transaction_error`` — after ``/swap`` returns the unsigned
  transaction and BEFORE it is signed: every top-level instruction must be on
  an allowlist of (program, instruction type, account relationships) that
  matches the shape observed in real Jupiter Metis ``/swap/v1`` builds (see
  ``tests/fixtures/jupiter_dryrun/``). Anything else (unknown programs, extra
  SOL transfers, tips, platform-fee accounts, malformed indices, a second
  route instruction, ...) is refused.

Neither function signs, sends or does network I/O (address lookup tables
are resolved by the caller and passed in). Never raises.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

SOL_MINT = "So11111111111111111111111111111111111111112"
COMPUTE_BUDGET_PROGRAM = "ComputeBudget111111111111111111111111111111"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ATA_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
JUPITER_V6_PROGRAM = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
JUPITER_EVENT_AUTHORITY = "D8cy77BBepLMngZx6ZukaTff5hCt1HrWyKk3Hnd9oitf"
ADDRESS_LOOKUP_TABLE_PROGRAM = "AddressLookupTab1e1111111111111111111111111"
TOKEN_PROGRAMS = (SPL_TOKEN_PROGRAM, TOKEN_2022_PROGRAM)

# Anchor discriminator of Jupiter v6 ``route`` (sha256("global:route")[:8]),
# the only route instruction observed in the dry-run builds.
JUPITER_ROUTE_DISCRIMINATOR = bytes.fromhex("e517cb977ae3ad2a")
# ``route`` args end with in_amount u64, quoted_out_amount u64,
# slippage_bps u16, platform_fee_bps u8 (19 bytes); the single RoutePlanStep
# ends with percent u8, input_index u8, output_index u8.
_ROUTE_TAIL = 8 + 8 + 2 + 1
_MAX_COMPUTE_UNIT_LIMIT = 1_400_000
FULL_ALLOCATION_PERCENT = 100
FULL_ALLOCATION_BPS = 10_000


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def route_quote_error(
    quote: object, input_mint: str, output_mint: str, amount: int, slippage_bps: int,
) -> Optional[str]:
    """Why a quote violates the direct-route policy, or None. Never raises.

    Assumes the generic checks of ``jupiter.swap_quote_error`` (mints echo,
    ExactIn, inAmount == amount, positive outAmount) also run.
    """
    try:
        return _route_quote_error(quote, input_mint, output_mint, amount, slippage_bps)
    except Exception as exc:  # noqa: BLE001 - fail closed on anything odd
        return f"route policy check failed: {exc}"


def _route_quote_error(quote, input_mint, output_mint, amount, slippage_bps):
    if not isinstance(quote, dict):
        return "quote is not an object"
    if (input_mint == SOL_MINT) == (output_mint == SOL_MINT):
        return "route policy allows only SOL->token or token->SOL"
    if quote.get("inputMint") != input_mint or quote.get("outputMint") != output_mint:
        return "quote mints do not match request"
    if not _is_int(amount) or str(quote.get("inAmount")) != str(amount) or not isinstance(quote.get("inAmount"), str):
        return "quote inAmount does not match request"
    q_slip = quote.get("slippageBps")
    if not _is_int(q_slip) or not _is_int(slippage_bps) or q_slip != slippage_bps:
        return "quote slippageBps does not match configured slippage"
    if "platformFee" in quote and quote["platformFee"] is not None:
        fee = quote["platformFee"]
        if not isinstance(fee, dict) or set(fee) - {"amount", "feeBps"}:
            return "quote platformFee malformed"
        if fee.get("amount") not in (None, "0") or fee.get("feeBps") not in (None, 0) \
                or isinstance(fee.get("feeBps"), bool):
            return "quote carries a nonzero or ambiguous platformFee"
    plan = quote.get("routePlan")
    if not isinstance(plan, list):
        return "quote routePlan missing or not a list"
    if len(plan) != 1:
        return f"quote routePlan has {len(plan)} steps; exactly one direct step required"
    step = plan[0]
    if not isinstance(step, dict):
        return "routePlan step malformed"
    info = step.get("swapInfo")
    if not isinstance(info, dict):
        return "routePlan step swapInfo missing"
    if info.get("inputMint") != input_mint or info.get("outputMint") != output_mint:
        return "routePlan step mints do not match request"
    if not isinstance(info.get("ammKey"), str) or not info["ammKey"]:
        return "routePlan step ammKey missing"
    if str(info.get("inAmount")) != str(amount) or not isinstance(info.get("inAmount"), str):
        return "routePlan step inAmount does not match request"
    if info.get("outAmount") != quote.get("outAmount"):
        return "routePlan step outAmount does not match quote outAmount"
    percent, bps = step.get("percent"), step.get("bps")
    if percent is None and bps is None:
        return "routePlan step allocation missing (no percent or bps)"
    if percent is not None and (not _is_int(percent) or percent != FULL_ALLOCATION_PERCENT):
        return "routePlan step percent is not exactly 100"
    if bps is not None and (not _is_int(bps) or bps != FULL_ALLOCATION_BPS):
        return "routePlan step bps is not exactly 10000"
    return None


def _ata(owner: str, mint: str, token_program: str) -> Optional[str]:
    from solders.pubkey import Pubkey
    try:
        addr, _ = Pubkey.find_program_address(
            [bytes(Pubkey.from_string(owner)), bytes(Pubkey.from_string(token_program)),
             bytes(Pubkey.from_string(mint))],
            Pubkey.from_string(ATA_PROGRAM),
        )
    except Exception:  # noqa: BLE001
        return None
    return str(addr)


def _owner_atas(owner: str, mint: str) -> set:
    """The owner's ATA for ``mint`` under either token program (wSOL: classic only)."""
    programs = (SPL_TOKEN_PROGRAM,) if mint == SOL_MINT else TOKEN_PROGRAMS
    return {a for a in (_ata(owner, mint, p) for p in programs) if a}


def resolve_account_keys(message: Any, lookup_tables: Mapping[str, Sequence[str]]) -> Optional[list]:
    """Static keys + ALT-loaded writable + readonly keys, in Solana's order,
    or None if any table is missing or any index is out of range."""
    try:
        keys = [str(k) for k in message.account_keys]
        lookups = list(getattr(message, "address_table_lookups", None) or [])
        writable, readonly = [], []
        for lookup in lookups:
            table = lookup_tables.get(str(lookup.account_key))
            if table is None:
                return None
            for idx in bytes(lookup.writable_indexes):
                if idx >= len(table):
                    return None
                writable.append(str(table[idx]))
            for idx in bytes(lookup.readonly_indexes):
                if idx >= len(table):
                    return None
                readonly.append(str(table[idx]))
        return keys + writable + readonly
    except Exception:  # noqa: BLE001
        return None


def swap_transaction_error(
    message: Any,
    account_keys: Optional[Sequence[str]],
    *,
    owner: str,
    input_mint: str,
    output_mint: str,
    amount: int,
    quote: Mapping,
    slippage_bps: int,
    max_priority_fee_lamports: int,
) -> Optional[str]:
    """Why an unsigned Jupiter swap transaction must not be signed, or None."""
    try:
        return _swap_transaction_error(
            message, account_keys, owner, input_mint, output_mint, amount, quote,
            slippage_bps, max_priority_fee_lamports,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed
        return f"transaction validation failed: {exc}"


def _swap_transaction_error(message, keys, owner, input_mint, output_mint, amount,
                            quote, slippage_bps, max_fee):
    if keys is None:
        return "account keys unresolved (address lookup table unavailable)"
    keys = list(keys)
    if (input_mint == SOL_MINT) == (output_mint == SOL_MINT):
        return "only SOL->token or token->SOL swaps may be signed"
    buy = input_mint == SOL_MINT
    token_mint = output_mint if buy else input_mint
    if message.header.num_required_signatures != 1:
        return "transaction requires more than one signer"
    static = [str(k) for k in message.account_keys]
    if not static or static[0] != owner:
        return "fee payer is not the owner"
    wsol_ata = _ata(owner, SOL_MINT, SPL_TOKEN_PROGRAM)
    token_atas = _owner_atas(owner, token_mint)
    out_amount = int(quote["outAmount"])

    seen = {"limit": None, "price": None, "transfer": None, "sync": None,
            "route": None, "close": None}
    for pos, ix in enumerate(message.instructions):
        idxs = list(bytes(ix.accounts))
        if ix.program_id_index >= len(static) or any(i >= len(keys) for i in idxs):
            return f"instruction {pos}: account index out of range"
        program = keys[ix.program_id_index]
        accts = [keys[i] for i in idxs]
        data = bytes(ix.data)
        where = f"instruction {pos} ({program})"

        if program == COMPUTE_BUDGET_PROGRAM:
            if accts:
                return f"{where}: compute budget with accounts"
            if len(data) == 5 and data[0] == 2:
                slot = "limit"
            elif len(data) == 9 and data[0] == 3:
                slot = "price"
            else:
                return f"{where}: compute budget instruction not allowed"
            if seen[slot] is not None:
                return f"{where}: duplicate compute budget {slot}"
            seen[slot] = int.from_bytes(data[1:], "little")
        elif program == ATA_PROGRAM:
            if data != b"\x01":
                return f"{where}: only CreateIdempotent is allowed"
            if len(accts) != 6:
                return f"{where}: unexpected account count"
            payer, ata, wallet, mint, system, token_prog = accts
            if payer != owner or wallet != owner or system != SYSTEM_PROGRAM:
                return f"{where}: ATA create not for the owner"
            if token_prog not in TOKEN_PROGRAMS or mint not in (SOL_MINT, token_mint):
                return f"{where}: ATA create for an unexpected mint/program"
            if ata != _ata(owner, mint, token_prog):
                return f"{where}: ATA address is not the owner's ATA"
        elif program == SYSTEM_PROGRAM:
            if not buy:
                return f"{where}: SOL transfer in a sell"
            if seen["transfer"] is not None:
                return f"{where}: more than one SOL transfer"
            if len(data) != 12 or int.from_bytes(data[:4], "little") != 2:
                return f"{where}: only a system Transfer is allowed"
            if accts != [owner, wsol_ata]:
                return f"{where}: SOL transfer not owner -> owner's wSOL ATA"
            if int.from_bytes(data[4:12], "little") != amount:
                return f"{where}: wSOL funding amount does not match swap amount"
            seen["transfer"] = pos
        elif program == SPL_TOKEN_PROGRAM:
            if data == b"\x11":
                if not buy or seen["sync"] is not None or seen["transfer"] is None:
                    return f"{where}: unexpected SyncNative"
                if accts != [wsol_ata]:
                    return f"{where}: SyncNative not on the owner's wSOL ATA"
                seen["sync"] = pos
            elif data == b"\x09":
                if seen["close"] is not None or seen["route"] is None:
                    return f"{where}: unexpected CloseAccount"
                if accts != [wsol_ata, owner, owner]:
                    return f"{where}: CloseAccount not owner's wSOL ATA -> owner"
                seen["close"] = pos
            else:
                return f"{where}: token instruction not allowed"
        elif program == JUPITER_V6_PROGRAM:
            if seen["route"] is not None:
                return f"{where}: more than one Jupiter instruction"
            if buy and seen["sync"] is None:
                return f"{where}: route before wSOL funding"
            err = _route_ix_error(data, accts, owner, input_mint, output_mint, amount,
                                  out_amount, slippage_bps, wsol_ata, token_atas, buy)
            if err:
                return f"{where}: {err}"
            seen["route"] = pos
        else:
            return f"{where}: program not allowed"
    if seen["route"] is None:
        return "no Jupiter route instruction"
    if seen["close"] is None:
        return "wSOL account is not closed back to the owner"
    if buy and seen["transfer"] is None:
        return "buy without wSOL funding transfer"
    limit = seen["limit"] if seen["limit"] is not None else _MAX_COMPUTE_UNIT_LIMIT
    if limit > _MAX_COMPUTE_UNIT_LIMIT:
        return "compute unit limit above protocol maximum"
    price = seen["price"] or 0
    fee = -(-price * limit // 1_000_000)
    if fee > max_fee:
        return f"priority fee {fee} lamports exceeds cap {max_fee}"
    return None


def _route_ix_error(data, accts, owner, input_mint, output_mint, amount, out_amount,
                    slippage_bps, wsol_ata, token_atas, buy):
    if data[:8] != JUPITER_ROUTE_DISCRIMINATOR:
        return "Jupiter instruction is not the allowlisted route"
    if len(data) < 8 + 4 + 3 + _ROUTE_TAIL:
        return "route data too short"
    if int.from_bytes(data[8:12], "little") != 1:
        return "route plan is not a single step"
    tail = data[-_ROUTE_TAIL:]
    in_amt = int.from_bytes(tail[0:8], "little")
    quoted_out = int.from_bytes(tail[8:16], "little")
    slip = int.from_bytes(tail[16:18], "little")
    platform_fee_bps = tail[18]
    percent, in_idx, out_idx = data[-_ROUTE_TAIL - 3], data[-_ROUTE_TAIL - 2], data[-_ROUTE_TAIL - 1]
    if (percent, in_idx, out_idx) != (100, 0, 1):
        return "route step is not 100% from input to output"
    if in_amt != amount or quoted_out != out_amount:
        return "route amounts do not match the quote"
    if slip != slippage_bps:
        return "route slippage does not match configured slippage"
    if platform_fee_bps != 0:
        return "route carries a platform fee"
    if len(accts) < 9:
        return "route account list too short"
    src_ok = accts[2] == wsol_ata if buy else accts[2] in token_atas
    dst_ok = accts[3] in token_atas if buy else accts[3] == wsol_ata
    if accts[0] not in TOKEN_PROGRAMS or accts[1] != owner or not src_ok or not dst_ok:
        return "route source/destination are not the owner's accounts"
    if accts[4] != JUPITER_V6_PROGRAM or accts[6] != JUPITER_V6_PROGRAM:
        return "route sets a destination override or platform fee account"
    if accts[5] != output_mint:
        return "route destination mint does not match"
    if accts[7] != JUPITER_EVENT_AUTHORITY or accts[8] != JUPITER_V6_PROGRAM:
        return "route event authority/program accounts unexpected"
    return None
