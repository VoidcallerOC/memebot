"""Test helpers: policy-conformant (SYNTHETIC) Jupiter quotes and unsigned
swap transactions, shaped like the real dry-run builds in
``tests/fixtures/jupiter_dryrun/`` but without address lookup tables so no
RPC is needed. Not real on-chain data."""
from __future__ import annotations

from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from bot.swap_policy import (
    ATA_PROGRAM, COMPUTE_BUDGET_PROGRAM, JUPITER_EVENT_AUTHORITY,
    JUPITER_ROUTE_DISCRIMINATOR, JUPITER_V6_PROGRAM, SOL_MINT, SPL_TOKEN_PROGRAM,
    SYSTEM_PROGRAM, _ata,
)

FAKE_AMM = "FakeAmm111111111111111111111111111111111111"


def direct_route(quote: dict, slippage_bps: int = 150) -> dict:
    """``quote`` plus a single 100% direct step and the configured slippage."""
    q = dict(quote)
    q["slippageBps"] = slippage_bps
    q["platformFee"] = None
    q["routePlan"] = [{"swapInfo": {
        "ammKey": FAKE_AMM, "label": "Fake", "inputMint": q["inputMint"],
        "outputMint": q["outputMint"], "inAmount": q["inAmount"], "outAmount": q["outAmount"],
    }, "percent": 100, "bps": None}]
    return q


def route_data(amount: int, out_amount: int, slippage_bps: int, *, steps: int = 1,
               percent: int = 100, platform_fee_bps: int = 0,
               disc: bytes = JUPITER_ROUTE_DISCRIMINATOR) -> bytes:
    return (disc + steps.to_bytes(4, "little") + b"\x11\x01" + bytes([percent, 0, 1])
            + amount.to_bytes(8, "little") + out_amount.to_bytes(8, "little")
            + slippage_bps.to_bytes(2, "little") + bytes([platform_fee_bps]))


def conformant_instructions(owner: Pubkey, input_mint: str, output_mint: str, amount: int,
                            out_amount: int, slippage_bps: int = 150, *,
                            cu_limit: int = 200_000, cu_price: int = 1_000) -> list:
    pk = Pubkey.from_string
    o = str(owner)
    buy = input_mint == SOL_MINT
    token = output_mint if buy else input_mint
    wsol = pk(_ata(o, SOL_MINT, SPL_TOKEN_PROGRAM))
    tok_ata = pk(_ata(o, token, SPL_TOKEN_PROGRAM))
    cb = pk(COMPUTE_BUDGET_PROGRAM)

    def ata_create(ata, mint):
        return Instruction(pk(ATA_PROGRAM), b"\x01", [
            AccountMeta(owner, True, True), AccountMeta(ata, False, True),
            AccountMeta(owner, False, False), AccountMeta(pk(mint), False, False),
            AccountMeta(pk(SYSTEM_PROGRAM), False, False),
            AccountMeta(pk(SPL_TOKEN_PROGRAM), False, False)])

    ixs = [
        Instruction(cb, b"\x02" + cu_limit.to_bytes(4, "little"), []),
        Instruction(cb, b"\x03" + cu_price.to_bytes(8, "little"), []),
        ata_create(wsol, SOL_MINT),
    ]
    if buy:
        ixs += [
            Instruction(pk(SYSTEM_PROGRAM), (2).to_bytes(4, "little") + amount.to_bytes(8, "little"),
                        [AccountMeta(owner, True, True), AccountMeta(wsol, False, True)]),
            Instruction(pk(SPL_TOKEN_PROGRAM), b"\x11", [AccountMeta(wsol, False, True)]),
            ata_create(tok_ata, token),
        ]
    src, dst = (wsol, tok_ata) if buy else (tok_ata, wsol)
    jup = pk(JUPITER_V6_PROGRAM)
    ixs.append(Instruction(jup, route_data(amount, out_amount, slippage_bps), [
        AccountMeta(pk(SPL_TOKEN_PROGRAM), False, False), AccountMeta(owner, True, False),
        AccountMeta(src, False, True), AccountMeta(dst, False, True),
        AccountMeta(jup, False, False), AccountMeta(pk(output_mint), False, False),
        AccountMeta(jup, False, False), AccountMeta(pk(JUPITER_EVENT_AUTHORITY), False, False),
        AccountMeta(jup, False, False), AccountMeta(pk(FAKE_AMM), False, True)]))
    ixs.append(Instruction(pk(SPL_TOKEN_PROGRAM), b"\x09", [
        AccountMeta(wsol, False, True), AccountMeta(owner, False, True),
        AccountMeta(owner, True, False)]))
    return ixs


def unsigned_tx_bytes(owner: Pubkey, instructions: list) -> bytes:
    msg = MessageV0.try_compile(owner, instructions, [], Hash.new_unique())
    return bytes(VersionedTransaction.populate(msg, [Signature.default()]))


def conformant_unsigned_tx(owner: Pubkey, input_mint: str, output_mint: str, amount: int,
                           out_amount: int, slippage_bps: int = 150, **kw) -> bytes:
    return unsigned_tx_bytes(owner, conformant_instructions(
        owner, input_mint, output_mint, amount, out_amount, slippage_bps, **kw))
