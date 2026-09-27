"""Phase 3 wallet token reconciliation — both token programs, no live wallet."""
from __future__ import annotations

from bot.chain import (
    SPL_TOKEN_PROGRAM,
    TOKEN_2022_PROGRAM,
    fetch_wallet_token_balances,
)
from bot.config import Config

WALLET = "Wallet1111111111111111111111111111111111111"
MINT = "Mint111111111111111111111111111111111111111"


def rpc_accounts(program_id: str, raw_amount: int):
    return {
        "programId": program_id,
        "raw": raw_amount,
    }


class Session:
    def __init__(self, specs):
        self.specs = list(specs)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        method = (json or {}).get("method")
        params = (json or {}).get("params") or []
        self.calls.append({"url": url, "method": method, "params": params})
        if method != "getTokenAccountsByOwner":
            raise AssertionError(f"unexpected method {method}")
        program_id = params[1]["programId"]
        commitment = params[2].get("commitment")
        spec = next(item for item in self.specs if item["programId"] == program_id)

        class Resp:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                raw = spec["raw"]
                value = []
                if raw:
                    value.append({
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "mint": MINT,
                                        "tokenAmount": {
                                            "amount": str(raw),
                                            "decimals": 6,
                                            "uiAmount": raw / 1_000_000,
                                        },
                                    }
                                }
                            }
                        }
                    })
                return {"result": {"value": value, "commitment": commitment}}

        return Resp()


def test_wallet_token_reconcile_reads_both_programs_with_commitment():
    session = Session([
        rpc_accounts(SPL_TOKEN_PROGRAM, 0),
        rpc_accounts(TOKEN_2022_PROGRAM, 25_000_000),
    ])
    snapshot = fetch_wallet_token_balances(
        Config(confirmation_commitment="finalized"), WALLET, session,
    )
    assert snapshot is not None
    assert snapshot.amounts == {MINT: 25.0}
    assert snapshot.commitment == "finalized"
    programs = [call["params"][1]["programId"] for call in session.calls]
    assert programs == [SPL_TOKEN_PROGRAM, TOKEN_2022_PROGRAM]
    assert all(call["params"][2]["commitment"] == "finalized" for call in session.calls)


def test_wallet_token_reconcile_merges_classic_and_token2022():
    session = Session([
        rpc_accounts(SPL_TOKEN_PROGRAM, 1_000_000),
        rpc_accounts(TOKEN_2022_PROGRAM, 2_000_000),
    ])
    snapshot = fetch_wallet_token_balances(Config(), WALLET, session)
    assert snapshot.amounts == {MINT: 3.0}
