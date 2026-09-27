"""Jupiter integration — price quotes and guarded swap execution."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import requests

from .config import MAX_EXPERIMENT_USD, Config

log = logging.getLogger(__name__)
HTTP_TIMEOUT = 12
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class TransactionStatus(str, Enum):
    SUBMITTED = "submitted"
    CONFIRMED_SUCCESS = "confirmed_success"
    CONFIRMED_FAILURE = "confirmed_failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class SwapResult:
    ok: bool
    simulated: bool
    in_amount: int
    out_amount: int
    tx_signature: Optional[str] = None
    error: Optional[str] = None
    status: TransactionStatus = TransactionStatus.UNKNOWN
    # Phase 3 supplies these from confirmed chain data; Phase 2 never invents them.
    actual_in_amount: Optional[int] = None
    actual_out_amount: Optional[int] = None
    fee_lamports: Optional[int] = None


def _status_from_rpc(value: Any, commitment: str) -> Optional[TransactionStatus]:
    if value is None:
        return None
    err = value.get("err") if isinstance(value, dict) else getattr(value, "err", None)
    if err is not None:
        return TransactionStatus.CONFIRMED_FAILURE
    confirmation = (
        value.get("confirmationStatus")
        if isinstance(value, dict)
        else getattr(value, "confirmation_status", None)
    )
    required = {"confirmed", "finalized"} if commitment != "finalized" else {"finalized"}
    return TransactionStatus.CONFIRMED_SUCCESS if confirmation in required else None


class JupiterClient:
    def __init__(self, cfg: Config, session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.session = session or requests.Session()

    @property
    def quote_url(self) -> str:
        return f"{self.cfg.jupiter_base_url}/quote"

    @property
    def swap_url(self) -> str:
        return f"{self.cfg.jupiter_base_url}/swap"

    def _headers(self) -> dict:
        return {"x-api-key": self.cfg.jupiter_api_key} if self.cfg.jupiter_api_key else {}

    def quote(self, input_mint: str, output_mint: str, amount: int) -> Optional[dict]:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(self.cfg.max_slippage_bps),
        }
        try:
            resp = self.session.get(self.quote_url, params=params, headers=self._headers(), timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data if data and "outAmount" in data else None
        except Exception as exc:
            log.debug("jupiter quote failed (%s->%s): %s", input_mint[:4], output_mint[:4], exc)
            return None


class SwapExecutor:
    """Turns a quote into a fill and resolves submitted transaction status."""

    def __init__(self, cfg: Config, jup: Optional[JupiterClient] = None,
                 session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.jup = jup or JupiterClient(cfg, self.session)

    def swap(self, input_mint: str, output_mint: str, amount: int,
             *, allocation_usd: Optional[float] = None) -> SwapResult:
        if self.cfg.is_armed and (
            not isinstance(allocation_usd, (int, float))
            or isinstance(allocation_usd, bool)
            or allocation_usd <= 0
            or allocation_usd > MAX_EXPERIMENT_USD
        ):
            return SwapResult(
                ok=False, simulated=False, in_amount=amount, out_amount=0,
                status=TransactionStatus.UNKNOWN,
                error=f"live allocation exceeds hard ${MAX_EXPERIMENT_USD:.2f} ceiling",
            )
        quote = self.jup.quote(input_mint, output_mint, amount)
        if quote is None:
            return SwapResult(
                ok=False, simulated=not self.cfg.is_armed, in_amount=amount,
                out_amount=0, status=TransactionStatus.CONFIRMED_FAILURE,
                error="no route / quote unavailable",
            )
        out_amount = int(quote["outAmount"])
        if not self.cfg.is_armed:
            log.info("[DRY-RUN] would swap %s of %s -> %s (out=%s)", amount, input_mint[:4], output_mint[:4], out_amount)
            return SwapResult(
                ok=True, simulated=True, in_amount=amount, out_amount=out_amount,
                tx_signature="DRYRUN", status=TransactionStatus.CONFIRMED_SUCCESS,
            )
        return self._execute_live(quote, amount, out_amount)

    def _execute_live(self, quote: dict, amount: int, out_amount: int) -> SwapResult:
        """Build, sign, submit, and explicitly resolve a live transaction."""
        try:
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction
            from solana.rpc.api import Client
        except ImportError:
            return SwapResult(
                ok=False, simulated=False, in_amount=amount, out_amount=out_amount,
                status=TransactionStatus.CONFIRMED_FAILURE,
                error="solana/solders not installed; cannot trade live",
            )
        try:
            keypair = self._load_keypair(Keypair)
            user_pubkey = str(keypair.pubkey())
            if not self.cfg.burner_wallet_pubkey or user_pubkey != self.cfg.burner_wallet_pubkey:
                return SwapResult(
                    ok=False, simulated=False, in_amount=amount, out_amount=out_amount,
                    status=TransactionStatus.CONFIRMED_FAILURE,
                    error="signer does not match BURNER_WALLET_PUBKEY",
                )
            swap_resp = self.session.post(
                self.jup.swap_url,
                json={
                    "quoteResponse": quote,
                    "userPublicKey": user_pubkey,
                    "wrapAndUnwrapSol": True,
                    "dynamicComputeUnitLimit": True,
                    "prioritizationFeeLamports": "auto",
                },
                headers=self.jup._headers(), timeout=HTTP_TIMEOUT,
            )
            swap_resp.raise_for_status()
            raw = __import__("base64").b64decode(swap_resp.json()["swapTransaction"])
            tx = VersionedTransaction.from_bytes(raw)
            signed = VersionedTransaction(tx.message, [keypair])
            client = Client(self.cfg.rpc_url)
            signature = str(client.send_raw_transaction(bytes(signed)).value)
            log.info("LIVE swap submitted: %s", signature)
            status = self._confirm_transaction(client, signature)
            if status == TransactionStatus.CONFIRMED_SUCCESS:
                return SwapResult(
                    ok=True, simulated=False, in_amount=amount, out_amount=out_amount,
                    tx_signature=signature, status=status,
                )
            return SwapResult(
                ok=False, simulated=False, in_amount=amount, out_amount=out_amount,
                tx_signature=signature, status=status, error=f"transaction {status.value}",
            )
        except Exception as exc:
            log.error("LIVE swap failed: %s", exc)
            return SwapResult(
                ok=False, simulated=False, in_amount=amount, out_amount=out_amount,
                status=TransactionStatus.UNKNOWN, error=str(exc),
            )

    def _confirm_transaction(self, client: Any, signature: str) -> TransactionStatus:
        """Poll signature status until the configured commitment resolves."""
        deadline = time.monotonic() + max(self.cfg.confirmation_timeout_seconds, 0.0)
        while True:
            try:
                response = client.get_signature_statuses([signature])
                value = getattr(response, "value", None)
                if value is None and isinstance(response, dict):
                    value = (response.get("result") or {}).get("value")
                status = _status_from_rpc(
                    value[0] if value else None, self.cfg.confirmation_commitment
                )
                if status is not None:
                    return status
            except Exception as exc:
                log.warning("transaction status query failed: %s", exc)
                return TransactionStatus.UNKNOWN
            if time.monotonic() >= deadline:
                return TransactionStatus.TIMEOUT
            time.sleep(max(self.cfg.confirmation_poll_interval_seconds, 0.01))

    def _load_keypair(self, Keypair):
        """Accept either a base58 secret key or a JSON byte-array."""
        key = self.cfg.wallet_private_key.strip()
        if key.startswith("["):
            import json
            return Keypair.from_bytes(bytes(json.loads(key)))
        import base58
        return Keypair.from_bytes(base58.b58decode(key))
