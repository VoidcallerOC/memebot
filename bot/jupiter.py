"""Jupiter integration — price quotes and swap execution.

Jupiter (https://dev.jup.ag/docs/swap-api/) aggregates Solana DEX liquidity.
We use it for two things:

  * quoting — what would I receive if I swapped X of token A for token B?
  * swapping — build, sign, and send the actual transaction (LIVE only).

The signing/sending path is fully implemented but is only reachable when the
config's safety lock is armed (``LIVE_TRADING=true`` + a wallet key). In
dry-run mode :meth:`SwapExecutor.swap` returns a simulated fill and touches no
funds.

Endpoint note: the legacy ``quote-api.jup.ag/v6`` host was deprecated by Jupiter
on 2025-10-01. The base URL is now configurable (``JUPITER_BASE_URL``) and
defaults to the keyless free tier ``https://lite-api.jup.ag/swap/v1``. Set a
``JUPITER_API_KEY`` and point ``JUPITER_BASE_URL`` at ``https://api.jup.ag/swap/v1``
for the higher-rate-limit paid tier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

from .config import Config

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 12

# Canonical mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@dataclass
class SwapResult:
    ok: bool
    simulated: bool
    in_amount: int           # base units of input token
    out_amount: int          # base units of output token
    tx_signature: Optional[str] = None
    error: Optional[str] = None


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
        # Jupiter's paid tier (api.jup.ag) authenticates via an x-api-key
        # header; the keyless lite tier ignores it.
        return {"x-api-key": self.cfg.jupiter_api_key} if self.cfg.jupiter_api_key else {}

    def quote(
        self, input_mint: str, output_mint: str, amount: int
    ) -> Optional[dict]:
        """Return Jupiter's best quote, or None if no route / error.
        ``amount`` is in the input token's base units."""
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(self.cfg.max_slippage_bps),
        }
        try:
            resp = self.session.get(
                self.quote_url, params=params, headers=self._headers(),
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data or "outAmount" not in data:
                return None
            return data
        except Exception as exc:
            log.debug("jupiter quote failed (%s->%s): %s", input_mint[:4], output_mint[:4], exc)
            return None


class SwapExecutor:
    """Turns a quote into a fill. Honors the safety lock."""

    def __init__(self, cfg: Config, jup: Optional[JupiterClient] = None,
                 session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.jup = jup or JupiterClient(cfg, self.session)

    def swap(self, input_mint: str, output_mint: str, amount: int) -> SwapResult:
        quote = self.jup.quote(input_mint, output_mint, amount)
        if quote is None:
            return SwapResult(ok=False, simulated=not self.cfg.is_armed,
                              in_amount=amount, out_amount=0,
                              error="no route / quote unavailable")

        out_amount = int(quote["outAmount"])

        # --- SAFETY LOCK -----------------------------------------------------
        # If we are not genuinely armed for live trading, stop here. We have a
        # real quote (so PnL tracking is realistic) but we spend nothing.
        if not self.cfg.is_armed:
            log.info("[DRY-RUN] would swap %s of %s -> %s (out=%s)",
                     amount, input_mint[:4], output_mint[:4], out_amount)
            return SwapResult(ok=True, simulated=True, in_amount=amount,
                              out_amount=out_amount, tx_signature="DRYRUN")

        return self._execute_live(quote, amount, out_amount)

    # -- live execution -----------------------------------------------------

    def _execute_live(self, quote: dict, amount: int, out_amount: int) -> SwapResult:
        """Build, sign and send the swap. Imports are local so that dry-run
        users never need solana/solders installed."""
        try:
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction
            from solana.rpc.api import Client
        except ImportError:
            return SwapResult(ok=False, simulated=False, in_amount=amount,
                              out_amount=out_amount,
                              error="solana/solders not installed; cannot trade live")

        try:
            keypair = self._load_keypair(Keypair)
            user_pubkey = str(keypair.pubkey())

            swap_resp = self.session.post(
                self.jup.swap_url,
                json={
                    "quoteResponse": quote,
                    "userPublicKey": user_pubkey,
                    "wrapAndUnwrapSol": True,
                    "dynamicComputeUnitLimit": True,
                    "prioritizationFeeLamports": "auto",
                },
                headers=self.jup._headers(),
                timeout=HTTP_TIMEOUT,
            )
            swap_resp.raise_for_status()
            swap_tx_b64 = swap_resp.json()["swapTransaction"]

            raw = __import__("base64").b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(raw)
            signed = VersionedTransaction(tx.message, [keypair])

            client = Client(self.cfg.rpc_url)
            sig = client.send_raw_transaction(bytes(signed)).value
            log.info("LIVE swap sent: %s", sig)
            return SwapResult(ok=True, simulated=False, in_amount=amount,
                              out_amount=out_amount, tx_signature=str(sig))
        except Exception as exc:
            log.error("LIVE swap failed: %s", exc)
            return SwapResult(ok=False, simulated=False, in_amount=amount,
                              out_amount=out_amount, error=str(exc))

    def _load_keypair(self, Keypair):
        """Accept either a base58 secret key or a JSON byte-array."""
        key = self.cfg.wallet_private_key.strip()
        if key.startswith("["):
            import json
            return Keypair.from_bytes(bytes(json.loads(key)))
        import base58
        return Keypair.from_bytes(base58.b58decode(key))
