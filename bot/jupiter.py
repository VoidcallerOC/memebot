"""Jupiter integration — price quotes and swap execution.

Jupiter (https://station.jup.ag/docs/apis/swap-api) aggregates Solana DEX
liquidity. We use it for two things:

  * quoting — what would I receive if I swapped X of token A for token B?
  * swapping — build, sign, and send the actual transaction (LIVE only).

The signing/sending path is fully implemented but is only reachable when the
config's safety lock is armed (``LIVE_TRADING=true`` + a wallet key). In
dry-run mode :meth:`SwapExecutor.swap` returns a simulated fill and touches no
funds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

from .config import Config

log = logging.getLogger(__name__)

JUP_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUP_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
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
            resp = self.session.get(JUP_QUOTE_URL, params=params, timeout=HTTP_TIMEOUT)
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
                JUP_SWAP_URL,
                json={
                    "quoteResponse": quote,
                    "userPublicKey": user_pubkey,
                    "wrapAndUnwrapSol": True,
                    "dynamicComputeUnitLimit": True,
                    "prioritizationFeeLamports": "auto",
                },
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
