"""Jupiter integration — price quotes and guarded swap execution."""
from __future__ import annotations

import logging
import math
import re
import time
from fractions import Fraction
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import requests

from .chain import (
    extract_fill,
    get_confirmed_transaction,
    get_signature_status,
    send_raw_transaction,
)
from .config import MAX_EXPERIMENT_USD, Config

log = logging.getLogger(__name__)
HTTP_TIMEOUT = 12
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# Raw-unit scales. Jupiter ``amount``/``inAmount``/``outAmount`` are integers
# in the smallest unit of their mint: SOL has 9 decimals (lamports), USDC 6.
SOL_DECIMALS = 9
USDC_DECIMALS = 6
LAMPORTS_PER_SOL = 10 ** SOL_DECIMALS
USDC_RAW_PER_USD = 10 ** USDC_DECIMALS
# An armed SOL-funded swap may spend at most this multiple of the caller's
# allocation (valued at an independently fetched SOL price) to absorb price
# movement between sizing and execution. It never lifts MAX_EXPERIMENT_USD.
LIVE_ALLOCATION_TOLERANCE = Fraction(105, 100)
# SPL mint decimals are a u8, but 10**decimals raw units must fit in a u64
# Jupiter amount, so anything above 18 cannot be quoted as one whole token.
MAX_QUOTABLE_DECIMALS = 18
_U64_MAX = 2 ** 64 - 1
_CANONICAL_RAW_INT = re.compile(r"^(0|[1-9][0-9]*)$")


def parse_raw_amount(raw: object) -> Optional[int]:
    """A canonical raw integer amount in ``0 .. 2**64-1``, or None.

    Accepts a canonical decimal string or a real ``int`` (not bool). Floats,
    signs, whitespace, underscores and exponents are rejected, never coerced.
    """
    if isinstance(raw, str):
        if _CANONICAL_RAW_INT.fullmatch(raw) is None:
            return None
        value = int(raw, 10)
    elif isinstance(raw, int) and not isinstance(raw, bool):
        value = raw
    else:
        return None
    if value < 0 or value > _U64_MAX:
        return None
    return value


def valid_token_decimals(value: object) -> Optional[int]:
    """Verified SPL decimals as an int in ``0 .. 18``; anything else is None.

    Unknown decimals are never guessed: callers treat None as "no mark".
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > MAX_QUOTABLE_DECIMALS:
        return None
    return value


def swap_quote_error(
    quote: object, input_mint: str, output_mint: str, amount: int,
) -> Optional[str]:
    """Why a quote must not be executed or booked, or None if it may.

    A swap quote must be a dict that echoes the requested ``inputMint`` and
    ``outputMint``, has ``inAmount`` exactly equal to the requested raw
    ``amount``, a positive canonical integer ``outAmount`` (via
    ``parse_raw_amount``), ``swapMode`` ExactIn when present, and, when
    present, a positive canonical integer ``otherAmountThreshold`` that does
    not exceed ``outAmount``. Never raises.
    """
    if not isinstance(quote, dict):
        return "quote is not an object"
    if quote.get("inputMint") != input_mint or quote.get("outputMint") != output_mint:
        return "quote mints do not match request"
    if "swapMode" in quote and quote["swapMode"] != "ExactIn":
        return "quote is not ExactIn"
    if parse_raw_amount(quote.get("inAmount")) != amount:
        return "quote inAmount does not match request"
    out_raw = parse_raw_amount(quote.get("outAmount"))
    if out_raw is None or out_raw <= 0:
        return "quote outAmount missing, malformed or not positive"
    if "otherAmountThreshold" in quote:
        min_out = parse_raw_amount(quote["otherAmountThreshold"])
        if min_out is None or min_out <= 0 or min_out > out_raw:
            return "quote otherAmountThreshold malformed or inconsistent"
    return None


def sol_usdc_raw_per_sol(jup: Any) -> Optional[int]:
    """USDC raw units (1e-6 USD) Jupiter returns for exactly 1 SOL, or None.

    Quotes exactly ``10**9`` lamports SOL->USDC and fails closed unless the
    quote is a dict whose ``inputMint``/``outputMint`` are SOL/USDC, whose
    ``inAmount`` is exactly ``10**9`` and whose ``outAmount`` is a positive
    canonical raw integer. There is no default/fallback price.
    """
    try:
        quote = jup.quote(SOL_MINT, USDC_MINT, LAMPORTS_PER_SOL)
    except Exception as exc:  # network/client failure -> no price
        log.debug("SOL/USD quote failed: %s", exc)
        return None
    if not isinstance(quote, dict):
        return None
    if quote.get("inputMint") != SOL_MINT or quote.get("outputMint") != USDC_MINT:
        return None
    if parse_raw_amount(quote.get("inAmount")) != LAMPORTS_PER_SOL:
        return None
    out_raw = parse_raw_amount(quote.get("outAmount"))
    if out_raw is None or out_raw <= 0:
        return None
    return out_raw


def sol_price_usd(jup: Any) -> Optional[float]:
    """Correctly scaled USD price of 1 SOL (``outAmount / 10**6``), or None."""
    out_raw = sol_usdc_raw_per_sol(jup)
    if out_raw is None:
        return None
    price = out_raw / USDC_RAW_PER_USD
    if not math.isfinite(price) or price <= 0:
        return None
    return float(price)


def unsupported_tx_version(tx: Any) -> Optional[str]:
    """Refusal reason unless ``tx`` is a legacy or v0 transaction, else None.

    Positive allowlist on the decoded solders object (0.29): legacy is a
    ``solders.message.Message`` with ``version() == Legacy.Legacy``; v0 is a
    ``MessageV0`` with ``version() == 0``. Anything else (e.g. ``MessageV1``
    / version 1, which ``get_confirmed_transaction`` cannot fetch with
    ``maxSupportedTransactionVersion: 0``), any disagreement between message
    type and version, or any error is refused. Never raises.
    """
    try:
        from solders.message import Message, MessageV0
        from solders.transaction import Legacy
        message = tx.message
        version = tx.version()
    except Exception as exc:  # noqa: BLE001 - fail closed on anything odd
        return f"refused before signing: unsupported transaction version (unreadable: {exc})"
    if type(message) is Message and version == Legacy.Legacy:
        return None
    if (
        type(message) is MessageV0 and isinstance(version, int)
        and not isinstance(version, bool) and version == 0
    ):
        return None
    return f"refused before signing: unsupported transaction version {version!r}"


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
    actual_in_amount: Optional[int] = None
    actual_out_amount: Optional[int] = None
    fee_lamports: Optional[int] = None
    in_decimals: Optional[int] = None
    out_decimals: Optional[int] = None
    fill_source: Optional[str] = None


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
        if self.cfg.is_armed and input_mint == SOL_MINT:
            exposure_error = self._armed_sol_exposure_error(amount, allocation_usd)
            if exposure_error is not None:
                log.error("live swap refused before signing: %s", exposure_error)
                return SwapResult(
                    ok=False, simulated=False, in_amount=amount, out_amount=0,
                    status=TransactionStatus.CONFIRMED_FAILURE,
                    error=exposure_error,
                )
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            return SwapResult(
                ok=False, simulated=not self.cfg.is_armed, in_amount=amount,
                out_amount=0, status=TransactionStatus.CONFIRMED_FAILURE,
                error="swap amount is not a positive integer raw amount",
            )
        quote = self.jup.quote(input_mint, output_mint, amount)
        if quote is None:
            return SwapResult(
                ok=False, simulated=not self.cfg.is_armed, in_amount=amount,
                out_amount=0, status=TransactionStatus.CONFIRMED_FAILURE,
                error="no route / quote unavailable",
            )
        # Validate before anything is booked, built, signed or sent. A bad
        # quote is the same clean failure as "no route" (nothing submitted).
        quote_error = swap_quote_error(quote, input_mint, output_mint, amount)
        if quote_error is not None:
            log.warning("swap refused before execution: %s", quote_error)
            return SwapResult(
                ok=False, simulated=not self.cfg.is_armed, in_amount=amount,
                out_amount=0, status=TransactionStatus.CONFIRMED_FAILURE,
                error=f"invalid quote: {quote_error}",
            )
        out_amount = parse_raw_amount(quote["outAmount"])
        if not self.cfg.is_armed:
            log.info("[DRY-RUN] would swap %s of %s -> %s (out=%s)", amount, input_mint[:4], output_mint[:4], out_amount)
            return SwapResult(
                ok=True, simulated=True, in_amount=amount, out_amount=out_amount,
                tx_signature="DRYRUN", status=TransactionStatus.CONFIRMED_SUCCESS,
            )
        return self._execute_live(
            quote, amount, out_amount,
            input_mint=input_mint, output_mint=output_mint,
        )

    def _armed_sol_exposure_error(self, amount: object, allocation_usd: float) -> Optional[str]:
        """Why an armed SOL-funded swap must not be signed, or None if it may.

        The caller-supplied ``allocation_usd`` is not trusted for exposure:
        the raw lamport amount is valued at an independently fetched, correctly
        scaled SOL price and must not exceed
        ``min(allocation_usd * 1.05, MAX_EXPERIMENT_USD)``. A missing or
        invalid price, or a non-positive/non-integer amount, fails closed.
        Nothing here signs, builds, or sends a transaction.
        """
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            return "live SOL amount is not a positive integer lamport amount"
        if not math.isfinite(allocation_usd):
            return "live allocation is not a finite USD amount"
        out_raw = sol_usdc_raw_per_sol(self.jup)
        if out_raw is None:
            return "SOL/USD price unavailable or invalid; live exposure unverifiable"
        actual_usd = Fraction(amount, LAMPORTS_PER_SOL) * Fraction(out_raw, USDC_RAW_PER_USD)
        limit_usd = min(
            Fraction(allocation_usd) * LIVE_ALLOCATION_TOLERANCE,
            Fraction(MAX_EXPERIMENT_USD),
        )
        if actual_usd > limit_usd:
            return (
                f"live SOL amount worth ${float(actual_usd):.2f} exceeds limit "
                f"${float(limit_usd):.2f} (allocation ${allocation_usd:.2f}, "
                f"hard ceiling ${MAX_EXPERIMENT_USD:.2f})"
            )
        return None

    def _execute_live(
        self,
        quote: dict,
        amount: int,
        out_amount: int,
        input_mint: Optional[str] = None,
        output_mint: Optional[str] = None,
    ) -> SwapResult:
        try:
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction
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
            swap_tx_b64 = swap_resp.json()["swapTransaction"]
            try:
                raw = __import__("base64").b64decode(swap_tx_b64, validate=True)
                tx = VersionedTransaction.from_bytes(raw)
            except Exception as exc:
                log.error("live swap refused before signing: undecodable transaction: %s", exc)
                return SwapResult(
                    ok=False, simulated=False, in_amount=amount, out_amount=out_amount,
                    status=TransactionStatus.CONFIRMED_FAILURE,
                    error="refused before signing: undecodable swap transaction",
                )
            version_error = unsupported_tx_version(tx)
            if version_error is not None:
                log.error("live swap %s", version_error)
                return SwapResult(
                    ok=False, simulated=False, in_amount=amount, out_amount=out_amount,
                    status=TransactionStatus.CONFIRMED_FAILURE, error=version_error,
                )
            signed = VersionedTransaction(tx.message, [keypair])
            signature = send_raw_transaction(self.session, self.cfg.rpc_url, bytes(signed))
            log.info("LIVE swap submitted: %s", signature)
            status = self._confirm_transaction(self.session, signature)
            if status != TransactionStatus.CONFIRMED_SUCCESS:
                return SwapResult(
                    ok=False, simulated=False, in_amount=amount, out_amount=out_amount,
                    tx_signature=signature, status=status, error=f"transaction {status.value}",
                )
            fill = self._extract_confirmed_fill(
                signature, user_pubkey, input_mint, output_mint,
            )
            return SwapResult(
                ok=True, simulated=False, in_amount=amount, out_amount=out_amount,
                tx_signature=signature, status=status,
                actual_in_amount=fill.actual_in_amount if fill else None,
                actual_out_amount=fill.actual_out_amount if fill else None,
                fee_lamports=fill.fee_lamports if fill else None,
                in_decimals=fill.in_decimals if fill else None,
                out_decimals=fill.out_decimals if fill else None,
                fill_source="getTransaction" if fill and fill.complete else None,
            )
        except Exception as exc:
            log.error("LIVE swap failed: %s", exc)
            return SwapResult(
                ok=False, simulated=False, in_amount=amount, out_amount=out_amount,
                status=TransactionStatus.UNKNOWN, error=str(exc),
            )

    def _extract_confirmed_fill(
        self,
        signature: str,
        owner: str,
        input_mint: Optional[str],
        output_mint: Optional[str],
    ):
        if not input_mint or not output_mint:
            return None
        try:
            tx = get_confirmed_transaction(
                self.session, self.cfg.rpc_url, signature, self.cfg.confirmation_commitment,
            )
        except Exception as exc:
            log.warning("getTransaction failed for %s: %s", signature, exc)
            return None
        if not tx:
            return None
        return extract_fill(tx, owner, input_mint, output_mint)

    def _confirm_transaction(self, client: Any, signature: str) -> TransactionStatus:
        deadline = time.monotonic() + max(self.cfg.confirmation_timeout_seconds, 0.0)
        while True:
            try:
                if hasattr(client, "get_signature_statuses"):
                    response = client.get_signature_statuses([signature])
                    value = getattr(response, "value", None)
                    if value is None and isinstance(response, dict):
                        value = (response.get("result") or {}).get("value")
                    raw = value[0] if value else None
                else:
                    raw = get_signature_status(client, self.cfg.rpc_url, signature)
                status = _status_from_rpc(raw, self.cfg.confirmation_commitment)
                if status is not None:
                    return status
            except Exception as exc:
                log.warning("transaction status query failed: %s", exc)
                return TransactionStatus.UNKNOWN
            if time.monotonic() >= deadline:
                return TransactionStatus.TIMEOUT
            time.sleep(max(self.cfg.confirmation_poll_interval_seconds, 0.01))

    def _load_keypair(self, Keypair):
        key = self.cfg.wallet_private_key.strip()
        if key.startswith("["):
            import json
            return Keypair.from_bytes(bytes(json.loads(key)))
        import base58
        return Keypair.from_bytes(base58.b58decode(key))
