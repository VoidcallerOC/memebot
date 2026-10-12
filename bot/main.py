"""The loop that ties everything together.

Each tick:
  1. Manage existing positions first (exits before entries — protect capital).
  2. If the daily loss limit isn't hit and we have room, look for one new
     entry, screen it for safety, and (paper or live) buy a risk-sized amount.

Run with: ``python -m bot.main``
"""
from __future__ import annotations

import copy
import logging
import math
import re
import signal
import sys
import time
from enum import Enum
from fractions import Fraction

import requests

from .alerts import Notifier
from .chain import (
    extract_fill, fetch_wallet_token_balances, get_confirmed_transaction,
    get_signature_status,
)
from .config import Config, load_config
from .jupiter import (
    JupiterClient, LAMPORTS_PER_SOL, SOL_MINT, USDC_MINT, USDC_RAW_PER_USD,
    SwapExecutor, TransactionStatus, parse_raw_amount, sol_usdc_raw_per_sol,
    valid_token_decimals,
)
from .preflight import run_preflight
from .portfolio import Portfolio, is_raw_units
from .paths import log_resolved_paths, with_resolved_data_paths
from .process_lock import ProcessLock
from .reconcile import reconcile_wallet
from .risk import RiskManager
from .safety import SafetyScreener
from .state import durable_saves_supported, load_state, save_state
from .strategy import build_strategy

try:
    from .decision.bridge import DecisionShadowBridge
except Exception:  # pragma: no cover - optional until package present
    DecisionShadowBridge = None  # type: ignore

log = logging.getLogger("bot")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


class ExitQuoteKind(Enum):
    """How a Jupiter quote may be used as an exit mark.

    ``MISSING`` is no quote body or no outAmount (timeout, error, empty).
    ``MALFORMED`` is an outAmount that is not a canonical raw integer in
    ``0 .. 2**64-1``. ``PRICE`` is an executable mark, including a
    legitimate numeric 0.0 from canonical ``"0"`` or int ``0``.
    """

    MISSING = "missing"
    MALFORMED = "malformed"
    PRICE = "price"


# Jupiter ``outAmount`` is the raw token amount in smallest units, not a
# decimal: ``JupiterClient.quote`` sends ``amount`` as that integer's decimal
# string, and ``SwapExecutor`` reads ``outAmount`` with ``int()``. A USDC
# quote of 1_000_000 raw input units is priced as ``outAmount / 1_000_000``.
# Solana SPL amounts are u64, and this client never shows a wider integer,
# so only 0 through 2**64-1 is executable.
_MAX_OUT_AMOUNT_RAW = 2**64 - 1
_CANONICAL_OUT_AMOUNT = re.compile(r"^(0|[1-9][0-9]*)$")

def _exact_fraction(fraction: object) -> Fraction | None:
    """Sell fraction as an exact rational, or None if invalid.

    Policy: a real number in (0, 1] (bools, NaN, inf and anything else are
    rejected) converted through its shortest round-trip decimal string,
    ``Fraction(repr(float(f)))``, so a configured ``0.3`` is exactly 3/10
    (not the binary double 0.29999999999999998889...). Binary-exact values
    such as the default ladder's 0.5 / 0.25 and the stop-loss 1.0 are
    unchanged by this conversion.
    """
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        return None
    value = float(fraction)
    if not math.isfinite(value) or value <= 0 or value > 1:
        return None
    return Fraction(repr(value))


def exact_raw_sell_amount(pos, fraction: object) -> tuple[int | None, str]:
    """Exact integer raw units to sell for ``fraction`` of the ORIGINAL buy.

    ``floor(raw_tokens_initial * fraction)`` (exact rational arithmetic,
    rounding toward zero so a sell never exceeds what the rule allows),
    capped at the current ``raw_tokens``; a full exit (fraction 1, the
    stop-loss) therefore sells exactly the current raw holding. Returns
    ``(None, reason)`` — and the caller refuses before submitting — for
    unverified decimals, a missing/malformed/inconsistent raw holding (e.g.
    a legacy position persisted before raw tracking), a holding outside the
    u64 range ``1 .. 2**64-1`` or above the amount bought, an invalid
    fraction, or a result of zero. Never derived from float token counts;
    nothing is clamped, truncated or repaired.
    """
    if valid_token_decimals(getattr(pos, "token_decimals", None)) is None:
        return None, "token decimals unverified"
    raw = getattr(pos, "raw_tokens", None)
    initial = getattr(pos, "raw_tokens_initial", None)
    if (
        isinstance(raw, bool) or not isinstance(raw, int)
        or isinstance(initial, bool) or not isinstance(initial, int)
    ):
        return None, (
            "no exact raw holding recorded (legacy or unverified position); operator "
            "action required: close or re-verify this position outside the bot"
        )
    if not (is_raw_units(raw) and is_raw_units(initial) and raw <= initial):
        return None, (
            f"raw holding inconsistent or out of range 1..2**64-1 (held={raw}, bought={initial})"
        )
    frac = _exact_fraction(fraction)
    if frac is None:
        return None, f"invalid sell fraction {fraction!r}"
    amount = min(initial * frac.numerator // frac.denominator, raw)
    if amount <= 0:
        return None, "sell amount rounds to zero raw units"
    if not is_raw_units(amount):  # unreachable given the checks above; never clamped
        return None, "sell amount out of range 1..2**64-1"
    return amount, ""


# Live sell intents (armed only). A sell intent is written before the swap and
# moves through these statuses (see ``TradingBot._resolve_sell_intents``):
#   pending    written before the swap; still pending only after a crash
#   blocked    refused before any send (missing decimals / zero size)
#   unresolved the swap did not confirm a settled fill (any non-success, or
#              success without fill data); ``outcome`` records the status
#   failed     terminal: evidence shows the sell did NOT execute; the position
#              is untouched and normal exit management (retry) resumes
#   confirmed_unbooked
#              the sell is proven on chain and its verified fill evidence
#              (actual in/out, fee) is persisted, but its accounting is not
#              yet; booked exactly once from that stored evidence (also after
#              a crash, at startup) with no further chain lookup
#   settled    terminal: the sell executed and was booked exactly once; the
#              booking and this status are persisted in the same atomic write
# Open intents freeze sells of that mint and block new entries; they never
# stop stop-loss / take-profit management of other positions.
OPEN_SELL_INTENT_STATUSES = frozenset({"pending", "unresolved", "blocked", "confirmed_unbooked"})
TERMINAL_SELL_INTENTS_KEPT = 50


def _rung_value(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _take_profit_rung(reason: object) -> float | None:
    """The ladder gain-target of a ``take_profit:<gain>`` reason, else None."""
    if not isinstance(reason, str) or not reason.startswith("take_profit:"):
        return None
    try:
        return _rung_value(float(reason.split(":", 1)[1]))
    except ValueError:
        return None


def _intent_rung(intent: dict) -> float | None:
    """The take-profit rung a sell intent is for. New intents carry
    ``rung``; a legacy intent (created when rungs were consumed at the
    threshold, so its rung is already marked) falls back to its reason, which
    only re-adds the same rung (idempotent)."""
    rung = _rung_value(intent.get("rung"))
    return rung if rung is not None else _take_profit_rung(intent.get("reason"))
# A Solana transaction's recent blockhash is valid for 150 slots (~60-90 s;
# solana.com docs "Transactions" / "Confirmation & Expiration"). After this
# conservative wall-clock window an unconfirmed transaction can no longer land,
# so an unchanged on-chain balance is evidence the sell did not execute.
SELL_INTENT_EXPIRY_SECONDS = 300


# Pre-buy sellability gate: the slippage-adjusted SOL -> token -> SOL round
# trip for the exact planned entry may lose at most this fraction of the input
# lamports. Both legs use Jupiter's ExactIn ``otherAmountThreshold`` (minimum
# output after ``slippageBps``), never the optimistic ``outAmount``. Exactly
# 10% passes; anything worse is rejected. Compared in integers
# (sell_min_out * 10 >= lamports * 9); deliberately a constant, not a config knob,
# so nothing can loosen it. A quote is a routing estimate, not a simulation:
# passing this gate does NOT guarantee a future sell will succeed.
MAX_ROUNDTRIP_LOSS = Fraction(1, 10)


def _parse_out_amount_usd(raw: object) -> float | None:
    """USD value of a USDC ``outAmount``, or None when it is not executable.

    Accept only a canonical decimal integer string (``^(0|[1-9][0-9]*)$``)
    or a real Python ``int`` (bool is not an int here), each in
    ``0 .. 2**64-1``. Signed zero, a leading ``+``, underscores, surrounding
    whitespace, decimals, and scientific notation are rejected as written —
    they are not stripped or rewritten into a price. Canonical ``"0"`` and
    int ``0`` return 0.0.
    """
    if isinstance(raw, str):
        if _CANONICAL_OUT_AMOUNT.fullmatch(raw) is None:
            return None
        try:
            amount = int(raw, 10)
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(raw, int) and not isinstance(raw, bool):
        amount = raw
    else:
        return None
    if amount < 0 or amount > _MAX_OUT_AMOUNT_RAW:
        return None
    try:
        price = amount / 1_000_000
    except OverflowError:
        return None
    if not math.isfinite(price):
        return None
    return float(price)


def _is_executable_mark(price: object) -> bool:
    """True only for a finite USD mark >= 0, including a legitimate 0.0."""
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        return False
    return math.isfinite(price) and price >= 0


def _classify_exit_quote(quote: object) -> tuple[ExitQuoteKind, float | None]:
    """Split a Jupiter quote into missing, malformed, or an executable price."""
    if quote is None:
        return ExitQuoteKind.MISSING, None
    if not isinstance(quote, dict):
        return ExitQuoteKind.MALFORMED, None
    if "outAmount" not in quote:
        return ExitQuoteKind.MISSING, None
    price = _parse_out_amount_usd(quote["outAmount"])
    if price is None or not _is_executable_mark(price):
        return ExitQuoteKind.MALFORMED, None
    return ExitQuoteKind.PRICE, price


def _price_usd(jup: JupiterClient, mint: str, decimals: int | None) -> float | None:
    """USD mark for 1 WHOLE token, or None when that mark must not be traded.

    ``decimals`` must be the mint's verified SPL decimals (from the safety
    screener's getAccountInfo mint fetch or the confirmed fill). Exactly
    ``10**decimals`` raw units (one whole token) are quoted to USDC, so the
    USDC ``outAmount / 10**6`` is the whole-token price for 6- and 9-decimal
    mints alike. Unknown/invalid decimals return None (no mark) and are never
    guessed.

    None also covers a missing quote, a malformed outAmount (not a canonical
    raw integer in 0 .. 2**64-1), and a quote whose ``inAmount`` or mints, when
    present, disagree with the request. A parsed 0.0 from canonical ``"0"`` or
    int ``0`` is returned as 0.0 so a legitimate zero quote still follows
    the exit rules.
    """
    decimals = valid_token_decimals(decimals)
    if decimals is None:
        log.warning("no verified decimals for %s — no mark", mint)
        return None
    raw_in = 10 ** decimals
    quote = jup.quote(mint, USDC_MINT, raw_in)
    if isinstance(quote, dict) and (
        ("inAmount" in quote and parse_raw_amount(quote["inAmount"]) != raw_in)
        or ("inputMint" in quote and quote["inputMint"] != mint)
        or ("outputMint" in quote and quote["outputMint"] != USDC_MINT)
    ):
        log.warning("inconsistent exit quote for %s — not an executable price", mint)
        return None
    kind, price = _classify_exit_quote(quote)
    if kind is ExitQuoteKind.MALFORMED:
        log.warning("malformed exit quote for %s — not an executable price", mint)
        return None
    if kind is not ExitQuoteKind.PRICE or not _is_executable_mark(price):
        return None
    return float(price)


def _validated_route_out(
    quote: object, input_mint: str, output_mint: str, in_amount: int,
) -> tuple[int | None, str]:
    """Slippage-adjusted minimum output of an ExactIn quote, or a reason.

    Returns ``(min_out_raw, "")`` or ``(None, reason)``. The quote must be a
    dict with the requested ``inputMint``/``outputMint``, ``inAmount`` equal
    to the requested raw amount, ``swapMode`` ExactIn when present, and both
    ``outAmount`` and ``otherAmountThreshold`` positive canonical integers
    (via ``parse_raw_amount``) with ``otherAmountThreshold <= outAmount``.
    Per Jupiter's Swap API (v1 ``QuoteResponse``), ``otherAmountThreshold`` is
    the minimum acceptable output: ``outAmount`` after ``slippageBps``. A
    missing, malformed or contradictory threshold is rejected; it never
    falls back to ``outAmount``.
    """
    if quote is None:
        return None, "no route"
    if not isinstance(quote, dict):
        return None, "malformed quote"
    if quote.get("inputMint") != input_mint or quote.get("outputMint") != output_mint:
        return None, "quote mints do not match request"
    if "swapMode" in quote and quote["swapMode"] != "ExactIn":
        return None, "quote is not ExactIn"
    if parse_raw_amount(quote.get("inAmount")) != in_amount:
        return None, "quote inAmount does not match request"
    out_raw = parse_raw_amount(quote.get("outAmount"))
    if out_raw is None or out_raw <= 0:
        return None, "quote outAmount missing, malformed or not positive"
    min_out = parse_raw_amount(quote.get("otherAmountThreshold"))
    if min_out is None or min_out <= 0:
        return None, "quote otherAmountThreshold missing, malformed or not positive"
    if min_out > out_raw:
        return None, "quote otherAmountThreshold exceeds outAmount"
    return min_out, ""


class TradingBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.jup = JupiterClient(cfg, self.session)
        self.executor = SwapExecutor(cfg, self.jup, self.session)
        self.screener = SafetyScreener(cfg, self.session)
        self.risk = RiskManager(cfg)
        self.portfolio = Portfolio()
        self.strategy = build_strategy(cfg, self.session)
        self.notifier = Notifier(cfg, self.session)
        self._running = True
        self._reconciliation_required = False
        self.decision_shadow = None
        if cfg.decision_shadow_enabled and DecisionShadowBridge is not None:
            # Share the bot HTTP session so safety + META + decision reuse one client.
            self.decision_shadow = DecisionShadowBridge(
                cfg, self.risk, self.portfolio, screener=self.screener,
                session=self.session,
            )
            log.info(
                "decision shadow ENABLED (dry-run journal/gate only; never authorizes live)"
            )
        load_state(cfg.state_file, self.portfolio, self.risk)
        if cfg.is_armed:
            self._flag_unverified_rungs()

    def _flag_unverified_rungs(self) -> None:
        """Legacy policy. State written before rungs were tied to bookings
        could mark a rung filled at the threshold although its sale never
        happened (crash, refusal, failure); it is indistinguishable from a
        sold rung. Such a rung is KEPT filled (never re-sold automatically:
        that could sell twice) and flagged for operator review. A rung is
        verified when a settled intent for it is on record."""
        settled = {
            (i.get("mint"), _intent_rung(i))
            for i in getattr(self.portfolio, "pending_intents", None) or []
            if isinstance(i, dict) and i.get("status") == "settled"
        }
        for mint, pos in self.portfolio.positions.items():
            for rung in sorted(getattr(pos, "ladder_filled", None) or ()):
                if (mint, _rung_value(rung)) not in settled:
                    log.warning(
                        "%s: take-profit rung %g is marked filled without a settled sale on "
                        "record (legacy state or pruned history); kept filled, NOT re-sold; "
                        "operator review: compare with on-chain history",
                        getattr(pos, "symbol", str(mint)[:8]), rung,
                    )

    def stop(self, *_):
        log.info("Kill switch received — shutting down after this tick.")
        self._running = False

    def run(self) -> None:
        self._banner()
        if self.cfg.is_armed and not durable_saves_supported():
            # Live sells require a durable intent before send; without
            # directory fsync no save is ever durable, so refuse to trade.
            log.error("live startup blocked: durable state saves (directory fsync) are "
                      "not supported on this platform")
            return
        mode = "LIVE — REAL MONEY" if self.cfg.is_armed else "DRY-RUN"
        if self.notifier.enabled:
            self.notifier.startup(mode)
        if self.cfg.is_armed:
            if self._open_sell_intents():
                # Settle/clear persisted intents on evidence first, so a sell
                # that landed before a restart is booked, not left as drift.
                self._resolve_sell_intents()
            reconciliation = reconcile_wallet(
                self.cfg, self.portfolio, self.session, self.notifier
            )
            if reconciliation is None or not reconciliation.clean:
                log.error("live startup blocked: wallet reconciliation is not clean")
                return
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                log.exception("tick error: %s", exc)
            for _ in range(self.cfg.poll_interval_seconds):
                if not self._running:
                    break
                time.sleep(1)
        self._save()
        self._summary()

    def _tick(self) -> None:
        if self._open_sell_intents():
            if not self.cfg.is_armed:
                log.error(
                    "trading halted: %d persisted live sell intent(s) need an armed "
                    "session to resolve against the chain; operator action required",
                    len(self._open_sell_intents()),
                )
                self._running = False
                return
            self._resolve_sell_intents()
        if self._reconciliation_required:
            reconciliation = reconcile_wallet(
                self.cfg, self.portfolio, self.session, self.notifier
            )
            if reconciliation is None or not reconciliation.clean:
                log.error("trading halted: reconciliation required after unknown transaction")
                self._running = False
                return
            self._reconciliation_required = False
            return
        self._manage_positions()
        still_open = self._open_sell_intents()
        if still_open:
            log.error(
                "new entries blocked: %d unresolved live sell intent(s) (%s)",
                len(still_open), ", ".join(sorted({str(i.get("mint"))[:8] for i in still_open})),
            )
            return
        if self.risk.can_open_new_position(self.portfolio.open_count):
            self._seek_entry()

    def _manage_positions(self) -> None:
        for mint in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[mint]
            if self._sell_frozen(mint):
                log.error(
                    "%s: exits frozen by an unresolved live sell intent; no new sell "
                    "until it is resolved (prevents a double sell)", pos.symbol,
                )
                continue
            decimals = self._verified_decimals(mint, pos)
            if decimals is None:
                log.warning(
                    "%s: token decimals unverified — no mark; holding position and cost basis",
                    pos.symbol,
                )
                continue
            price = _price_usd(self.jup, mint, decimals)
            # Missing and malformed quotes are not marks. Numeric 0.0 is.
            # Refuse anything else before exit rules or portfolio accounting.
            if not _is_executable_mark(price):
                if price is None:
                    log.warning(
                        "%s: exit quote unavailable — holding position and cost basis",
                        pos.symbol,
                    )
                else:
                    log.warning(
                        "%s: malformed exit quote — holding position and cost basis",
                        pos.symbol,
                    )
                continue
            for action in self.risk.evaluate_exit(pos.entry_price, price, pos.ladder_filled):
                # A take-profit rung is NOT consumed here: it is marked filled
                # only together with the sale's booking (paper: after the
                # accounting; live: in the settlement's atomic save). Until
                # then a failed / refused sell leaves it eligible, and an
                # unresolved one freezes the mint (no second sell).
                self._sell(mint, action.fraction, price, action.reason,
                           rung=_take_profit_rung(action.reason))

    def _seek_entry(self) -> None:
        for candidate in self.strategy.find_candidates():
            if self.portfolio.has(candidate.mint):
                continue
            # One DexScreener TokenSnapshot per candidate for safety + decision parity.
            market_snap = self._candidate_market_snapshot(candidate.mint)
            safety = self.screener.screen(candidate.mint, snapshot=market_snap)
            decision_shadow = getattr(self, "decision_shadow", None)
            if not safety.passed or safety.price_usd <= 0:
                # Still journal counterfactual when shadow is on (rejected by safety).
                if decision_shadow is not None and not self.cfg.is_armed:
                    try:
                        decision_shadow.evaluate_candidate(
                            safety, snapshot=market_snap,
                        )
                    except Exception as exc:
                        log.debug("decision shadow journal failed: %s", exc)
                continue

            # Phase 7/11: decision engine gates dry-run paper only; never live.
            if decision_shadow is not None and not self.cfg.is_armed:
                try:
                    eng = decision_shadow.evaluate_candidate(
                        safety, snapshot=market_snap,
                    )
                except Exception as exc:
                    log.error("decision shadow failed closed: %s", exc)
                    continue
                if not decision_shadow.allows_paper_entry(eng):
                    log.info(
                        "decision shadow skip %s: action=%s risk=%s",
                        safety.symbol, eng.decision.action, eng.risk.status,
                    )
                    continue

            size_usd = self.risk.position_size_usd()
            tokens = size_usd / safety.price_usd
            lamports = self._usd_to_lamports(size_usd)
            if lamports is None:
                # Fail closed: no verified SOL/USD price (or invalid size) means
                # no trade this tick — never a fabricated default price.
                log.warning(
                    "entry skipped for %s: SOL/USD price unavailable or size invalid",
                    safety.symbol,
                )
                return
            # Pre-buy sellability gate (paper and armed alike): quote-only,
            # before any swap, signature, send, or position side effect.
            if not self._sell_route_ok(candidate.mint, lamports):
                log.warning(
                    "entry skipped for %s: pre-buy sell route missing or uneconomic",
                    safety.symbol,
                )
                continue
            result = self.executor.swap(
                SOL_MINT, candidate.mint, lamports,
                allocation_usd=size_usd,
            )
            if not result.ok:
                log.info("entry aborted for %s: %s", safety.symbol, result.error)
                if self.cfg.is_armed and result.status in {
                    TransactionStatus.SUBMITTED,
                    TransactionStatus.TIMEOUT,
                    TransactionStatus.UNKNOWN,
                }:
                    self._reconciliation_required = True
                    log.error("live entry unresolved; new entries require reconciliation")
                continue
            if self.cfg.is_armed and result.status != TransactionStatus.CONFIRMED_SUCCESS:
                log.error("live entry was not confirmed; no position recorded")
                return
            if self.cfg.is_armed and (
                result.actual_in_amount is None or result.actual_out_amount is None
            ):
                self._reconciliation_required = True
                log.error("confirmed live entry lacks actual fill data; no position recorded")
                return
            if self.cfg.is_armed:
                # Decimals come from the confirmed fill, else the screener's
                # verified mint fetch; they are never guessed.
                decimals = valid_token_decimals(result.out_decimals)
                if decimals is None:
                    decimals = valid_token_decimals(getattr(safety, "decimals", None))
                if decimals is None:
                    self._reconciliation_required = True
                    log.error("confirmed live entry has unverified token decimals; no position recorded")
                    return
                if not is_raw_units(result.actual_out_amount):
                    self._reconciliation_required = True
                    log.error("confirmed live entry fill raw output is not an integer in "
                              "1..2**64-1; no position recorded")
                    return
                tokens = result.actual_out_amount / (10 ** decimals)
                if tokens <= 0:
                    self._reconciliation_required = True
                    log.error("confirmed live entry produced a zero fill")
                    return
                entry_price = size_usd / tokens
                self.portfolio.open(
                    candidate.mint, safety.symbol, entry_price, size_usd, tokens,
                    token_decimals=decimals,
                    entry_signature=result.tx_signature,
                    actual_in_amount=result.actual_in_amount,
                    actual_out_amount=result.actual_out_amount,
                    fee_lamports=result.fee_lamports,
                    raw_tokens=result.actual_out_amount,
                    raw_tokens_initial=result.actual_out_amount,
                )
            else:
                self.portfolio.open(
                    candidate.mint, safety.symbol, safety.price_usd, size_usd, tokens,
                    token_decimals=valid_token_decimals(getattr(safety, "decimals", None)),
                )
            opened = self.portfolio.positions[candidate.mint]
            self.notifier.buy(safety.symbol, size_usd, opened.entry_price, result.simulated)
            self._save()
            return

    def _candidate_market_snapshot(self, mint: str):
        """Fetch one META TokenSnapshot for this candidate (DexScreener + holders).

        Shared by SafetyScreener and DecisionShadowBridge so they observe the
        same market data. Returns None on provider failure (callers fail closed).
        """
        try:
            from .meta.adapters import attach_holder_concentration, snapshot_from_dexscreener
        except Exception as exc:  # pragma: no cover - package always present in tree
            log.debug("meta adapters unavailable: %s", exc)
            return None
        now = time.time()
        session = getattr(self, "session", None)
        try:
            snap = snapshot_from_dexscreener(mint, session, now=now)
        except Exception as exc:
            log.debug("candidate DexScreener snapshot failed for %s: %s", mint[:6], exc)
            return None
        if snap is None:
            return None
        try:
            rpc = getattr(getattr(self, "cfg", None), "rpc_url", "") or ""
            attach_holder_concentration(snap, rpc, session)
        except Exception as exc:
            log.debug("holder attach failed for %s: %s", mint[:6], exc)
        return snap

    def _sell(self, mint: str, fraction: float, price: float, reason: str,
              *, rung: float | None = None) -> None:
        if mint not in self.portfolio.positions:
            return
        pos = self.portfolio.positions[mint]
        symbol = pos.symbol
        if not _is_executable_mark(price):
            log.warning(
                "%s: refusing %s without an executable exit mark; position unchanged",
                symbol, reason,
            )
            return
        if self.cfg.is_armed:
            if self._sell_frozen(mint):
                log.error("%s: refusing %s; an earlier live sell is unresolved", symbol, reason)
                return
            # Settles and books atomically (or leaves an open intent).
            self._settle_live_sell(pos, fraction, price, reason, rung=rung)
            return
        self._book_sell(mint, fraction, price, reason, rung=rung)

    def _book_sell(self, mint: str, fraction: float, price: float, reason: str,
                   *, rung: float | None = None) -> None:
        """Paper-sell accounting (live sells book via ``_book_confirmed_sell``).
        A take-profit ``rung`` is marked filled only after the sell and its
        accounting succeeded."""
        symbol = self.portfolio.positions[mint].symbol
        was_halted = self.risk.trading_halted()
        pnl = self.portfolio.sell_fraction(mint, fraction, price, reason)
        self.risk.record_realized_pnl(pnl)
        self._mark_rung_filled(mint, rung)
        self.notifier.sell(symbol, fraction, price, reason, pnl)
        if self.risk.trading_halted() and not was_halted:
            self.notifier.halt(-self.risk.realized_pnl_today)
        self._save()

    def _accounting_snapshot(self):
        return (
            copy.deepcopy(self.portfolio.positions),
            self.portfolio.realized_pnl,
            self.risk.to_dict(),
        )

    def _restore_accounting(self, snapshot) -> None:
        self.portfolio.positions, self.portfolio.realized_pnl = snapshot[0], snapshot[1]
        self.risk.load_dict(snapshot[2])

    def _apply_sell_accounting(self, mint: str, fraction: float, price: float, reason: str,
                               *, raw_sold: int | None = None):
        """Live-sell portfolio + daily-risk accounting, in memory only.

        ``raw_sold`` (positions that track a raw holding) is the exact,
        fill-validated raw amount: the raw holding is decremented by exactly
        that. All-or-nothing: if anything raises, positions, realized P&L and
        the daily risk state are restored before re-raising, so memory never
        holds a half-booked sell that a later save could persist.
        """
        symbol = self.portfolio.positions[mint].symbol
        snapshot = self._accounting_snapshot()
        try:
            was_halted = self.risk.trading_halted()
            pnl = self.portfolio.sell_fraction(mint, fraction, price, reason, raw_sold=raw_sold)
            self.risk.record_realized_pnl(pnl)
            newly_halted = self.risk.trading_halted() and not was_halted
        except BaseException:
            self._restore_accounting(snapshot)
            raise
        return symbol, pnl, newly_halted

    def _mark_rung_filled(self, mint, rung) -> None:
        """Consume a take-profit rung (only once its sale is booked). A fully
        closed position has no ladder left to mark."""
        pos = self.portfolio.positions.get(mint)
        if rung is not None and pos is not None:
            pos.ladder_filled.add(rung)

    def _notify_sell(self, symbol, fraction, price, reason, pnl, newly_halted) -> None:
        self.notifier.sell(symbol, fraction, price, reason, pnl)
        if newly_halted:
            self.notifier.halt(-self.risk.realized_pnl_today)

    @staticmethod
    def _tracks_raw(pos) -> bool:
        """True unless the position is legacy (no raw fields at all)."""
        return (
            getattr(pos, "raw_tokens", None) is not None
            or getattr(pos, "raw_tokens_initial", None) is not None
        )

    def _confirmed_fill_problem(self, intent: dict) -> str | None:
        """Why stored fill evidence cannot settle ``intent``, or None if it can.

        Every raw amount must be an int (not bool) in ``1 .. 2**64-1``; the
        on-chain input must equal the intent's persisted ``raw_amount``; the
        position must still be tracked and, if it tracks a raw holding, that
        holding must be consistent and cover ``raw_amount``. Nothing is
        clamped or repaired.
        """
        raw_amount = intent.get("raw_amount")
        if not is_raw_units(raw_amount):
            return "intent raw_amount is not an integer in 1..2**64-1"
        if not intent.get("tx_signature"):
            return "confirmed sell has no transaction signature"
        if not is_raw_units(intent.get("actual_in_amount")) or (
            intent.get("actual_in_amount") != raw_amount
        ):
            return "on-chain input amount does not match the intent raw_amount"
        if not is_raw_units(intent.get("actual_out_amount")):
            return "on-chain output amount missing, not positive or out of range"
        pos = self.portfolio.positions.get(intent.get("mint"))
        if pos is None:
            return "position no longer tracked"
        if self._tracks_raw(pos):
            held, initial = pos.raw_tokens, pos.raw_tokens_initial
            if not (is_raw_units(held) and is_raw_units(initial) and held <= initial):
                return f"raw holding inconsistent (held={held!r}, bought={initial!r})"
            if raw_amount > held:
                return f"raw_amount {raw_amount} exceeds the current raw holding {held}"
        return None

    def _record_confirmed_fill(self, intent: dict, fill, detail: str) -> None:
        """Persist verified fill evidence before any accounting.

        Best effort: the commit point is the booking save (which carries this
        evidence too). If this save fails and the process dies, the persisted
        intent is still the earlier open one, so nothing is lost or doubled.
        """
        intent["status"] = "confirmed_unbooked"
        intent["resolution"] = detail
        intent["actual_in_amount"] = fill.actual_in_amount
        intent["actual_out_amount"] = fill.actual_out_amount
        intent["fee_lamports"] = fill.fee_lamports
        intent["confirmed_at"] = time.time()
        self._save()

    def _book_confirmed_sell(self, intent: dict) -> bool:
        """Book a ``confirmed_unbooked`` intent exactly once from its evidence.

        Accounting (raw holding decremented by exactly the intent's persisted
        ``raw_amount``, position, realized P&L, daily risk) and the intent's
        transition to ``settled`` are persisted in ONE atomic state write, so a
        crash leaves either the unbooked intent (booked on the next run) or the
        booked, settled one; never both or neither. If that write fails, the
        in-memory booking is rolled back and the intent stays
        ``confirmed_unbooked`` (frozen, retried next tick): memory never treats
        a settlement as done that is not on disk. Returns True if booked.
        """
        if intent.get("status") != "confirmed_unbooked" or intent.get("booked_at") is not None:
            return False
        mint = intent.get("mint")
        problem = self._confirmed_fill_problem(intent)
        if problem is not None:
            reason = f"confirmed sell cannot be booked: {problem}; operator action required"
            if intent.get("resolution") != reason:
                intent["resolution"] = reason
                self._save()
            log.error("live sell for %s %s", str(mint)[:8], reason)
            return False
        pos = self.portfolio.positions[mint]
        # Legacy positions (no raw fields) keep the float path for an intent
        # already in flight; they can never submit a new live sell.
        raw_sold = intent["raw_amount"] if self._tracks_raw(pos) else None
        fraction, price, reason = intent["fraction"], intent["requested_price"], intent["reason"]
        snapshot = self._accounting_snapshot()
        symbol, pnl, newly_halted = self._apply_sell_accounting(
            mint, fraction, price, reason, raw_sold=raw_sold,
        )
        # The take-profit rung is consumed in the SAME atomic save as the
        # booking (the snapshot above covers it, so a failed save rolls the
        # rung back too).
        self._mark_rung_filled(mint, _intent_rung(intent))
        intent["status"] = "settled"
        intent["booked_at"] = time.time()
        intent["booked_pnl"] = pnl
        intent["booked_raw"] = raw_sold
        if self._save() is False:  # one atomic write: accounting + settled intent
            self._restore_accounting(snapshot)
            intent["status"] = "confirmed_unbooked"
            for key in ("booked_at", "booked_pnl", "booked_raw"):
                intent.pop(key, None)
            intent["resolution"] = "booking not durable (state save failed); retried next tick"
            log.error("live sell for %s confirmed but its booking could not be saved; "
                      "rolled back in memory, retried next tick", str(mint)[:8])
            return False
        self._notify_sell(symbol, fraction, price, reason, pnl, newly_halted)
        return True

    def _settle_live_sell(self, pos, fraction: float, price: float, reason: str,
                          *, rung: float | None = None) -> bool:
        """Submit an armed sell of exactly ``exact_raw_sell_amount`` raw units.

        The intent, carrying that exact ``raw_amount``, must be durably saved
        before anything is quoted, signed or sent. Once the swap confirms, the
        fill must prove ``actual_in == raw_amount``; it is then booked (and the
        raw holding decremented by ``raw_amount``) via
        ``_book_confirmed_sell``. Returns True once booked.
        """
        tokens_to_sell = min(pos.original_tokens * fraction, pos.tokens)
        decimals = pos.token_decimals
        raw_amount, refusal = exact_raw_sell_amount(pos, fraction)
        intent = {
            "mint": pos.mint,
            "fraction": fraction,
            "reason": reason,
            "requested_price": price,
            "status": "pending",
            "tx_signature": None,
            "created_at": time.time(),
            "raw_amount": raw_amount,
        }
        if rung is not None:
            intent["rung"] = rung  # consumed only when this intent is booked
        self.portfolio.pending_intents.append(intent)
        durable = self._save()
        if decimals is None or tokens_to_sell <= 0:
            intent["status"] = "blocked"
            self._reconciliation_required = True
            log.error("live sell fail-closed for %s: missing decimals or size", pos.symbol)
            self._save()
            return False
        if raw_amount is None:
            # Definitive pre-submit refusal: nothing sent, so no chain
            # uncertainty and no reconciliation; the intent is cleared by the
            # resolver and the exit is re-evaluated (and refused) next tick.
            intent["status"] = "blocked"
            intent["error"] = refusal
            log.error("live sell refused before submit for %s: %s", pos.symbol, refusal)
            self._save()
            return False
        if durable is False:
            # Never send without a durable intent: after a crash the sale would
            # be invisible on restart and the exit could be sent again.
            intent["status"] = "blocked"
            intent["error"] = "state save failed before submit; nothing sent"
            log.error("live sell refused before submit for %s: state not durable", pos.symbol)
            self._save()
            return False
        tokens_to_sell = raw_amount / (10 ** valid_token_decimals(decimals))
        result = self.executor.swap(
            pos.mint, SOL_MINT, raw_amount,
            allocation_usd=min(tokens_to_sell * price, self.cfg.max_position_usd()),
        )
        intent["tx_signature"] = result.tx_signature
        intent["outcome"] = getattr(result.status, "value", str(result.status))
        intent["error"] = result.error
        if not result.ok or result.status != TransactionStatus.CONFIRMED_SUCCESS:
            intent["status"] = "unresolved"
            if result.status in {
                TransactionStatus.SUBMITTED,
                TransactionStatus.TIMEOUT,
                TransactionStatus.UNKNOWN,
            }:
                self._reconciliation_required = True
            log.error("live sell fail-closed for %s: %s", pos.symbol, result.error or result.status)
            self._save()
            return False
        if result.actual_in_amount is None or result.actual_out_amount is None:
            intent["status"] = "unresolved"
            self._reconciliation_required = True
            log.error("live sell confirmed without fill data for %s", pos.symbol)
            self._save()
            return False
        intent["actual_in_amount"] = result.actual_in_amount
        intent["actual_out_amount"] = result.actual_out_amount
        intent["fee_lamports"] = result.fee_lamports
        problem = self._confirmed_fill_problem(intent)
        if problem is not None:
            # Validated (incl. actual_in == raw_amount) before anything
            # bookable is persisted; the same check guards every booking.
            intent["status"] = "unresolved"
            intent["resolution"] = f"confirmed fill does not prove the intended sell: {problem}"
            self._reconciliation_required = True
            log.error("live sell for %s not booked: %s", pos.symbol, problem)
            self._save()
            return False
        self._record_confirmed_fill(intent, result, "confirmed at submit with on-chain fill")
        return self._book_confirmed_sell(intent)


    # -- live sell intent resolution -----------------------------------------

    def _open_sell_intents(self) -> list:
        return [
            intent for intent in getattr(self.portfolio, "pending_intents", None) or []
            if isinstance(intent, dict) and intent.get("status") in OPEN_SELL_INTENT_STATUSES
        ]

    def _sell_frozen(self, mint: str) -> bool:
        return any(intent.get("mint") == mint for intent in self._open_sell_intents())

    def _resolve_sell_intents(self) -> None:
        """Resolve open live sell intents on evidence only.

        * NOT EXECUTED (-> ``failed``, position untouched, exits resume):
          refused before send (``blocked``, or ``confirmed_failure`` with no
          signature); the signature is ``confirmed``/``finalized`` with an
          error; or the blockhash window has passed
          (``SELL_INTENT_EXPIRY_SECONDS``), the signature is not confirmed and
          the wallet still holds the tokens.
        * EXECUTED (-> ``confirmed_unbooked`` -> ``settled``, booked once): the
          signature is confirmed without error AND ``getTransaction`` gives a
          complete fill whose ``actual_in_amount`` equals the intent's
          ``raw_amount``. The evidence is persisted first; booking and the
          ``settled`` transition then share one atomic save. A
          ``confirmed_unbooked`` intent found later (e.g. after a crash) is
          booked from its stored evidence without another chain lookup.
        * Anything else stays open (the mint stays frozen, entries stay
          blocked) with an explicit operator-visible reason. No fill, price or
          P&L is ever invented.
        """
        now = time.time()
        balances: dict = {}
        changed = False
        for intent in self._open_sell_intents():
            if intent.get("status") == "confirmed_unbooked":
                # Proven on chain with stored evidence: book it, no chain lookup.
                if self._book_confirmed_sell(intent):
                    log.warning("live sell intent for %s booked from stored fill evidence",
                                str(intent.get("mint"))[:8])
                continue
            if "created_at" not in intent and "first_seen_at" not in intent:
                intent["first_seen_at"] = now  # legacy intent: start its clock now
                changed = True
            verdict, detail, fill = self._sell_intent_evidence(intent, now, balances)
            if verdict == "not_executed":
                intent["status"] = "failed"
                intent["resolution"] = detail
                intent["resolved_at"] = now
                changed = True
                log.warning(
                    "live sell intent for %s resolved as NOT executed (%s); position "
                    "kept, exit management resumes", str(intent.get("mint"))[:8], detail,
                )
                continue
            if verdict == "executed" and fill is not None:
                mint = intent["mint"]
                intent["resolved_at"] = now
                # Evidence is persisted first, then booking + settled in one save.
                self._record_confirmed_fill(intent, fill, detail)
                if self._book_confirmed_sell(intent):
                    log.warning("live sell intent for %s settled from on-chain fill", mint[:8])
                continue
            if intent.get("resolution") != detail:
                intent["resolution"] = detail
                changed = True
            log.error(
                "live sell intent for %s UNRESOLVED: %s — exits for this mint frozen, "
                "new entries blocked; other positions still managed",
                str(intent.get("mint"))[:8], detail,
            )
            notifier = getattr(self, "notifier", None)
            if not intent.get("alerted") and getattr(notifier, "enabled", False):
                notifier.send(
                    f"⚠️ unresolved live sell for {str(intent.get('mint'))[:8]}: {detail}"
                )
                intent["alerted"] = True
                changed = True
        terminal = [
            i for i in self.portfolio.pending_intents
            if not (isinstance(i, dict) and i.get("status") in OPEN_SELL_INTENT_STATUSES)
        ]
        if len(terminal) > TERMINAL_SELL_INTENTS_KEPT:
            drop = {id(i) for i in terminal[:-TERMINAL_SELL_INTENTS_KEPT]}
            self.portfolio.pending_intents = [
                i for i in self.portfolio.pending_intents if id(i) not in drop
            ]
            changed = True
        if changed:
            self._save()

    def _sell_intent_evidence(self, intent: dict, now: float, balances: dict):
        """(verdict, detail, fill) with verdict executed / not_executed / unknown."""
        status = intent.get("status")
        sig = intent.get("tx_signature")
        mint = intent.get("mint")
        if status == "blocked":
            return "not_executed", "refused before send", None
        if not sig and intent.get("outcome") == TransactionStatus.CONFIRMED_FAILURE.value:
            return "not_executed", "rejected before send (no transaction)", None
        pos = self.portfolio.positions.get(mint)
        if pos is None:
            return "unknown", "position no longer tracked; operator action required", None
        if sig:
            try:
                raw = get_signature_status(self.session, self.cfg.rpc_url, sig)
            except Exception as exc:
                return "unknown", f"signature status lookup failed: {exc}", None
            required = (
                {"finalized"} if self.cfg.confirmation_commitment == "finalized"
                else {"confirmed", "finalized"}
            )
            if isinstance(raw, dict) and raw.get("confirmationStatus") in required:
                if raw.get("err") is not None:
                    return "not_executed", "transaction failed on chain", None
                fill = self._confirmed_sell_fill(sig, mint)
                raw_amount = intent.get("raw_amount")
                if (
                    fill is None
                    or not isinstance(raw_amount, int) or isinstance(raw_amount, bool)
                    or fill.actual_in_amount != raw_amount
                    or not fill.actual_out_amount or fill.actual_out_amount <= 0
                ):
                    return (
                        "unknown",
                        "sell confirmed on chain but settlement data is missing or does "
                        "not match the intent; operator action required",
                        None,
                    )
                return "executed", "confirmed on chain with matching fill", fill
        started = intent.get("created_at", intent.get("first_seen_at"))
        if (
            isinstance(started, bool) or not isinstance(started, (int, float))
            or now - started < SELL_INTENT_EXPIRY_SECONDS
        ):
            return "unknown", "outcome not yet known; waiting for blockhash expiry", None
        if "amounts" not in balances:
            balances["amounts"] = self._wallet_token_amounts()
        amounts = balances["amounts"]
        if amounts is None:
            return "unknown", "wallet balance unavailable", None
        intended = self._intent_tokens(intent, pos)
        if intended is None:
            return "unknown", "intended sell size unknown; operator action required", None
        held = amounts.get(mint, 0.0)
        if held >= pos.tokens - intended / 2:
            return "not_executed", "expired unconfirmed; wallet still holds the tokens", None
        return (
            "unknown",
            "wallet balance dropped but no confirmed settlement data; operator action required",
            None,
        )

    def _intent_tokens(self, intent: dict, pos) -> float | None:
        raw = intent.get("raw_amount")
        decimals = valid_token_decimals(pos.token_decimals)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 and decimals is not None:
            return raw / (10 ** decimals)
        fraction = intent.get("fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or fraction <= 0:
            return None
        tokens = min(pos.original_tokens * fraction, pos.tokens)
        return tokens if tokens > 0 else None

    def _confirmed_sell_fill(self, signature: str, mint: str):
        owner = self.cfg.burner_wallet_pubkey
        if not owner:
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
        return extract_fill(tx, owner, mint, SOL_MINT)

    def _wallet_token_amounts(self) -> dict | None:
        owner = self.cfg.burner_wallet_pubkey
        if not owner:
            return None
        try:
            snapshot = fetch_wallet_token_balances(self.cfg, owner, self.session)
        except Exception as exc:
            log.warning("wallet balance fetch failed: %s", exc)
            return None
        return snapshot.amounts if snapshot is not None else None

    def _save(self) -> bool:
        """Persist state. False means the write is NOT known durable (it may
        not have happened, or the new file may be visible but not yet
        durable); callers about to send or to treat a settlement as done must
        check and fail closed. See ``bot.state.save_state``."""
        return save_state(self.cfg.state_file, self.portfolio, self.risk)

    def _usd_to_lamports(self, usd: float) -> int | None:
        """Lamports worth at most ``usd`` at a verified SOL/USD price, or None.

        Prices exactly 1 SOL (``10**9`` lamports) SOL->USDC; the USDC
        ``outAmount`` is USD in 1e-6 units, so
        ``lamports = floor(usd * 10**9 * 10**6 / outAmount)`` computed exactly
        (rounding can never exceed the allocation). Returns None — and the
        caller skips the trade — for invalid ``usd`` (None, bool, <=0, NaN,
        inf), an unavailable/invalid quote, or a zero-lamport result. There is
        no fallback SOL price (the old ``or 150.0`` default is removed).
        """
        if isinstance(usd, bool) or not isinstance(usd, (int, float)):
            return None
        if not math.isfinite(usd) or usd <= 0:
            return None
        out_raw = sol_usdc_raw_per_sol(self.jup)
        if out_raw is None:
            return None
        lamports = (Fraction(usd) * LAMPORTS_PER_SOL * USDC_RAW_PER_USD) // out_raw
        if lamports <= 0:
            return None
        return int(lamports)

    def _sell_route_ok(self, mint: str, lamports: int) -> bool:
        """Fail-closed, quote-only check that a planned buy can be sold back.

        1. Buy quote SOL -> ``mint`` for exactly ``lamports``; it must echo
           ``inputMint == SOL``, ``outputMint == mint``, ``inAmount ==
           lamports`` and give positive canonical integer ``outAmount`` and
           ``otherAmountThreshold`` (threshold <= outAmount; raw token units,
           no decimals needed or guessed).
        2. Sell quote ``mint`` -> SOL for exactly the buy's
           ``otherAmountThreshold`` (the fewest tokens the buy may deliver
           after slippage), validated the same way (amounts in lamports).
        3. Pass iff ``sell_min_out * 10 >= lamports * 9`` where
           ``sell_min_out`` is the sell quote's ``otherAmountThreshold``
           (slippage-adjusted round-trip loss <= ``MAX_ROUNDTRIP_LOSS`` = 10%),
           compared in integers. ``outAmount`` is never used as a fallback.

        Any missing/no-route/None quote, provider exception or timeout
        (``JupiterClient.quote`` returns None on its HTTP timeout), malformed,
        contradictory, wrong-mint, zero or negative amount returns False.
        It never signs, builds or sends a transaction, and has no fallback.
        Quotes are routing estimates, not simulations: a pass does NOT
        guarantee that a later sell will succeed (e.g. Token-2022 transfer
        hooks, freezes, or liquidity pulled after entry).
        """
        short = mint[:6] if isinstance(mint, str) else "?"
        if not isinstance(mint, str) or not mint or mint == SOL_MINT:
            log.warning("sell-route check %s: invalid mint", short)
            return False
        if isinstance(lamports, bool) or not isinstance(lamports, int) or lamports <= 0:
            log.warning("sell-route check %s: invalid planned lamports", short)
            return False
        jup = getattr(self, "jup", None)
        if jup is None or not callable(getattr(jup, "quote", None)):
            log.warning("sell-route check %s: quote client unavailable", short)
            return False
        try:
            buy_quote = jup.quote(SOL_MINT, mint, lamports)
        except Exception as exc:
            log.warning("sell-route check %s: buy quote failed (%s)", short, type(exc).__name__)
            return False
        token_min, reason = _validated_route_out(buy_quote, SOL_MINT, mint, lamports)
        if token_min is None:
            log.warning("sell-route check %s: buy quote rejected: %s", short, reason)
            return False
        try:
            sell_quote = jup.quote(mint, SOL_MINT, token_min)
        except Exception as exc:
            log.warning("sell-route check %s: sell quote failed (%s)", short, type(exc).__name__)
            return False
        sell_min_out, reason = _validated_route_out(sell_quote, mint, SOL_MINT, token_min)
        if sell_min_out is None:
            log.warning("sell-route check %s: sell quote rejected: %s", short, reason)
            return False
        keep = MAX_ROUNDTRIP_LOSS.denominator - MAX_ROUNDTRIP_LOSS.numerator
        if sell_min_out * MAX_ROUNDTRIP_LOSS.denominator < lamports * keep:
            log.warning(
                "sell-route check %s: min round trip returns %d of %d lamports (> %d%% loss)",
                short, sell_min_out, lamports,
                MAX_ROUNDTRIP_LOSS.numerator * 100 // MAX_ROUNDTRIP_LOSS.denominator,
            )
            return False
        return True

    def _verified_decimals(self, mint: str, pos=None) -> int | None:
        """Verified SPL decimals for ``mint``, or None (never guessed).

        Uses the position's recorded decimals (confirmed fill, or the
        screener's mint fetch at entry), else the screener's verified
        getAccountInfo mint fetch, cached only on success.
        """
        recorded = valid_token_decimals(getattr(pos, "token_decimals", None))
        if recorded is not None:
            return recorded
        cache = getattr(self, "_decimals_cache", None)
        if cache is None:
            cache = {}
            self._decimals_cache = cache
        if mint in cache:
            return cache[mint]
        fetch = getattr(getattr(self, "screener", None), "_fetch_mint_account", None)
        if not callable(fetch):
            return None
        try:
            info = fetch(mint)
        except Exception as exc:
            log.debug("mint decimals fetch failed for %s: %s", mint[:6], exc)
            return None
        decimals = valid_token_decimals(info.get("decimals")) if isinstance(info, dict) else None
        if decimals is not None:
            cache[mint] = decimals
        return decimals

    def _banner(self) -> None:
        mode = "LIVE — REAL MONEY" if self.cfg.is_armed else "DRY-RUN (no funds at risk)"
        log.info("=" * 64)
        log.info("Solana memecoin bot starting — mode: %s", mode)
        log.info("Bankroll $%.2f | max %.1f%%/trade | stop -%.1f%% | daily stop -%.1f%%",
                 self.cfg.bankroll_usd, self.cfg.max_position_pct,
                 self.cfg.stop_loss_pct, self.cfg.daily_loss_limit_pct)
        if self.cfg.live_trading and not self.cfg.is_armed:
            log.warning("LIVE_TRADING=true but no wallet key — safety lock held DRY-RUN.")
        log.info("=" * 64)

    def _summary(self) -> None:
        log.info("Session realized PnL: $%.2f | open positions: %d",
                 self.portfolio.realized_pnl, self.portfolio.open_count)


def main() -> int:
    _setup_logging()
    try:
        cfg = with_resolved_data_paths(load_config())
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    log_resolved_paths(
        state_file=cfg.state_file,
        process_lock_file=cfg.process_lock_file,
        decision_shadow_file=cfg.decision_shadow_file,
        decision_model_path=cfg.decision_model_path,
    )
    process_lock = ProcessLock(cfg.process_lock_file)
    if not process_lock.acquire():
        log.error("another memebot instance is running or process lock is unavailable")
        return 3
    try:
        if cfg.is_armed:
            preflight = run_preflight(cfg, process_lock_held=True)
            if not preflight.eligible:
                log.error(
                    "live startup blocked by preflight: %s",
                    ", ".join(preflight.reason_codes),
                )
                return 4
        bot = TradingBot(cfg)
        signal.signal(signal.SIGINT, bot.stop)
        signal.signal(signal.SIGTERM, bot.stop)
        bot.run()
        return 0
    finally:
        process_lock.release()


if __name__ == "__main__":
    sys.exit(main())
