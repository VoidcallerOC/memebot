"""The loop that ties everything together.

Each tick:
  1. Manage existing positions first (exits before entries — protect capital).
  2. If the daily loss limit isn't hit and we have room, look for one new
     entry, screen it for safety, and (paper or live) buy a risk-sized amount.

Run with: ``python -m bot.main``
"""
from __future__ import annotations

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
from .config import Config, load_config
from .jupiter import (
    JupiterClient, LAMPORTS_PER_SOL, SOL_MINT, USDC_MINT, USDC_RAW_PER_USD,
    SwapExecutor, TransactionStatus, parse_raw_amount, sol_usdc_raw_per_sol,
    valid_token_decimals,
)
from .preflight import run_preflight
from .portfolio import Portfolio
from .paths import log_resolved_paths, with_resolved_data_paths
from .process_lock import ProcessLock
from .reconcile import reconcile_wallet
from .risk import RiskManager
from .safety import SafetyScreener
from .state import load_state, save_state
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

# Pre-buy sellability gate: the quoted SOL -> token -> SOL round trip for the
# exact planned entry may lose at most this fraction of the input lamports.
# Exactly 10% passes; anything worse is rejected. Compared in integers
# (sell_out * 10 >= lamports * 9); deliberately a constant, not a config knob,
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
    """Raw ``outAmount`` of an ExactIn quote that matches the request exactly.

    Returns ``(out_raw, "")`` or ``(None, reason)``. The quote must be a dict
    with the requested ``inputMint``/``outputMint``, ``inAmount`` equal to the
    requested raw amount, a positive canonical integer ``outAmount`` (via
    ``parse_raw_amount``), and ``swapMode`` ExactIn when present.
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
    return out_raw, ""


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

    def stop(self, *_):
        log.info("Kill switch received — shutting down after this tick.")
        self._running = False

    def run(self) -> None:
        self._banner()
        mode = "LIVE — REAL MONEY" if self.cfg.is_armed else "DRY-RUN"
        if self.notifier.enabled:
            self.notifier.startup(mode)
        if self.cfg.is_armed:
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
        if any(
            intent.get("status") in {"pending", "unresolved", "blocked"}
            for intent in getattr(self.portfolio, "pending_intents", [])
        ):
            self._reconciliation_required = True
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
        if self.risk.can_open_new_position(self.portfolio.open_count):
            self._seek_entry()

    def _manage_positions(self) -> None:
        for mint in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[mint]
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
                if action.reason.startswith("take_profit:"):
                    target = float(action.reason.split(":")[1])
                    pos.ladder_filled.add(target)
                self._sell(mint, action.fraction, price, action.reason)

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

    def _sell(self, mint: str, fraction: float, price: float, reason: str) -> None:
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
            if not self._settle_live_sell(pos, fraction, price, reason):
                return
        was_halted = self.risk.trading_halted()
        pnl = self.portfolio.sell_fraction(mint, fraction, price, reason)
        self.risk.record_realized_pnl(pnl)
        self.notifier.sell(symbol, fraction, price, reason, pnl)
        if self.risk.trading_halted() and not was_halted:
            self.notifier.halt(-self.risk.realized_pnl_today)
        self._save()

    def _settle_live_sell(self, pos, fraction: float, price: float, reason: str) -> bool:
        tokens_to_sell = min(pos.original_tokens * fraction, pos.tokens)
        decimals = pos.token_decimals
        intent = {
            "mint": pos.mint,
            "fraction": fraction,
            "reason": reason,
            "requested_price": price,
            "status": "pending",
            "tx_signature": None,
        }
        self.portfolio.pending_intents.append(intent)
        self._save()
        if decimals is None or tokens_to_sell <= 0:
            intent["status"] = "blocked"
            self._reconciliation_required = True
            log.error("live sell fail-closed for %s: missing decimals or size", pos.symbol)
            self._save()
            return False
        raw_amount = int(tokens_to_sell * (10 ** decimals))
        if raw_amount <= 0:
            intent["status"] = "blocked"
            self._reconciliation_required = True
            self._save()
            return False
        result = self.executor.swap(
            pos.mint, SOL_MINT, raw_amount,
            allocation_usd=min(tokens_to_sell * price, self.cfg.max_position_usd()),
        )
        intent["tx_signature"] = result.tx_signature
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
        intent["status"] = "settled"
        self._save()
        return True

    def _save(self) -> None:
        save_state(self.cfg.state_file, self.portfolio, self.risk)

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
           lamports`` and give a positive canonical integer ``outAmount``
           (raw token units; no decimals needed or guessed).
        2. Sell quote ``mint`` -> SOL for exactly that raw ``outAmount``,
           validated the same way (``outAmount`` in lamports).
        3. Pass iff ``sell_out * 10 >= lamports * 9`` (round-trip loss
           <= ``MAX_ROUNDTRIP_LOSS`` = 10%), compared in integers.

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
        token_raw, reason = _validated_route_out(buy_quote, SOL_MINT, mint, lamports)
        if token_raw is None:
            log.warning("sell-route check %s: buy quote rejected: %s", short, reason)
            return False
        try:
            sell_quote = jup.quote(mint, SOL_MINT, token_raw)
        except Exception as exc:
            log.warning("sell-route check %s: sell quote failed (%s)", short, type(exc).__name__)
            return False
        sell_out, reason = _validated_route_out(sell_quote, mint, SOL_MINT, token_raw)
        if sell_out is None:
            log.warning("sell-route check %s: sell quote rejected: %s", short, reason)
            return False
        keep = MAX_ROUNDTRIP_LOSS.denominator - MAX_ROUNDTRIP_LOSS.numerator
        if sell_out * MAX_ROUNDTRIP_LOSS.denominator < lamports * keep:
            log.warning(
                "sell-route check %s: round trip returns %d of %d lamports (> %d%% loss)",
                short, sell_out, lamports,
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
