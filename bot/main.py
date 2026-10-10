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
import signal
import sys
import time
from enum import Enum

import requests

from .alerts import Notifier
from .config import Config, load_config
from .jupiter import (
    JupiterClient, SOL_MINT, USDC_MINT, SwapExecutor, TransactionStatus,
)
from .preflight import run_preflight
from .portfolio import Portfolio
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
    ``MALFORMED`` is an outAmount that is not a finite price >= 0.
    ``PRICE`` is an executable mark, including a legitimate numeric 0.0.
    """

    MISSING = "missing"
    MALFORMED = "malformed"
    PRICE = "price"


def _parse_out_amount_usd(raw: object) -> float | None:
    """USD value of a USDC ``outAmount``, or None when it is not executable.

    Raw units are scaled by 1_000_000 (USDC decimals). Integer zero is a
    legitimate quote and returns 0.0. Negatives, non-finite results, empty
    values, non-numeric text, and non-integral types are rejected.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return None
    try:
        if isinstance(raw, str):
            text = raw.strip()
            if text == "":
                return None
            amount = int(text, 10)
        else:
            amount = raw
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        return None
    price = amount / 1_000_000
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


def _price_usd(jup: JupiterClient, mint: str) -> float | None:
    """USD mark for 1 token unit, or None when that mark must not be traded.

    None covers a missing quote and a malformed outAmount (negative, NaN,
    infinity, empty, non-numeric, or wrong type). A parsed 0.0 is returned
    as 0.0 so a legitimate zero quote still follows the exit rules.
    """
    quote = jup.quote(mint, USDC_MINT, 1_000_000)
    kind, price = _classify_exit_quote(quote)
    if kind is ExitQuoteKind.MALFORMED:
        log.warning("malformed exit quote for %s — not an executable price", mint)
        return None
    if kind is not ExitQuoteKind.PRICE or not _is_executable_mark(price):
        return None
    return float(price)


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
            price = _price_usd(self.jup, mint)
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
            result = self.executor.swap(
                SOL_MINT, candidate.mint, self._usd_to_lamports(size_usd),
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
                decimals = result.out_decimals if result.out_decimals is not None else 6
                tokens = result.actual_out_amount / (10 ** decimals)
                if tokens <= 0:
                    self._reconciliation_required = True
                    log.error("confirmed live entry produced a zero fill")
                    return
                entry_price = size_usd / tokens
                self.portfolio.open(
                    candidate.mint, safety.symbol, entry_price, size_usd, tokens,
                    token_decimals=result.out_decimals,
                    entry_signature=result.tx_signature,
                    actual_in_amount=result.actual_in_amount,
                    actual_out_amount=result.actual_out_amount,
                    fee_lamports=result.fee_lamports,
                )
            else:
                self.portfolio.open(
                    candidate.mint, safety.symbol, safety.price_usd, size_usd, tokens,
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

    def _usd_to_lamports(self, usd: float) -> int:
        sol_price = _price_usd(self.jup, SOL_MINT) or 150.0
        sol_amount = usd / sol_price
        return int(sol_amount * 1_000_000_000)

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
        cfg = load_config()
    except ValueError as exc:
        log.error("%s", exc)
        return 2
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
