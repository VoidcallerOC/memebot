"""The loop that ties everything together.

Each tick:
  1. Manage existing positions first (exits before entries — protect capital).
  2. If the daily loss limit isn't hit and we have room, look for one new
     entry, screen it for safety, and (paper or live) buy a risk-sized amount.

Run with: ``python -m bot.main``
"""

from __future__ import annotations

import logging
import signal
import sys
import time

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

log = logging.getLogger("bot")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _price_usd(jup: JupiterClient, mint: str) -> float | None:
    """Price one token in USDC via a small Jupiter quote (1 token, 6dp est).
    Returns None if there's no route."""
    quote = jup.quote(mint, USDC_MINT, 1_000_000)  # ~1 token at 6 decimals
    if not quote:
        return None
    try:
        return int(quote["outAmount"]) / 1_000_000
    except (KeyError, ValueError):
        return None


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
        # Restore any persisted positions / daily state before trading.
        load_state(cfg.state_file, self.portfolio, self.risk)

    def stop(self, *_):
        log.info("Kill switch received — shutting down after this tick.")
        self._running = False

    def run(self) -> None:
        self._banner()
        mode = "LIVE — REAL MONEY" if self.cfg.is_armed else "DRY-RUN"
        if self.notifier.enabled:
            self.notifier.startup(mode)
        # When armed, verify our tracked positions match the actual wallet
        # before we start trading on a possibly-stale picture.
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
            except Exception as exc:  # never let one bad tick kill the bot
                log.exception("tick error: %s", exc)
            for _ in range(self.cfg.poll_interval_seconds):
                if not self._running:
                    break
                time.sleep(1)
        self._save()
        self._summary()

    # -- one iteration ------------------------------------------------------

    def _tick(self) -> None:
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
            if price is None:
                # Can't price it = possible rug/liquidity pull. Try to bail.
                log.warning("%s: no price/route — attempting emergency exit", pos.symbol)
                self._sell(mint, 1.0, price or 0.0, "no_route_exit")
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
            safety = self.screener.screen(candidate.mint)
            if not safety.passed or safety.price_usd <= 0:
                continue

            size_usd = self.risk.position_size_usd()
            # Buy with SOL; size in lamports requires SOL price, but for the
            # paper path we record USD cost basis directly and simulate fill.
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
            self.portfolio.open(candidate.mint, safety.symbol,
                                safety.price_usd, size_usd, tokens)
            self.notifier.buy(safety.symbol, size_usd, safety.price_usd, result.simulated)
            self._save()
            return  # one entry per tick keeps things calm

    def _sell(self, mint: str, fraction: float, price: float, reason: str) -> None:
        symbol = self.portfolio.positions[mint].symbol if mint in self.portfolio.positions else mint[:6]
        was_halted = self.risk.trading_halted()
        pnl = self.portfolio.sell_fraction(mint, fraction, price, reason)
        self.risk.record_realized_pnl(pnl)
        self.notifier.sell(symbol, fraction, price, reason, pnl)
        if self.risk.trading_halted() and not was_halted:
            self.notifier.halt(-self.risk.realized_pnl_today)
        self._save()

    def _save(self) -> None:
        save_state(self.cfg.state_file, self.portfolio, self.risk)

    # -- helpers ------------------------------------------------------------

    def _usd_to_lamports(self, usd: float) -> int:
        """Rough USD->lamports using a live SOL/USDC quote; falls back to a
        conservative constant if the quote is unavailable."""
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
