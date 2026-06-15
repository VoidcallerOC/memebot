"""Entry signal — deliberately simple and meant to be replaced.

IMPORTANT: This is the part of the bot you should trust the *least*. No public
entry signal has an edge in memecoins; the protective machinery in
``safety.py`` and ``risk.py`` is what actually keeps you alive. Treat this as a
placeholder you can swap for your own thesis (new-pair sniping, social signal,
momentum, copy-trading a wallet, etc.).

Two candidate *sources* ship in the box, selected by ``STRATEGY``:

  * ``boosted`` (default) — recently-active tokens from DexScreener's boosted
    feed. Broad and noisy; leans entirely on the safety screen.
  * ``smart_money`` — tokens that a configured watchlist of wallets is *newly*
    accumulating, read straight from the Solana RPC. Following proven on-chain
    actors is a more defensible edge than chasing boosts, but it is only as
    good as the wallet list you give it.

Neither is a buy signal: every candidate still passes through ``safety.py`` and
``risk.py`` before a cent moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

from .config import Config
from .jupiter import SOL_MINT, USDC_MINT

log = logging.getLogger(__name__)

# Boosted/active tokens feed (no key). This is a candidate *source*, not a buy
# signal — everything still passes through safety screening.
DEXSCREENER_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/latest/v1"
HTTP_TIMEOUT = 8

# SPL token programs whose accounts we scan for a wallet's holdings.
SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

# Quote/stable mints a wallet always holds — never a memecoin "buy" signal.
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
QUOTE_MINTS = frozenset({SOL_MINT, USDC_MINT, USDT_MINT})


@dataclass
class Candidate:
    mint: str


class Strategy:
    """Default candidate source: DexScreener's boosted/active token feed."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()

    def find_candidates(self) -> list[Candidate]:
        """Return a list of candidate mints to evaluate this tick."""
        try:
            resp = self.session.get(DEXSCREENER_BOOSTS_URL, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            items = resp.json() or []
        except Exception as exc:
            log.debug("candidate fetch failed: %s", exc)
            return []

        candidates: list[Candidate] = []
        for item in items:
            if item.get("chainId") != "solana":
                continue
            mint = item.get("tokenAddress")
            if mint:
                candidates.append(Candidate(mint=mint))
        return candidates


class SmartMoneyStrategy:
    """Surface tokens a watchlist of wallets is *newly* accumulating.

    Each tick we read every watched wallet's current SPL holdings from the RPC
    and emit any mint that wallet did not hold the last time we looked. The
    first time we see a wallet we record its holdings as a baseline and emit
    nothing (unless ``surface_existing``), so the bot doesn't try to buy a
    wallet's entire pre-existing bag on startup.

    The RPC fetch (:meth:`_wallet_mints`) is isolated from the diff logic so the
    accumulation detection can be tested without a network.
    """

    def __init__(
        self,
        rpc_url: str,
        wallets: list[str],
        session: Optional[requests.Session] = None,
        surface_existing: bool = False,
    ):
        self.rpc_url = rpc_url
        self.wallets = [w.strip() for w in wallets if w.strip()]
        self.session = session or requests.Session()
        self.surface_existing = surface_existing
        # wallet -> every mint we've ever seen it hold (so a token is only
        # surfaced once, not re-bought every tick it keeps holding it).
        self._seen: dict[str, set[str]] = {}
        if not self.wallets:
            log.warning(
                "smart_money strategy selected but SMART_MONEY_WALLETS is empty "
                "— no candidates will ever be produced."
            )

    def find_candidates(self) -> list[Candidate]:
        new_mints: set[str] = set()
        for wallet in self.wallets:
            mints = self._wallet_mints(wallet)
            if mints is None:
                # RPC hiccup: leave this wallet's baseline untouched so we don't
                # later mistake its whole bag for fresh accumulation.
                continue
            mints = {m for m in mints if m not in QUOTE_MINTS}
            if wallet not in self._seen:
                if self.surface_existing:
                    new_mints |= mints
                self._seen[wallet] = set(mints)
                continue
            fresh = mints - self._seen[wallet]
            if fresh:
                log.info("smart-money: %s newly holds %d token(s)",
                         wallet[:8], len(fresh))
                new_mints |= fresh
            self._seen[wallet] |= mints
        return [Candidate(mint=m) for m in sorted(new_mints)]

    # -- data fetching (isolated so it's easy to mock in tests) -------------

    def _wallet_mints(self, wallet: str) -> Optional[set[str]]:
        """Return the set of mints ``wallet`` currently holds (positive
        balance), or None if the RPC could not be read at all."""
        any_ok = False
        mints: set[str] = set()
        for program in (SPL_TOKEN_PROGRAM, TOKEN_2022_PROGRAM):
            accounts = self._fetch_token_accounts(wallet, program)
            if accounts is None:
                continue
            any_ok = True
            for acct in accounts:
                try:
                    info = acct["account"]["data"]["parsed"]["info"]
                    ui = float(info["tokenAmount"]["uiAmount"] or 0)
                    if ui > 0:
                        mints.add(info["mint"])
                except (KeyError, TypeError, ValueError):
                    continue
        return mints if any_ok else None

    def _fetch_token_accounts(self, wallet: str, program: str) -> Optional[list]:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [wallet, {"programId": program}, {"encoding": "jsonParsed"}],
        }
        try:
            resp = self.session.post(self.rpc_url, json=payload, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return (resp.json().get("result") or {}).get("value") or []
        except Exception as exc:
            log.debug("smart-money fetch failed for %s (%s): %s",
                      wallet[:8], program[:4], exc)
            return None


def build_strategy(cfg: Config, session: Optional[requests.Session] = None):
    """Pick the candidate source named by ``cfg.strategy``."""
    if cfg.strategy == "smart_money":
        log.info("strategy: smart_money (%d wallet(s) watched)",
                 len(cfg.smart_money_wallets))
        return SmartMoneyStrategy(
            cfg.rpc_url, cfg.smart_money_wallets, session,
            surface_existing=cfg.smart_money_surface_existing,
        )
    if cfg.strategy != "boosted":
        log.warning("unknown STRATEGY=%r; falling back to 'boosted'", cfg.strategy)
    log.info("strategy: boosted (DexScreener feed)")
    return Strategy(session)
