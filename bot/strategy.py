"""Entry signal — deliberately simple and meant to be replaced.

IMPORTANT: This is the part of the bot you should trust the *least*. No public
entry signal has an edge in memecoins; the protective machinery in
``safety.py`` and ``risk.py`` is what actually keeps you alive. Treat this as a
placeholder you can swap for your own thesis (new-pair sniping, social signal,
momentum, copy-trading a wallet, etc.).

The default implementation surfaces recently-active tokens from DexScreener's
Solana trending/boosted feeds and lets the safety screen + risk manager decide
whether any of them are tradeable. It does NOT chase pumps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

log = logging.getLogger(__name__)

# Boosted/active tokens feed (no key). This is a candidate *source*, not a buy
# signal — everything still passes through safety screening.
DEXSCREENER_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/latest/v1"
HTTP_TIMEOUT = 8


@dataclass
class Candidate:
    mint: str


class Strategy:
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
