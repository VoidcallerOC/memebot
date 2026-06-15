"""Token safety screening — the most important file in this project.

The fastest way to lose money in memecoins is to buy a token you can never
sell (a honeypot) or one whose creator can drain the liquidity pool (a rug).
Good entry timing means nothing if the token is a trap. This module screens a
token *before* the bot is ever allowed to buy it.

It uses two public data sources:
  * DexScreener  — liquidity, volume, pair metadata (no API key needed).
  * The Solana RPC — mint authority / freeze authority on the SPL mint.

Optionally it will consult RugCheck (https://rugcheck.xyz) if reachable.

Every check is conservative: when in doubt, REJECT. A missed opportunity costs
nothing; a honeypot costs everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

from .config import Config

log = logging.getLogger(__name__)

DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{mint}"
RUGCHECK_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary"
HTTP_TIMEOUT = 8


@dataclass
class TokenSafety:
    mint: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    liquidity_usd: float = 0.0
    price_usd: float = 0.0
    symbol: str = ""

    def reject(self, reason: str) -> None:
        self.passed = False
        self.reasons.append(reason)


class SafetyScreener:
    """Screens SPL tokens for honeypot / rug indicators."""

    def __init__(self, cfg: Config, session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.session = session or requests.Session()

    # -- public API ---------------------------------------------------------

    def screen(self, mint: str) -> TokenSafety:
        result = TokenSafety(mint=mint, passed=True)

        market = self._fetch_market(mint)
        if market is None:
            result.reject("no market data from DexScreener (untradeable/too new)")
            return result

        result.liquidity_usd = market["liquidity_usd"]
        result.price_usd = market["price_usd"]
        result.symbol = market["symbol"]

        self._check_liquidity(result)
        self._check_mint_authorities(result, mint)
        self._check_holder_concentration(result, mint)
        self._consult_rugcheck(result, mint)

        if result.passed:
            log.info("SAFE  %s (%s): liq $%.0f", result.symbol, mint[:6], result.liquidity_usd)
        else:
            log.info("REJECT %s (%s): %s", result.symbol, mint[:6], "; ".join(result.reasons))
        return result

    # -- individual checks --------------------------------------------------

    def _check_liquidity(self, result: TokenSafety) -> None:
        if result.liquidity_usd < self.cfg.min_liquidity_usd:
            result.reject(
                f"liquidity ${result.liquidity_usd:.0f} < min "
                f"${self.cfg.min_liquidity_usd:.0f}"
            )

    def _check_mint_authorities(self, result: TokenSafety, mint: str) -> None:
        """A live mint authority = dev can print infinite supply and dump.
        A live freeze authority = dev can freeze your tokens (honeypot)."""
        info = self._fetch_mint_account(mint)
        if info is None:
            # Be conservative: if we can't verify, and the operator requires it,
            # we reject rather than assume safety.
            if self.cfg.require_mint_revoked or self.cfg.require_freeze_revoked:
                result.reject("could not verify mint/freeze authority")
            return

        if self.cfg.require_mint_revoked and info.get("mint_authority") is not None:
            result.reject("mint authority NOT revoked (dev can print more supply)")
        if self.cfg.require_freeze_revoked and info.get("freeze_authority") is not None:
            result.reject("freeze authority NOT revoked (possible honeypot)")

    def _check_holder_concentration(self, result: TokenSafety, mint: str) -> None:
        top_pct = self._fetch_top_holder_pct(mint)
        if top_pct is None:
            return  # informational; don't hard-reject on RPC gaps here
        if top_pct > self.cfg.max_top_holder_pct:
            result.reject(
                f"top holder owns {top_pct:.0f}% > max {self.cfg.max_top_holder_pct:.0f}%"
            )

    def _consult_rugcheck(self, result: TokenSafety, mint: str) -> None:
        """Best-effort third-party opinion. Never the sole gate, but a hard
        'danger' verdict or unlocked-liquidity flag will veto a buy."""
        try:
            resp = self.session.get(RUGCHECK_URL.format(mint=mint), timeout=HTTP_TIMEOUT)
            if resp.status_code != 200:
                return
            data = resp.json()
        except Exception as exc:  # network/parse issues are non-fatal
            log.debug("rugcheck unavailable for %s: %s", mint[:6], exc)
            return

        score = data.get("score") or data.get("score_normalised")
        risks = data.get("risks") or []
        risk_names = {str(r.get("name", "")).lower() for r in risks}

        if self.cfg.require_liquidity_locked:
            unlocked = any("lp unlocked" in n or "liquidity" in n and "unlocked" in n
                           for n in risk_names)
            if unlocked:
                result.reject("RugCheck: liquidity unlocked")

        if any("honeypot" in n for n in risk_names):
            result.reject("RugCheck: honeypot flagged")

        # RugCheck score: higher = riskier in their summary schema.
        if isinstance(score, (int, float)) and score is not None and score >= 40000:
            result.reject(f"RugCheck risk score high ({score})")

    # -- data fetching (isolated so they're easy to mock in tests) ----------

    def _fetch_market(self, mint: str) -> Optional[dict]:
        try:
            resp = self.session.get(
                DEXSCREENER_TOKEN_URL.format(mint=mint), timeout=HTTP_TIMEOUT
            )
            resp.raise_for_status()
            pairs = resp.json().get("pairs") or []
        except Exception as exc:
            log.debug("dexscreener fetch failed for %s: %s", mint[:6], exc)
            return None
        if not pairs:
            return None
        # Use the deepest-liquidity pair as the reference market.
        best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0)
        return {
            "liquidity_usd": float((best.get("liquidity") or {}).get("usd", 0) or 0),
            "price_usd": float(best.get("priceUsd") or 0) or 0.0,
            "symbol": (best.get("baseToken") or {}).get("symbol", "?"),
        }

    def _fetch_mint_account(self, mint: str) -> Optional[dict]:
        """Read the SPL mint account via RPC and extract authorities."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [mint, {"encoding": "jsonParsed"}],
        }
        try:
            resp = self.session.post(self.cfg.rpc_url, json=payload, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            value = (resp.json().get("result") or {}).get("value")
            info = (((value or {}).get("data") or {}).get("parsed") or {}).get("info")
            if not info:
                return None
            return {
                "mint_authority": info.get("mintAuthority"),
                "freeze_authority": info.get("freezeAuthority"),
            }
        except Exception as exc:
            log.debug("mint account fetch failed for %s: %s", mint[:6], exc)
            return None

    def _fetch_top_holder_pct(self, mint: str) -> Optional[float]:
        """Approximate the largest holder's share via getTokenLargestAccounts
        over the total supply."""
        try:
            largest = self.session.post(
                self.cfg.rpc_url,
                json={
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTokenLargestAccounts", "params": [mint],
                },
                timeout=HTTP_TIMEOUT,
            ).json()
            supply = self.session.post(
                self.cfg.rpc_url,
                json={
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTokenSupply", "params": [mint],
                },
                timeout=HTTP_TIMEOUT,
            ).json()
            accounts = (largest.get("result") or {}).get("value") or []
            total = float(((supply.get("result") or {}).get("value") or {}).get("uiAmount") or 0)
            if not accounts or total <= 0:
                return None
            top = float(accounts[0].get("uiAmount") or 0)
            return (top / total) * 100.0
        except Exception as exc:
            log.debug("holder concentration fetch failed for %s: %s", mint[:6], exc)
            return None
