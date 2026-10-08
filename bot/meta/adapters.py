"""Read-only provider adapters for the META DETECTOR.

Reconstructed implementation mirroring the DexScreener and Solana JSON-RPC
patterns already used by ``strategy.py`` and ``safety.py``.

Providers that the repository does not have (social firehose, unique
buyer/seller indexer, creator wallet graph) are not emulated: their windows
are simply absent from the snapshot so scoring reports them UNVERIFIED.
A missing number is None, never zero.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from .model import MarketWindow, TokenSnapshot

log = logging.getLogger(__name__)

DEXSCREENER_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/latest/v1"
DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{mint}"
HTTP_TIMEOUT = 8

WINDOW_KEYS = {"5m": "m5", "1h": "h1", "6h": "h6", "24h": "h24"}

# Boost descriptions are only served by the boosts feed; keep the latest ones
# so a later per-mint pair fetch can still classify the narrative.
_PROFILE_CACHE: dict[str, dict[str, Any]] = {}


def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _int(value: Any) -> Optional[int]:
    out = _num(value)
    return int(out) if out is not None else None


def fetch_boosted_profiles(session: Optional[requests.Session] = None) -> list[dict[str, Any]]:
    session = session or requests.Session()
    try:
        resp = session.get(DEXSCREENER_BOOSTS_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        items = resp.json() or []
    except Exception as exc:
        log.debug("boosts fetch failed: %s", exc)
        return []
    profiles: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("chainId") != "solana":
            continue
        mint = item.get("tokenAddress")
        if not mint:
            continue
        profile = {
            "mint": mint,
            "description": str(item.get("description") or ""),
            "links": [str(l.get("label") or l.get("type") or "") for l in (item.get("links") or []) if isinstance(l, dict)],
            "boost_amount": _num(item.get("amount")),
            "boost_total": _num(item.get("totalAmount")),
        }
        _PROFILE_CACHE[mint] = profile
        profiles.append(profile)
    return profiles


def fetch_boosted_mints(session: Optional[requests.Session] = None) -> list[str]:
    seen: set[str] = set()
    mints: list[str] = []
    for profile in fetch_boosted_profiles(session):
        mint = profile["mint"]
        if mint not in seen:
            seen.add(mint)
            mints.append(mint)
    return mints


def fetch_pairs(mint: str, session: Optional[requests.Session] = None) -> list[dict[str, Any]]:
    session = session or requests.Session()
    try:
        resp = session.get(DEXSCREENER_TOKEN_URL.format(mint=mint), timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        pairs = resp.json().get("pairs") or []
    except Exception as exc:
        log.debug("dexscreener pair fetch failed for %s: %s", mint[:6], exc)
        return []
    return [p for p in pairs if isinstance(p, dict) and p.get("chainId") == "solana"]


def _window(pair: dict[str, Any], key: str) -> Optional[MarketWindow]:
    volume = _num((pair.get("volume") or {}).get(key))
    txns = (pair.get("txns") or {}).get(key) or {}
    buys = _int(txns.get("buys"))
    sells = _int(txns.get("sells"))
    change = _num((pair.get("priceChange") or {}).get(key))
    if volume is None and buys is None and sells is None and change is None:
        return None
    return MarketWindow(volume_usd=volume, tx_buys=buys, tx_sells=sells,
                        price_change_pct=change, source="dexscreener")


def snapshot_from_pair(mint: str, pair: dict[str, Any], now: Optional[float] = None,
                       profile: Optional[dict[str, Any]] = None) -> TokenSnapshot:
    now = now if now is not None else time.time()
    base = pair.get("baseToken") or {}
    profile = profile or _PROFILE_CACHE.get(mint) or {}
    created = _num(pair.get("pairCreatedAt"))
    snap = TokenSnapshot(
        mint=mint,
        symbol=str(base.get("symbol") or ""),
        name=str(base.get("name") or ""),
        description=str(profile.get("description") or ""),
        observed_at=now,
        price_usd=_num(pair.get("priceUsd")),
        liquidity_usd=_num((pair.get("liquidity") or {}).get("usd")),
        market_cap_usd=_num(pair.get("marketCap")) if pair.get("marketCap") is not None else _num(pair.get("fdv")),
        pair_address=str(pair.get("pairAddress") or ""),
        pair_created_at=created / 1000.0 if created else None,
        texts=[t for t in profile.get("links") or [] if t],
        source="dexscreener",
    )
    for window, key in WINDOW_KEYS.items():
        item = _window(pair, key)
        if item is not None:
            snap.market[window] = item
    return snap


def snapshot_from_dexscreener(mint: str, session: Optional[requests.Session] = None,
                              now: Optional[float] = None,
                              profile: Optional[dict[str, Any]] = None) -> Optional[TokenSnapshot]:
    pairs = fetch_pairs(mint, session)
    if not pairs:
        return None
    best = max(pairs, key=lambda p: _num((p.get("liquidity") or {}).get("usd")) or 0.0)
    return snapshot_from_pair(mint, best, now=now, profile=profile)


def fetch_holder_concentration(mint: str, rpc_url: str,
                               session: Optional[requests.Session] = None) -> Optional[dict[str, float]]:
    """Top holder and top-10 share of supply via getTokenLargestAccounts."""
    session = session or requests.Session()
    try:
        largest = session.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "getTokenLargestAccounts", "params": [mint]},
            timeout=HTTP_TIMEOUT,
        ).json()
        supply = session.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "getTokenSupply", "params": [mint]},
            timeout=HTTP_TIMEOUT,
        ).json()
    except Exception as exc:
        log.debug("holder concentration fetch failed for %s: %s", mint[:6], exc)
        return None
    accounts = (largest.get("result") or {}).get("value") or []
    total = _num(((supply.get("result") or {}).get("value") or {}).get("uiAmount"))
    if not accounts or not total or total <= 0:
        return None
    amounts = [(_num(a.get("uiAmount")) or 0.0) for a in accounts if isinstance(a, dict)]
    if not amounts:
        return None
    return {
        "top_holder_pct": amounts[0] / total * 100.0,
        "top10_holder_pct": sum(amounts[:10]) / total * 100.0,
    }


def attach_holder_concentration(
    snap: TokenSnapshot,
    rpc_url: str,
    session: Optional[requests.Session] = None,
    *,
    force: bool = False,
) -> bool:
    """Populate top-holder share from RPC. Creator linkage stays unavailable.

    Skips the RPC call when both holder fields are already present unless
    ``force=True`` — so safety / META / decision can share one concentration fetch.
    """
    if not force and snap.top_holder_pct is not None and snap.top10_holder_pct is not None:
        return True
    if not rpc_url:
        return False
    conc = fetch_holder_concentration(snap.mint, rpc_url, session)
    if conc is None:
        return False
    snap.top_holder_pct = conc["top_holder_pct"]
    snap.top10_holder_pct = conc["top10_holder_pct"]
    return True
