"""Deterministic component scoring for one TokenSnapshot.

Reconstructed implementation built on the surviving ``formulas.py``.

Acceleration = short_window_rate / long_window_rate.
Score        = clip(50 + 25 * log2(acceleration), 0, 100).
States       = DECELERATING < 0.7 <= NORMAL < 1.5 <= ACCELERATING < 4 <= EXTREME.

Two acceleration families are kept apart on purpose:
  * live: DexScreener 5m vs 1h trailing windows from a single tick
  * historical: 5m vs a 4h window reconstructed from prior JSONL snapshots
A live comparison is never reported as a historical one.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from .formulas import accel_state, accel_to_score, acceleration, saturate_pct, weighted_mean
from .model import (
    CONFLICTING,
    FLOW_ALIGNED,
    FLOW_VOLUME_WITHOUT_WALLETS,
    FLOW_WALLETS_WITHOUT_VOLUME,
    FRESH,
    STALE,
    UNVERIFIED,
    MarketWindow,
    Score,
    SocialWindow,
    TokenSnapshot,
    WalletWindow,
    ok,
    unverified,
)

LIVE_SHORT = "5m"
LIVE_LONG = "1h"
HISTORY_LONG = "4h"

LIQUIDITY_SHOCK_PCT = -50.0
LOW_LIQUIDITY_USD = 20_000.0
THIN_LIQUIDITY_MCAP_RATIO = 20.0
CREATOR_CONCENTRATION_HIGH_PCT = 25.0
SOCIAL_TOP_ACCOUNT_SHARE = 0.5
SOCIAL_MIN_MENTIONS = 20
SOCIAL_CONTRIBUTOR_RATIO = 0.10
BUY_PRESSURE_EXTREME = 0.85
BUY_PRESSURE_MIN_TXNS = 50

# Absolute 1h volume (USD) below which a 5m/1h ratio is not scored: with
# 5m == 1h volume the ratio is 12 whatever the size, so $0.05 would read as
# EXTREME_ACCELERATION. Eligibility only; the formula and states are unchanged.
MIN_VOLUME_FLOOR_USD = 500.0
DUST_VOLUME = "DUST_VOLUME"

ACCEL_CODES = {
    "EXTREME_ACCELERATION": "EXTREME",
    "ACCELERATING": "ACCELERATING",
    "DECELERATING": "DECELERATING",
}


def _accel_score(short_value: Optional[float], long_value: Optional[float],
                 short_window: str, long_window: str, label: str,
                 prefix: str, missing: str, volume_floor: Optional[float] = None) -> Score:
    if short_value is None or long_value is None:
        return unverified(missing, short_window=short_window, long_window=long_window)
    if volume_floor is not None and long_value < volume_floor:
        score = unverified(f"{label}: {long_window} volume below dust floor",
                           short_window=short_window, long_window=long_window,
                           short_value=short_value, long_value=long_value, volume_floor=volume_floor)
        score.reason_codes.append(DUST_VOLUME)
        return score
    acc = acceleration(short_value, short_window, long_value, long_window)
    if acc is None:
        return unverified(f"{label}: both windows are zero", short_window=short_window, long_window=long_window)
    state = accel_state(acc)
    value = accel_to_score(acc)
    codes = [f"{prefix}_{ACCEL_CODES[state]}"] if state in ACCEL_CODES else []
    acc_out = None if math.isinf(acc) else acc
    return ok(
        value,
        f"{label} = clip(50 + 25*log2(({short_window} value/{short_window} min) / ({long_window} value/{long_window} min)), 0, 100)",
        codes,
        acceleration=acc_out,
        state=state,
        short_window=short_window,
        long_window=long_window,
        short_value=short_value,
        long_value=long_value,
    )


def _pick(windows: dict, window: str, getter: Callable) -> Optional[float]:
    item = windows.get(window)
    if item is None:
        return None
    value = getter(item)
    if value is None:
        return None
    return float(value)


def score_trading(snap: TokenSnapshot) -> Score:
    """Live DexScreener 5m vs 1h volume acceleration."""
    short = _pick(snap.market, LIVE_SHORT, lambda w: w.volume_usd)
    long = _pick(snap.market, LIVE_LONG, lambda w: w.volume_usd)
    score = _accel_score(short, long, LIVE_SHORT, LIVE_LONG, "trading_velocity",
                         "VOLUME", "market provider did not supply both 5m and 1h volume",
                         volume_floor=MIN_VOLUME_FLOOR_USD)
    if score.verified:
        pressure = buy_pressure(snap.market.get(LIVE_SHORT))
        if pressure is not None:
            score.details["buy_pressure_5m"] = pressure
            if pressure >= BUY_PRESSURE_EXTREME:
                score.reason_codes.append("BUY_PRESSURE_EXTREME")
        score.details["family"] = "live"
    return score


def score_historical_trading(snap: TokenSnapshot) -> Score:
    """5m volume vs a 4h window that must come from reconstructed history."""
    window = snap.market.get(HISTORY_LONG)
    if window is None or not window.source.startswith("history"):
        return unverified("no reconstructed 4h history for this token", long_window=HISTORY_LONG)
    short = _pick(snap.market, LIVE_SHORT, lambda w: w.volume_usd)
    score = _accel_score(short, window.volume_usd, LIVE_SHORT, HISTORY_LONG, "historical_trading_velocity",
                         "HISTORICAL_VOLUME", "5m volume missing for historical comparison")
    if score.verified:
        score.details["family"] = "historical"
        score.details["history_source"] = window.source
    return score


def _wallet_metric(window: Optional[WalletWindow]) -> Optional[int]:
    if window is None:
        return None
    for value in (window.unique_buyers, window.new_holders, window.active_wallets):
        if value is not None:
            return value
    return None


def score_wallets(snap: TokenSnapshot) -> Score:
    short = _pick(snap.wallets, LIVE_SHORT, _wallet_metric)
    long = _pick(snap.wallets, LIVE_LONG, _wallet_metric)
    return _accel_score(short, long, LIVE_SHORT, LIVE_LONG, "wallet_velocity", "WALLET_GROWTH",
                        "unique buyer/seller flow provider not configured")


def score_attention(snap: TokenSnapshot) -> Score:
    short = _pick(snap.social, LIVE_SHORT, lambda w: w.mentions)
    long = _pick(snap.social, LIVE_LONG, lambda w: w.mentions)
    score = _accel_score(short, long, LIVE_SHORT, LIVE_LONG, "attention_velocity", "ATTENTION",
                         "social velocity provider not configured")
    if social_concentrated(snap.social.get(LIVE_SHORT)):
        if score.verified:
            score.reason_codes.append("SOCIAL_CONCENTRATION")
        else:
            score.details["social_concentration"] = True
    return score


def social_concentrated(window: Optional[SocialWindow]) -> bool:
    if window is None:
        return False
    if window.top_account_share is not None and window.top_account_share >= SOCIAL_TOP_ACCOUNT_SHARE:
        return True
    if (window.mentions is not None and window.unique_contributors is not None
            and window.mentions >= SOCIAL_MIN_MENTIONS
            and window.unique_contributors / max(window.mentions, 1) <= SOCIAL_CONTRIBUTOR_RATIO):
        return True
    return False


def buy_pressure(window: Optional[MarketWindow]) -> Optional[float]:
    if window is None or window.tx_buys is None or window.tx_sells is None:
        return None
    total = window.tx_buys + window.tx_sells
    if total < BUY_PRESSURE_MIN_TXNS:
        return None
    return window.tx_buys / total


def score_liquidity(snap: TokenSnapshot) -> Score:
    if snap.liquidity_usd is None:
        return unverified("market provider did not supply liquidity")
    liq = max(float(snap.liquidity_usd), 0.0)
    codes: list[str] = []
    if snap.liquidity_removed:
        codes.append("LIQUIDITY_REMOVAL")
    if snap.lp_change_pct is not None and snap.lp_change_pct <= LIQUIDITY_SHOCK_PCT:
        codes.append("LIQUIDITY_SHOCK")
    if liq < LOW_LIQUIDITY_USD:
        codes.append("LOW_LIQUIDITY")
    ratio = None
    if snap.market_cap_usd is not None and liq > 0:
        ratio = float(snap.market_cap_usd) / liq
        if ratio > THIN_LIQUIDITY_MCAP_RATIO:
            codes.append("THIN_LIQUIDITY_VS_MCAP")
    if snap.liquidity_removed or liq <= 0:
        value = 0.0
    else:
        value = saturate_pct(100.0 * math.log10(max(liq, 1.0) / 1000.0) / 3.0)
    return ok(value, "liquidity_quality = clip(100*log10(liquidity_usd/1000)/3, 0, 100); 0 if removed",
              codes, liquidity_usd=liq, lp_change_pct=snap.lp_change_pct, mcap_to_liquidity=ratio)


def score_creator(snap: TokenSnapshot) -> Score:
    if snap.creator_pct is not None:
        pct, basis = float(snap.creator_pct), "creator_pct"
    elif snap.top_holder_pct is not None:
        pct, basis = float(snap.top_holder_pct), "top_holder_pct_proxy"
    else:
        return unverified("creator-linked wallet graph provider not configured")
    codes = ["HIGH_CREATOR_CONCENTRATION"] if pct > CREATOR_CONCENTRATION_HIGH_PCT else []
    return ok(saturate_pct(pct), "creator_concentration = holder share pct (creator, else top holder proxy)",
              codes, basis=basis, top10_holder_pct=snap.top10_holder_pct)


def flow_alignment(trading: Score, wallets: Score) -> str:
    if not (trading.verified and wallets.verified):
        return UNVERIFIED
    t_state = trading.details.get("state")
    w_state = wallets.details.get("state")
    t_up = t_state in ("ACCELERATING", "EXTREME_ACCELERATION")
    w_up = w_state in ("ACCELERATING", "EXTREME_ACCELERATION")
    if t_up and not w_up:
        return FLOW_VOLUME_WITHOUT_WALLETS
    if w_up and not t_up:
        return FLOW_WALLETS_WITHOUT_VOLUME
    return FLOW_ALIGNED


def score_manipulation(snap: TokenSnapshot, trading: Score, wallets: Score, attention: Score,
                       liquidity: Score, creator: Score, alignment: str) -> Score:
    parts: list[tuple[float, float]] = []
    codes: list[str] = []
    if snap.social.get(LIVE_SHORT) is not None:
        concentrated = social_concentrated(snap.social.get(LIVE_SHORT))
        parts.append((100.0 if concentrated else 0.0, 0.25))
    pressure = trading.details.get("buy_pressure_5m") if trading.verified else None
    if pressure is not None:
        parts.append((saturate_pct((pressure - 0.5) * 200.0), 0.2))
    if creator.verified:
        parts.append((creator.value, 0.25))
    if alignment != UNVERIFIED:
        parts.append((100.0 if alignment == FLOW_VOLUME_WITHOUT_WALLETS else 0.0, 0.2))
        if alignment == FLOW_VOLUME_WITHOUT_WALLETS:
            codes.append("VOLUME_WITHOUT_WALLET_GROWTH")
    if liquidity.verified:
        shock = "LIQUIDITY_SHOCK" in liquidity.reason_codes or "LIQUIDITY_REMOVAL" in liquidity.reason_codes
        parts.append((100.0 if shock else 0.0, 0.3))
    value = weighted_mean(parts)
    if value is None:
        return unverified("no manipulation inputs available from configured providers")
    if value >= 60:
        codes.append("MANIPULATION_RISK_HIGH")
    return ok(value, "manipulation_risk = weighted mean(social concentration, buy pressure, "
                     "creator concentration, volume/wallet mismatch, liquidity shock)", codes,
              components=len(parts))


def provider_freshness(snap: TokenSnapshot) -> str:
    if snap.conflicting:
        return CONFLICTING
    if snap.stale:
        return STALE
    if snap.market or snap.social or snap.wallets or snap.liquidity_usd is not None:
        return FRESH
    return UNVERIFIED
