"""Structured feature extraction from META snapshots / signals.

Pure arithmetic. No prose. Missing values follow schema.missing rules.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..meta.model import FLOW_ALIGNED, FRESH, MetaReport, MetaSignal, OK, TokenSnapshot
from .schema import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    N_FEATURES,
    empty_vector,
)


@dataclass
class FeatureRecord:
    """One timestamped feature vector ready for the local model."""

    mint: str
    observed_at: float
    values: list[float]
    missing_mask: list[bool]
    missing_frac: float
    symbol: str = ""
    source: str = ""
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def to_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "observed_at": self.observed_at,
            "feature_schema_version": self.feature_schema_version,
            "values": list(self.values),
            "missing_mask": list(self.missing_mask),
            "missing_frac": self.missing_frac,
            "source": self.source,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeatureRecord":
        values = [float(x) for x in raw["values"]]
        if len(values) != N_FEATURES:
            raise ValueError(
                f"feature length {len(values)} != schema {N_FEATURES} "
                f"(schema={raw.get('feature_schema_version')})"
            )
        mask = [bool(x) for x in raw.get("missing_mask") or [False] * N_FEATURES]
        return cls(
            mint=str(raw.get("mint") or ""),
            symbol=str(raw.get("symbol") or ""),
            observed_at=float(raw.get("observed_at") or 0.0),
            values=values,
            missing_mask=mask,
            missing_frac=float(raw.get("missing_frac") or 0.0),
            source=str(raw.get("source") or ""),
            feature_schema_version=str(
                raw.get("feature_schema_version") or FEATURE_SCHEMA_VERSION
            ),
        )


def _log1p(value: Optional[float]) -> tuple[float, bool]:
    if value is None or value < 0 or math.isnan(value) or math.isinf(value):
        return 0.0, True
    return math.log1p(float(value)), False


def _score_or_missing(score_obj: Any) -> tuple[float, bool]:
    if score_obj is None:
        return -1.0, True
    status = getattr(score_obj, "status", None)
    value = getattr(score_obj, "value", None)
    if status != OK or value is None:
        return -1.0, True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return -1.0, True
    if math.isnan(v) or math.isinf(v):
        return -1.0, True
    return v, False


def _ratio(buys: Optional[int], sells: Optional[int]) -> tuple[float, bool]:
    if buys is None or sells is None:
        return 0.5, True
    total = int(buys) + int(sells)
    if total <= 0:
        return 0.5, True
    return float(buys) / float(total), False


def _opt_float(value: Optional[float], default: float = 0.0) -> tuple[float, bool]:
    if value is None:
        return default, True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default, True
    if math.isnan(v) or math.isinf(v):
        return default, True
    return v, False


def extract_features(
    snap: TokenSnapshot,
    signal: Optional[MetaSignal] = None,
    *,
    in_active_meta: bool = False,
    now: Optional[float] = None,
) -> FeatureRecord:
    """Build a FeatureRecord from a META TokenSnapshot (+ optional MetaSignal)."""
    now = now if now is not None else time.time()
    observed_at = float(snap.observed_at or now)
    values = empty_vector()
    missing = [False] * N_FEATURES

    def set_feat(name: str, value: float, is_missing: bool) -> None:
        idx = FEATURE_INDEX[name]
        values[idx] = float(value)
        missing[idx] = bool(is_missing)

    liq, miss = _log1p(snap.liquidity_usd)
    set_feat("log_liquidity_usd", liq, miss)
    mcap, miss = _log1p(snap.market_cap_usd)
    set_feat("log_market_cap_usd", mcap, miss)

    if snap.pair_created_at and snap.pair_created_at > 0:
        age_h = max(0.0, (observed_at - float(snap.pair_created_at)) / 3600.0)
        set_feat("token_age_hours", age_h, False)
    else:
        set_feat("token_age_hours", 0.0, True)

    m5 = snap.market.get("5m")
    m1 = snap.market.get("1h")
    vol5, miss = _opt_float(m5.volume_usd if m5 else None)
    set_feat("vol_5m_usd", vol5, miss)
    vol1, miss = _opt_float(m1.volume_usd if m1 else None)
    set_feat("vol_1h_usd", vol1, miss)

    if signal is not None:
        v, miss = _score_or_missing(signal.trading_velocity)
        set_feat("vol_accel_live", v, miss)
        v, miss = _score_or_missing(signal.historical_trading_velocity)
        set_feat("vol_accel_hist", v, miss)
        v, miss = _score_or_missing(signal.liquidity_quality)
        set_feat("liquidity_quality", v, miss)
        v, miss = _score_or_missing(signal.manipulation_risk)
        set_feat("manipulation_risk", v, miss)
        v, miss = _score_or_missing(signal.wallet_velocity)
        set_feat("wallet_velocity", v, miss)
        v, miss = _score_or_missing(signal.attention_velocity)
        set_feat("attention_velocity", v, miss)
        set_feat("flow_aligned", 1.0 if signal.flow_alignment == FLOW_ALIGNED else 0.0, False)
        set_feat("provider_fresh", 1.0 if signal.provider_freshness == FRESH else 0.0, False)
    else:
        set_feat("vol_accel_live", -1.0, True)
        set_feat("vol_accel_hist", -1.0, True)
        set_feat("liquidity_quality", -1.0, True)
        set_feat("manipulation_risk", -1.0, True)
        set_feat("wallet_velocity", -1.0, True)
        set_feat("attention_velocity", -1.0, True)
        set_feat("flow_aligned", 0.0, True)
        set_feat("provider_fresh", 0.0, True)

    ratio, miss = _ratio(m5.tx_buys if m5 else None, m5.tx_sells if m5 else None)
    set_feat("buy_sell_ratio_5m", ratio, miss)
    ratio, miss = _ratio(m1.tx_buys if m1 else None, m1.tx_sells if m1 else None)
    set_feat("buy_sell_ratio_1h", ratio, miss)

    if m5 and m5.tx_buys is not None and m5.tx_sells is not None:
        set_feat("tx_velocity_5m", float(m5.tx_buys + m5.tx_sells), False)
    else:
        set_feat("tx_velocity_5m", 0.0, True)

    for name, attr, default in (
        ("top_holder_pct", snap.top_holder_pct, -1.0),
        ("top10_holder_pct", snap.top10_holder_pct, -1.0),
        ("creator_pct", snap.creator_pct, -1.0),
    ):
        v, miss = _opt_float(attr, default)
        # treat default sentinel as missing when attr was None
        set_feat(name, v if not miss else default, miss)

    set_feat("liquidity_removed", 1.0 if snap.liquidity_removed else 0.0, False)
    lp, miss = _opt_float(snap.lp_change_pct, 0.0)
    set_feat("lp_change_pct", lp, miss)
    set_feat("in_active_meta", 1.0 if in_active_meta else 0.0, False)

    pc5, miss = _opt_float(m5.price_change_pct if m5 else None)
    set_feat("price_change_5m_pct", pc5, miss)
    pc1, miss = _opt_float(m1.price_change_pct if m1 else None)
    set_feat("price_change_1h_pct", pc1, miss)

    # Primary features for missing_frac exclude the derived missing_feature_frac itself.
    primary = [i for i, name in enumerate(FEATURE_NAMES) if name != "missing_feature_frac"]
    miss_count = sum(1 for i in primary if missing[i])
    miss_frac = miss_count / max(1, len(primary))
    set_feat("missing_feature_frac", miss_frac, False)

    # Sanitize NaN/inf that somehow leaked — fail closed to schema defaults.
    for i, v in enumerate(values):
        if math.isnan(v) or math.isinf(v):
            values[i] = empty_vector()[i]
            missing[i] = True
            miss_frac = sum(1 for j in primary if missing[j]) / max(1, len(primary))
            values[FEATURE_INDEX["missing_feature_frac"]] = miss_frac

    return FeatureRecord(
        mint=snap.mint,
        symbol=snap.symbol or (signal.symbol if signal else ""),
        observed_at=observed_at,
        values=values,
        missing_mask=missing,
        missing_frac=miss_frac,
        source=snap.source or "meta",
    )


def extract_from_report(report: MetaReport, snapshots: Sequence[TokenSnapshot]) -> list[FeatureRecord]:
    """Extract features for every signal in a MetaReport, joining snapshots by mint."""
    by_mint = {s.mint: s for s in snapshots}
    active = set(report.candidate_mints())
    out: list[FeatureRecord] = []
    for signal in report.signals:
        snap = by_mint.get(signal.token)
        if snap is None:
            # Reconstruct a minimal snapshot from the signal alone.
            snap = TokenSnapshot(
                mint=signal.token,
                symbol=signal.symbol,
                observed_at=signal.observed_at,
                price_usd=signal.price_usd,
                liquidity_usd=signal.liquidity_usd,
                market_cap_usd=signal.market_cap_usd,
            )
        out.append(
            extract_features(
                snap, signal,
                in_active_meta=signal.token in active,
                now=report.generated_at,
            )
        )
    return out
