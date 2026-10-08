"""Outcome labeling for the decision target.

Target: hit_plus10_before_minus5
Entry price MUST be latency-adjusted (price at detection + latency), never the
observed wallet fill price.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .schema import (
    DEFAULT_DETECTION_LATENCY_SECONDS,
    TARGET_HORIZON_SECONDS,
    TARGET_NAME,
    TARGET_STOP_PCT,
    TARGET_TAKE_PROFIT_PCT,
)


@dataclass(frozen=True)
class PriceTick:
    ts: float
    price: float


@dataclass
class OutcomeLabel:
    target_name: str
    label: int  # 1 = hit TP before stop, 0 otherwise
    entry_ts: float
    entry_price: float
    detection_ts: float
    detection_latency_seconds: float
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    exit_ts: Optional[float]
    exit_price: Optional[float]
    exit_reason: str  # take_profit | stop | timeout | no_path | invalid
    horizon_seconds: float
    take_profit_pct: float
    stop_pct: float

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "label": self.label,
            "entry_ts": self.entry_ts,
            "entry_price": self.entry_price,
            "detection_ts": self.detection_ts,
            "detection_latency_seconds": self.detection_latency_seconds,
            "max_favorable_excursion_pct": self.max_favorable_excursion_pct,
            "max_adverse_excursion_pct": self.max_adverse_excursion_pct,
            "exit_ts": self.exit_ts,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "horizon_seconds": self.horizon_seconds,
            "take_profit_pct": self.take_profit_pct,
            "stop_pct": self.stop_pct,
        }


def _price_at_or_after(path: Sequence[PriceTick], ts: float) -> Optional[PriceTick]:
    for tick in path:
        if tick.ts >= ts and tick.price > 0:
            return tick
    return None


def label_path(
    path: Sequence[PriceTick],
    detection_ts: float,
    *,
    latency_seconds: float = DEFAULT_DETECTION_LATENCY_SECONDS,
    take_profit_pct: float = TARGET_TAKE_PROFIT_PCT,
    stop_pct: float = TARGET_STOP_PCT,
    horizon_seconds: float = TARGET_HORIZON_SECONDS,
) -> OutcomeLabel:
    """Label one opportunity from a forward price path.

    Simulated entry uses the first tick at or after detection_ts + latency.
    """
    entry_ts = detection_ts + float(latency_seconds)
    entry_tick = _price_at_or_after(path, entry_ts)
    if entry_tick is None:
        return OutcomeLabel(
            target_name=TARGET_NAME, label=0,
            entry_ts=entry_ts, entry_price=0.0,
            detection_ts=detection_ts, detection_latency_seconds=latency_seconds,
            max_favorable_excursion_pct=0.0, max_adverse_excursion_pct=0.0,
            exit_ts=None, exit_price=None, exit_reason="no_path",
            horizon_seconds=horizon_seconds,
            take_profit_pct=take_profit_pct, stop_pct=stop_pct,
        )
    entry = entry_tick.price
    if entry <= 0:
        return OutcomeLabel(
            target_name=TARGET_NAME, label=0,
            entry_ts=entry_tick.ts, entry_price=entry,
            detection_ts=detection_ts, detection_latency_seconds=latency_seconds,
            max_favorable_excursion_pct=0.0, max_adverse_excursion_pct=0.0,
            exit_ts=None, exit_price=None, exit_reason="invalid",
            horizon_seconds=horizon_seconds,
            take_profit_pct=take_profit_pct, stop_pct=stop_pct,
        )

    tp = entry * (1.0 + take_profit_pct / 100.0)
    sl = entry * (1.0 - stop_pct / 100.0)
    deadline = entry_tick.ts + horizon_seconds
    mfe = 0.0
    mae = 0.0
    exit_ts = None
    exit_price = None
    exit_reason = "timeout"
    label = 0

    for tick in path:
        if tick.ts < entry_tick.ts:
            continue
        if tick.ts > deadline:
            break
        if tick.price <= 0:
            continue
        ret = (tick.price / entry - 1.0) * 100.0
        mfe = max(mfe, ret)
        mae = min(mae, ret)
        if tick.price >= tp:
            label = 1
            exit_ts, exit_price, exit_reason = tick.ts, tick.price, "take_profit"
            break
        if tick.price <= sl:
            label = 0
            exit_ts, exit_price, exit_reason = tick.ts, tick.price, "stop"
            break

    return OutcomeLabel(
        target_name=TARGET_NAME,
        label=label,
        entry_ts=entry_tick.ts,
        entry_price=entry,
        detection_ts=detection_ts,
        detection_latency_seconds=latency_seconds,
        max_favorable_excursion_pct=mfe,
        max_adverse_excursion_pct=mae,
        exit_ts=exit_ts,
        exit_price=exit_price,
        exit_reason=exit_reason,
        horizon_seconds=horizon_seconds,
        take_profit_pct=take_profit_pct,
        stop_pct=stop_pct,
    )
