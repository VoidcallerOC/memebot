"""Outcome labeling for the decision target.

Target: hit_plus10_before_minus5
Entry price MUST be latency-adjusted (price at detection + latency), never the
observed wallet fill price.

Incomplete / censored paths (observation stream ended before the full target
horizon without hitting TP or stop) are UNLABELED — never invented as label=0.
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

# Explicit unlabeled sentinel for incomplete / censored observations.
UNLABELED: None = None


@dataclass(frozen=True)
class PriceTick:
    ts: float
    price: float


@dataclass
class OutcomeLabel:
    target_name: str
    label: Optional[int]  # 1 = TP before stop, 0 = stop or true timeout, None = UNLABELED
    entry_ts: float
    entry_price: float
    detection_ts: float
    detection_latency_seconds: float
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    exit_ts: Optional[float]
    exit_price: Optional[float]
    exit_reason: str  # take_profit | stop | timeout | incomplete | no_path | invalid
    horizon_seconds: float
    take_profit_pct: float
    stop_pct: float

    @property
    def is_labeled(self) -> bool:
        return self.label is not None

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "label": self.label,
            "is_labeled": self.is_labeled,
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

    A sample is UNLABELED (label=None) when:
      - no entry tick exists (no_path)
      - entry price is invalid
      - the path ends before the full horizon without hitting TP or stop

    True timeout (full horizon observed, neither TP nor stop) → label=0.
    Never invent outcomes from truncated streams.
    """
    entry_ts = detection_ts + float(latency_seconds)
    entry_tick = _price_at_or_after(path, entry_ts)
    if entry_tick is None:
        return OutcomeLabel(
            target_name=TARGET_NAME, label=UNLABELED,
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
            target_name=TARGET_NAME, label=UNLABELED,
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
    label: Optional[int] = 0
    last_in_window_ts: Optional[float] = None

    for tick in path:
        if tick.ts < entry_tick.ts:
            continue
        if tick.ts > deadline:
            break
        if tick.price <= 0:
            continue
        last_in_window_ts = tick.ts
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

    # Incomplete / censored: stream ended before full horizon without TP/SL.
    # Require a tick at or after the deadline (or a resolved TP/SL) to label.
    if exit_reason == "timeout":
        covered = last_in_window_ts is not None and last_in_window_ts >= deadline
        # Also accept coverage if any tick exists at/after deadline in the raw path
        # (loop breaks when tick.ts > deadline, so check path for boundary coverage).
        if not covered:
            for tick in path:
                if tick.ts >= deadline and tick.price > 0:
                    covered = True
                    break
        if not covered:
            return OutcomeLabel(
                target_name=TARGET_NAME,
                label=UNLABELED,
                entry_ts=entry_tick.ts,
                entry_price=entry,
                detection_ts=detection_ts,
                detection_latency_seconds=latency_seconds,
                max_favorable_excursion_pct=mfe,
                max_adverse_excursion_pct=mae,
                exit_ts=last_in_window_ts,
                exit_price=None,
                exit_reason="incomplete",
                horizon_seconds=horizon_seconds,
                take_profit_pct=take_profit_pct,
                stop_pct=stop_pct,
            )

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
