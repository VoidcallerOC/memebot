"""Documented scoring formulas for the META DETECTOR.

Every numeric score is 0-100. Acceleration is a ratio of short-window rate
to long-window rate. A ratio is not itself bullish or bearish.
"""
from __future__ import annotations

import math
from typing import Optional

WINDOW_MINUTES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "24h": 1440,
    "7d": 10080,
}

ACCEL_DECEL = 0.7
ACCEL_NORMAL_HIGH = 1.5
ACCEL_EXTREME = 4.0


def rate(value: Optional[float], window: str) -> Optional[float]:
    if value is None:
        return None
    minutes = WINDOW_MINUTES.get(window)
    if not minutes:
        return None
    return float(value) / minutes


def acceleration(short_value: Optional[float], short_window: str,
                 long_value: Optional[float], long_window: str) -> Optional[float]:
    short_rate = rate(short_value, short_window)
    long_rate = rate(long_value, long_window)
    if short_rate is None or long_rate is None:
        return None
    if long_rate <= 0:
        return None if short_rate <= 0 else float("inf")
    return short_rate / long_rate


def accel_state(acc: Optional[float]) -> str:
    if acc is None:
        return "UNVERIFIED"
    if acc == float("inf") or acc >= ACCEL_EXTREME:
        return "EXTREME_ACCELERATION"
    if acc >= ACCEL_NORMAL_HIGH:
        return "ACCELERATING"
    if acc < ACCEL_DECEL:
        return "DECELERATING"
    return "NORMAL"


def accel_to_score(acc: Optional[float]) -> Optional[float]:
    if acc is None:
        return None
    if acc <= 0:
        return 0.0
    if acc == float("inf"):
        return 100.0
    return max(0.0, min(100.0, 50.0 + 25.0 * math.log(acc, 2)))


def weighted_mean(pairs: list[tuple[float, float]]) -> Optional[float]:
    total_w = 0.0
    total = 0.0
    for value, weight in pairs:
        if weight <= 0 or value is None or math.isnan(value):
            continue
        total += value * weight
        total_w += weight
    if total_w <= 0:
        return None
    return total / total_w


def saturate_pct(raw: Optional[float], cap: float = 100.0) -> Optional[float]:
    if raw is None:
        return None
    return max(0.0, min(cap, raw))
