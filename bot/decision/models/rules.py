"""Deterministic rules-only baseline (no learned weights).

Used as comparison arm A: existing META/heuristic thresholds without a fitted
model. Still returns a TypedDecision-shaped probability via a heuristic score.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from ..schema import FEATURE_INDEX, FEATURE_SCHEMA_VERSION, N_FEATURES
from .base import LocalModel, ModelPrediction


class RulesModel(LocalModel):
    kind = "rules"
    model_version = "rules.v1"
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def __init__(
        self,
        *,
        model_version: str = "rules.v1",
        buy_threshold: float = 0.65,
    ):
        self.model_version = model_version
        self.buy_threshold = buy_threshold

    def predict_proba(self, features: Sequence[float]) -> ModelPrediction:
        if len(features) != N_FEATURES:
            return ModelPrediction(
                0.0, 0.0, self.model_version, [f"INVALID_FEATURES:len={len(features)}"]
            )
        reasons: list[str] = []
        score = 0.40  # prior: slightly below WATCH

        def f(name: str) -> float:
            return float(features[FEATURE_INDEX[name]])

        miss = f("missing_feature_frac")
        if miss > 0.5:
            return ModelPrediction(0.05, 5.0, self.model_version, ["TOO_MANY_MISSING"])

        if f("liquidity_removed") >= 0.5:
            return ModelPrediction(0.01, 1.0, self.model_version, ["LIQUIDITY_REMOVED"])

        manip = f("manipulation_risk")
        if manip >= 0 and manip >= 70:
            reasons.append("HIGH_MANIPULATION_RISK")
            score -= 0.25

        liq_q = f("liquidity_quality")
        if liq_q >= 0:
            score += (liq_q - 50.0) / 200.0
            reasons.append(f"liq_q={liq_q:.0f}")

        accel = f("vol_accel_live")
        if accel >= 0:
            if 55 <= accel <= 85:
                score += 0.15
                reasons.append("HEALTHY_ACCEL")
            elif accel > 90:
                score -= 0.10
                reasons.append("EXTREME_ACCEL")

        if f("flow_aligned") >= 0.5:
            score += 0.10
            reasons.append("FLOW_ALIGNED")
        if f("in_active_meta") >= 0.5:
            score += 0.12
            reasons.append("ACTIVE_META")
        if f("provider_fresh") < 0.5:
            score -= 0.15
            reasons.append("STALE_OR_UNVERIFIED")

        buy_pressure = f("buy_sell_ratio_5m")
        if buy_pressure >= 0.6:
            score += 0.08
            reasons.append("BUY_PRESSURE")
        elif buy_pressure <= 0.35:
            score -= 0.08
            reasons.append("SELL_PRESSURE")

        top = f("top_holder_pct")
        if top >= 0 and top > 40:
            score -= 0.20
            reasons.append("CONCENTRATED_HOLDER")

        p = max(0.0, min(1.0, score))
        if math.isnan(p):
            return ModelPrediction(0.0, 0.0, self.model_version, ["NAN_SCORE"])
        return ModelPrediction(p, p * 100.0, self.model_version, reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "buy_threshold": self.buy_threshold,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RulesModel":
        return cls(
            model_version=str(raw.get("model_version") or "rules.v1"),
            buy_threshold=float(raw.get("buy_threshold") or 0.65),
        )
