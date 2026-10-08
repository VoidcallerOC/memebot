"""Decision engine: features → local model → typed signal → risk firewall.

Default on critical uncertainty: NO TRADE (REJECT / risk BLOCK).
Never puts a frontier LLM in this path.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..meta.model import MetaSignal, TokenSnapshot
from .features import FeatureRecord, extract_features
from .models.base import LocalModel
from .risk_firewall import BLOCK, RiskFirewall, RiskVerdict
from .schema import FEATURE_SCHEMA_VERSION, N_FEATURES
from .signal import BUY, REJECT, WATCH, TypedDecision
from .versions import (
    BUY_CONFIDENCE_THRESHOLD,
    DECISION_ENGINE_VERSION,
    RISK_ENGINE_VERSION,
    WATCH_CONFIDENCE_THRESHOLD,
)

log = logging.getLogger(__name__)

# Hard fail-closed: if this many features are missing, never BUY.
MAX_MISSING_FRAC_FOR_BUY = 0.40
STALE_FEATURE_MAX_AGE_SECONDS = 120.0


@dataclass
class EngineResult:
    decision: TypedDecision
    risk: RiskVerdict
    features: FeatureRecord
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "risk": self.risk.to_dict(),
            "features": self.features.to_dict(),
            "latency_ms": dict(self.latency_ms),
        }


class DecisionEngine:
    def __init__(
        self,
        model: Optional[LocalModel],
        risk_firewall: RiskFirewall,
        *,
        buy_threshold: float = BUY_CONFIDENCE_THRESHOLD,
        watch_threshold: float = WATCH_CONFIDENCE_THRESHOLD,
        now_fn=None,
    ):
        self.model = model
        self.risk = risk_firewall
        self.buy_threshold = buy_threshold
        self.watch_threshold = watch_threshold
        self._now = now_fn
        self.version = DECISION_ENGINE_VERSION

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        risk_firewall: RiskFirewall,
        **kwargs: Any,
    ) -> "DecisionEngine":
        model = LocalModel.load(model_path)
        return cls(model, risk_firewall, **kwargs)

    def decide(
        self,
        snap: TokenSnapshot,
        signal: Optional[MetaSignal] = None,
        *,
        in_active_meta: bool = False,
        safety=None,
        skip_network_screen: bool = False,
        now: Optional[float] = None,
        relax_missing_threshold: bool = False,
    ) -> EngineResult:
        import time
        t0 = time.perf_counter()
        features = extract_features(
            snap, signal, in_active_meta=in_active_meta, now=now,
        )
        t1 = time.perf_counter()
        decision = self._infer(
            features, now=now or features.observed_at,
            relax_missing_threshold=relax_missing_threshold,
        )
        t2 = time.perf_counter()
        risk = self.risk.evaluate(
            features.mint, decision,
            safety=safety, skip_network_screen=skip_network_screen,
        )
        t3 = time.perf_counter()
        # If risk blocks a BUY, keep the model action for audit but record risk.
        return EngineResult(
            decision=decision,
            risk=risk,
            features=features,
            latency_ms={
                "feature_extraction_ms": (t1 - t0) * 1000.0,
                "model_inference_ms": (t2 - t1) * 1000.0,
                "risk_firewall_ms": (t3 - t2) * 1000.0,
                "end_to_end_ms": (t3 - t0) * 1000.0,
            },
        )

    def decide_from_features(
        self,
        features: FeatureRecord,
        *,
        safety=None,
        skip_network_screen: bool = True,
        now: Optional[float] = None,
        relax_missing_threshold: bool = False,
    ) -> EngineResult:
        import time
        t0 = time.perf_counter()
        decision = self._infer(
            features, now=now or features.observed_at,
            relax_missing_threshold=relax_missing_threshold,
        )
        t1 = time.perf_counter()
        risk = self.risk.evaluate(
            features.mint, decision,
            safety=safety, skip_network_screen=skip_network_screen,
        )
        t2 = time.perf_counter()
        return EngineResult(
            decision=decision,
            risk=risk,
            features=features,
            latency_ms={
                "feature_extraction_ms": 0.0,
                "model_inference_ms": (t1 - t0) * 1000.0,
                "risk_firewall_ms": (t2 - t1) * 1000.0,
                "end_to_end_ms": (t2 - t0) * 1000.0,
            },
        )

    def _infer(
        self,
        features: FeatureRecord,
        *,
        now: float,
        relax_missing_threshold: bool = False,
    ) -> TypedDecision:
        base_kwargs = dict(
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            decision_engine_version=DECISION_ENGINE_VERSION,
            risk_engine_version=RISK_ENGINE_VERSION,
            mint=features.mint,
            symbol=features.symbol,
            observed_at=features.observed_at,
            feature_vector=list(features.values),
            missing_feature_frac=features.missing_frac,
        )

        # --- failure modes: default NO TRADE ---
        if self.model is None:
            return TypedDecision(
                action=REJECT, confidence=0.0, score=0.0, probability=0.0,
                model_version="none", model_available=False,
                reasons=["MODEL_UNAVAILABLE"], **base_kwargs,
            )

        if features.feature_schema_version != FEATURE_SCHEMA_VERSION:
            return TypedDecision(
                action=REJECT, confidence=0.0, score=0.0, probability=0.0,
                model_version=getattr(self.model, "model_version", "unknown"),
                reasons=["FEATURE_SCHEMA_MISMATCH"], **base_kwargs,
            )

        if len(features.values) != N_FEATURES:
            return TypedDecision(
                action=REJECT, confidence=0.0, score=0.0, probability=0.0,
                model_version=self.model.model_version,
                reasons=["CORRUPTED_FEATURE_VECTOR"], **base_kwargs,
            )

        if any(math.isnan(v) or math.isinf(v) for v in features.values):
            return TypedDecision(
                action=REJECT, confidence=0.0, score=0.0, probability=0.0,
                model_version=self.model.model_version,
                reasons=["NAN_OR_INF_FEATURES"], **base_kwargs,
            )

        if not features.mint:
            return TypedDecision(
                action=REJECT, confidence=0.0, score=0.0, probability=0.0,
                model_version=self.model.model_version,
                reasons=["UNKNOWN_TOKEN"], **base_kwargs,
            )

        age = now - features.observed_at if features.observed_at else 0.0
        if age > STALE_FEATURE_MAX_AGE_SECONDS:
            return TypedDecision(
                action=REJECT, confidence=0.0, score=0.0, probability=0.0,
                model_version=self.model.model_version,
                reasons=[f"STALE_FEATURES:age={age:.1f}s"], **base_kwargs,
            )

        try:
            pred = self.model.predict_proba(features.values)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("model inference failed")
            return TypedDecision(
                action=REJECT, confidence=0.0, score=0.0, probability=0.0,
                model_version=getattr(self.model, "model_version", "unknown"),
                reasons=[f"MODEL_ERROR:{exc}"], **base_kwargs,
            )

        p = float(pred.probability)
        if math.isnan(p) or math.isinf(p):
            return TypedDecision(
                action=REJECT, confidence=0.0, score=0.0, probability=0.0,
                model_version=pred.model_version,
                reasons=["NAN_PROBABILITY"], **base_kwargs,
            )

        reasons = list(pred.reasons)
        # Sparse safety-bridge snapshots can exceed the missing threshold even
        # after SafetyScreener passed. When relax_missing_threshold=True the
        # model may still BUY; risk firewall retains the hard veto.
        if features.missing_frac > MAX_MISSING_FRAC_FOR_BUY and not relax_missing_threshold:
            action = REJECT
            reasons.append("MISSING_FEATURES_ABOVE_THRESHOLD")
        elif p >= self.buy_threshold:
            action = BUY
        elif p >= self.watch_threshold:
            action = WATCH
        else:
            action = REJECT

        return TypedDecision(
            action=action,
            confidence=p,
            score=pred.score,
            probability=p,
            model_version=pred.model_version,
            reasons=reasons,
            **base_kwargs,
        )
