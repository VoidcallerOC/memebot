"""Shadow / paper journal for the local decision engine.

Simulated entry MUST use the price available at detection + latency, never the
observed wallet entry price. Rejected opportunities get counterfactual
outcomes so we can tell true-positive rejections from false-positive ones.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..paths import resolve_data_path
from .engine import EngineResult
from .labels import OutcomeLabel, PriceTick, label_path
from .schema import DEFAULT_DETECTION_LATENCY_SECONDS
from .signal import BUY, REJECT


DEFAULT_SHADOW_FILE = "decision_shadow.jsonl"


@dataclass
class ShadowRecord:
    kind: str  # shadow_decision | counterfactual
    timestamp: float
    mint: str
    symbol: str
    model_version: str
    feature_schema_version: str
    decision_engine_version: str
    risk_engine_version: str
    action: str
    confidence: float
    score: float
    probability: float
    risk_status: str
    risk_reasons: list[str]
    features: list[float]
    missing_feature_frac: float
    detection_ts: float
    detection_latency_seconds: float
    simulated_entry_ts: Optional[float] = None
    simulated_entry_price: Optional[float] = None
    simulated_exit_ts: Optional[float] = None
    simulated_exit_price: Optional[float] = None
    simulated_exit_reason: Optional[str] = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    fee_pct: float = 0.3
    slippage_pct: float = 1.0
    gross_pnl_pct: Optional[float] = None
    net_pnl_pct: Optional[float] = None
    max_favorable_excursion_pct: Optional[float] = None
    max_adverse_excursion_pct: Optional[float] = None
    rejection_reason: Optional[str] = None
    model_rejection: Optional[str] = None
    risk_rejection: Optional[str] = None
    outcome_label: Optional[int] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timestamp": self.timestamp,
            "mint": self.mint,
            "symbol": self.symbol,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "decision_engine_version": self.decision_engine_version,
            "risk_engine_version": self.risk_engine_version,
            "action": self.action,
            "confidence": self.confidence,
            "score": self.score,
            "probability": self.probability,
            "risk_status": self.risk_status,
            "risk_reasons": list(self.risk_reasons),
            "features": list(self.features),
            "missing_feature_frac": self.missing_feature_frac,
            "detection_ts": self.detection_ts,
            "detection_latency_seconds": self.detection_latency_seconds,
            "simulated_entry_ts": self.simulated_entry_ts,
            "simulated_entry_price": self.simulated_entry_price,
            "simulated_exit_ts": self.simulated_exit_ts,
            "simulated_exit_price": self.simulated_exit_price,
            "simulated_exit_reason": self.simulated_exit_reason,
            "latency_ms": dict(self.latency_ms),
            "fee_pct": self.fee_pct,
            "slippage_pct": self.slippage_pct,
            "gross_pnl_pct": self.gross_pnl_pct,
            "net_pnl_pct": self.net_pnl_pct,
            "max_favorable_excursion_pct": self.max_favorable_excursion_pct,
            "max_adverse_excursion_pct": self.max_adverse_excursion_pct,
            "rejection_reason": self.rejection_reason,
            "model_rejection": self.model_rejection,
            "risk_rejection": self.risk_rejection,
            "outcome_label": self.outcome_label,
            "notes": list(self.notes),
        }


class ShadowJournal:
    def __init__(self, path: Optional[str] = None):
        self.path = path or resolve_data_path(
            os.getenv("DECISION_SHADOW_FILE", DEFAULT_SHADOW_FILE).strip() or DEFAULT_SHADOW_FILE
        )

    def append(self, record: ShadowRecord) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def load(self) -> list[dict[str, Any]]:
        path = Path(self.path)
        if not path.exists():
            return []
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows


def _apply_costs(gross_pct: float, fee_pct: float, slippage_pct: float) -> float:
    """Round-trip cost: fee+slippage on entry and exit, in percent points."""
    cost = 2.0 * (fee_pct + slippage_pct)
    return gross_pct - cost


def record_from_engine(
    result: EngineResult,
    *,
    path: Optional[Sequence[PriceTick]] = None,
    detection_latency_seconds: float = DEFAULT_DETECTION_LATENCY_SECONDS,
    fee_pct: float = 0.3,
    slippage_pct: float = 1.0,
    wallet_entry_price: Optional[float] = None,
    now: Optional[float] = None,
) -> ShadowRecord:
    """Build a shadow or counterfactual record from an EngineResult.

    If ``wallet_entry_price`` is supplied it is recorded only as a note — it is
    NEVER used as the simulated fill.
    """
    now = now if now is not None else time.time()
    d = result.decision
    risk = result.risk
    detection_ts = d.observed_at or now
    notes: list[str] = []
    if wallet_entry_price is not None:
        notes.append(
            f"wallet_entry_price={wallet_entry_price} IGNORED_FOR_SIMULATED_FILL"
        )

    outcome: Optional[OutcomeLabel] = None
    if path:
        outcome = label_path(
            path, detection_ts, latency_seconds=detection_latency_seconds,
        )
        notes.append(f"outcome_exit_reason={outcome.exit_reason}")

    would_enter = d.action == BUY and risk.allowed
    kind = "shadow_decision" if would_enter else "counterfactual"

    model_rej = None
    risk_rej = None
    rejection = None
    if d.action != BUY:
        model_rej = d.action + ":" + ",".join(d.reasons[:3])
        rejection = model_rej
    if not risk.allowed:
        risk_rej = ",".join(risk.reasons[:5])
        rejection = (rejection + "|" if rejection else "") + "RISK:" + risk_rej

    gross = None
    net = None
    entry_ts = outcome.entry_ts if outcome else None
    entry_px = outcome.entry_price if outcome else None
    exit_ts = outcome.exit_ts if outcome else None
    exit_px = outcome.exit_price if outcome else None
    exit_reason = outcome.exit_reason if outcome else None
    mfe = outcome.max_favorable_excursion_pct if outcome else None
    mae = outcome.max_adverse_excursion_pct if outcome else None
    label = outcome.label if outcome else None

    if outcome and outcome.entry_price > 0 and outcome.exit_price:
        # Slippage: buy above entry, sell below exit.
        fill_in = outcome.entry_price * (1.0 + slippage_pct / 100.0)
        fill_out = outcome.exit_price * (1.0 - slippage_pct / 100.0)
        gross = (fill_out / fill_in - 1.0) * 100.0
        # fees separate from slippage already applied to fills:
        net = gross - 2.0 * fee_pct
    elif outcome and outcome.entry_price > 0 and outcome.exit_reason == "timeout":
        # Use last known MFE/MAE neutral timeout → 0 gross before costs.
        gross = 0.0
        net = _apply_costs(0.0, fee_pct, slippage_pct)

    return ShadowRecord(
        kind=kind,
        timestamp=now,
        mint=d.mint,
        symbol=d.symbol,
        model_version=d.model_version,
        feature_schema_version=d.feature_schema_version,
        decision_engine_version=d.decision_engine_version,
        risk_engine_version=d.risk_engine_version,
        action=d.action,
        confidence=d.confidence,
        score=d.score,
        probability=d.probability,
        risk_status=risk.status,
        risk_reasons=list(risk.reasons),
        features=list(result.features.values),
        missing_feature_frac=result.features.missing_frac,
        detection_ts=detection_ts,
        detection_latency_seconds=detection_latency_seconds,
        simulated_entry_ts=entry_ts,
        simulated_entry_price=entry_px,
        simulated_exit_ts=exit_ts,
        simulated_exit_price=exit_px,
        simulated_exit_reason=exit_reason,
        latency_ms=dict(result.latency_ms),
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        gross_pnl_pct=gross,
        net_pnl_pct=net,
        max_favorable_excursion_pct=mfe,
        max_adverse_excursion_pct=mae,
        rejection_reason=rejection,
        model_rejection=model_rej,
        risk_rejection=risk_rej,
        outcome_label=label,
        notes=notes,
    )
