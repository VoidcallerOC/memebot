"""Typed decision signal. The model NEVER authorizes a transaction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

BUY = "BUY"
WATCH = "WATCH"
REJECT = "REJECT"
ACTIONS = frozenset({BUY, WATCH, REJECT})


@dataclass
class TypedDecision:
    action: str
    confidence: float
    score: float  # 0-100 ranking score derived from calibrated probability
    model_version: str
    feature_schema_version: str
    decision_engine_version: str
    risk_engine_version: str
    probability: float = 0.0  # P(target=1)
    reasons: list[str] = field(default_factory=list)
    mint: str = ""
    symbol: str = ""
    observed_at: float = 0.0
    feature_vector: Optional[list[float]] = None
    missing_feature_frac: float = 0.0
    model_available: bool = True

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"invalid action {self.action!r}")
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.probability = float(max(0.0, min(1.0, self.probability)))
        self.score = float(max(0.0, min(100.0, self.score)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
