"""Local model interface. Models classify/rank; they never authorize trades."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class ModelPrediction:
    probability: float  # P(target=1), ideally calibrated
    score: float  # 0-100
    model_version: str
    reasons: list[str]


class LocalModel(ABC):
    model_version: str = "unversioned"
    feature_schema_version: str = ""

    @abstractmethod
    def predict_proba(self, features: Sequence[float]) -> ModelPrediction:
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LocalModel":
        ...

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "LocalModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = raw.get("kind")
        if kind == "logistic":
            from .logistic import LogisticModel
            return LogisticModel.from_dict(raw)
        if kind == "rules":
            from .rules import RulesModel
            return RulesModel.from_dict(raw)
        if kind == "stumps":
            from .stumps import StumpEnsembleModel
            return StumpEnsembleModel.from_dict(raw)
        raise ValueError(f"unknown model kind {kind!r}")
