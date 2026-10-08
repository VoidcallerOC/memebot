"""Shallow decision-stump ensemble (AdaBoost-lite) for candidate comparison.

CPU-only, pure numpy, tiny memory. Useful as a second lightweight candidate
against logistic regression — not a deep model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..schema import FEATURE_SCHEMA_VERSION, N_FEATURES
from .base import LocalModel, ModelPrediction


@dataclass
class Stump:
    feature_index: int
    threshold: float
    left_value: float  # predicted probability contribution when x < threshold
    right_value: float
    weight: float


class StumpEnsembleModel(LocalModel):
    kind = "stumps"

    def __init__(
        self,
        stumps: Sequence[Stump],
        bias: float = 0.0,
        *,
        model_version: str = "stumps.v1",
        feature_schema_version: str = FEATURE_SCHEMA_VERSION,
        train_metrics: Optional[dict[str, Any]] = None,
    ):
        self.stumps = list(stumps)
        self.bias = float(bias)
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self.train_metrics = dict(train_metrics or {})

    def predict_proba(self, features: Sequence[float]) -> ModelPrediction:
        if len(features) != N_FEATURES:
            return ModelPrediction(
                0.0, 0.0, self.model_version, [f"INVALID_FEATURES:len={len(features)}"]
            )
        total_w = sum(s.weight for s in self.stumps) or 1.0
        raw = self.bias
        for s in self.stumps:
            x = float(features[s.feature_index])
            if math.isnan(x) or math.isinf(x):
                continue
            raw += s.weight * (s.left_value if x < s.threshold else s.right_value)
        p = max(0.0, min(1.0, raw / total_w if self.stumps else 0.5))
        return ModelPrediction(p, p * 100.0, self.model_version, [f"n_stumps={len(self.stumps)}"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "bias": self.bias,
            "stumps": [
                {
                    "feature_index": s.feature_index,
                    "threshold": s.threshold,
                    "left_value": s.left_value,
                    "right_value": s.right_value,
                    "weight": s.weight,
                }
                for s in self.stumps
            ],
            "train_metrics": dict(self.train_metrics),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StumpEnsembleModel":
        stumps = [
            Stump(
                feature_index=int(s["feature_index"]),
                threshold=float(s["threshold"]),
                left_value=float(s["left_value"]),
                right_value=float(s["right_value"]),
                weight=float(s["weight"]),
            )
            for s in (raw.get("stumps") or [])
        ]
        return cls(
            stumps=stumps,
            bias=float(raw.get("bias") or 0.0),
            model_version=str(raw.get("model_version") or "stumps.v1"),
            feature_schema_version=str(
                raw.get("feature_schema_version") or FEATURE_SCHEMA_VERSION
            ),
            train_metrics=raw.get("train_metrics"),
        )

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_stumps: int = 24,
        model_version: str = "stumps.v1",
        seed: int = 11,
    ) -> "StumpEnsembleModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.shape[1] != N_FEATURES:
            raise ValueError(f"X must be (n, {N_FEATURES})")
        n = X.shape[0]
        if n == 0:
            raise ValueError("empty training set")
        rng = np.random.default_rng(seed)
        weights = np.ones(n) / n
        stumps: list[Stump] = []

        for _ in range(n_stumps):
            best = None
            best_err = 1.0
            # Sample a few feature indices each round for speed.
            feats = rng.choice(N_FEATURES, size=min(8, N_FEATURES), replace=False)
            for fi in feats:
                col = X[:, fi]
                # Candidate thresholds: quantiles
                qs = np.quantile(col, [0.2, 0.4, 0.5, 0.6, 0.8])
                for thr in qs:
                    left_mask = col < thr
                    if left_mask.all() or (~left_mask).all():
                        continue
                    left_y = y[left_mask]
                    right_y = y[~left_mask]
                    left_val = float(left_y.mean()) if left_y.size else 0.5
                    right_val = float(right_y.mean()) if right_y.size else 0.5
                    pred = np.where(left_mask, left_val >= 0.5, right_val >= 0.5).astype(float)
                    err = float(np.sum(weights * (pred != y)))
                    if err < best_err:
                        best_err = err
                        best = Stump(int(fi), float(thr), left_val, right_val, 1.0)
            if best is None or best_err >= 0.5:
                break
            # AdaBoost-ish weight update on hard 0/1 prediction
            eps = max(best_err, 1e-6)
            alpha = 0.5 * math.log((1.0 - eps) / eps)
            best.weight = alpha
            left_mask = X[:, best.feature_index] < best.threshold
            pred = np.where(
                left_mask, best.left_value >= 0.5, best.right_value >= 0.5
            ).astype(float)
            weights *= np.exp(alpha * (pred != y).astype(float))
            weights /= weights.sum()
            stumps.append(best)

        model = cls(stumps=stumps, bias=0.5, model_version=model_version)
        # in-sample accuracy metadata only
        correct = 0
        for i in range(n):
            p = model.predict_proba(X[i]).probability
            correct += int((p >= 0.5) == bool(y[i]))
        model.train_metrics = {
            "in_sample_accuracy": correct / n,
            "n_train": int(n),
            "n_stumps": len(stumps),
        }
        return model
