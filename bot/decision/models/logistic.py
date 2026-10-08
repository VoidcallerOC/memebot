"""L2-regularized logistic regression — CPU, deterministic, tiny.

Chosen as the default local model: low latency, calibratable probabilities,
reproducible weights, easy to retrain, no GPU, no remote API.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from ..schema import FEATURE_SCHEMA_VERSION, N_FEATURES
from .base import LocalModel, ModelPrediction


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class LogisticModel(LocalModel):
    kind = "logistic"

    def __init__(
        self,
        weights: Sequence[float],
        bias: float = 0.0,
        *,
        model_version: str = "logistic.v1.untrained",
        feature_schema_version: str = FEATURE_SCHEMA_VERSION,
        feature_means: Optional[Sequence[float]] = None,
        feature_stds: Optional[Sequence[float]] = None,
        train_metrics: Optional[dict[str, Any]] = None,
    ):
        if len(weights) != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} weights, got {len(weights)}")
        self.weights = [float(w) for w in weights]
        self.bias = float(bias)
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self.feature_means = (
            [float(x) for x in feature_means]
            if feature_means is not None
            else [0.0] * N_FEATURES
        )
        self.feature_stds = (
            [float(x) if float(x) > 1e-12 else 1.0 for x in feature_stds]
            if feature_stds is not None
            else [1.0] * N_FEATURES
        )
        self.train_metrics = dict(train_metrics or {})

    def _transform(self, features: Sequence[float]) -> list[float]:
        if len(features) != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {len(features)}")
        out = []
        for i, x in enumerate(features):
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                v = self.feature_means[i]
            out.append((v - self.feature_means[i]) / self.feature_stds[i])
        return out

    def predict_proba(self, features: Sequence[float]) -> ModelPrediction:
        try:
            x = self._transform(features)
        except ValueError as exc:
            return ModelPrediction(
                probability=0.0, score=0.0, model_version=self.model_version,
                reasons=[f"INVALID_FEATURES:{exc}"],
            )
        logit = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
        if math.isnan(logit) or math.isinf(logit):
            return ModelPrediction(
                probability=0.0, score=0.0, model_version=self.model_version,
                reasons=["NAN_LOGIT"],
            )
        p = _sigmoid(logit)
        return ModelPrediction(
            probability=p,
            score=p * 100.0,
            model_version=self.model_version,
            reasons=[f"logistic_logit={logit:.4f}"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "weights": list(self.weights),
            "bias": self.bias,
            "feature_means": list(self.feature_means),
            "feature_stds": list(self.feature_stds),
            "train_metrics": dict(self.train_metrics),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LogisticModel":
        return cls(
            weights=raw["weights"],
            bias=float(raw.get("bias") or 0.0),
            model_version=str(raw.get("model_version") or "logistic.v1"),
            feature_schema_version=str(
                raw.get("feature_schema_version") or FEATURE_SCHEMA_VERSION
            ),
            feature_means=raw.get("feature_means"),
            feature_stds=raw.get("feature_stds"),
            train_metrics=raw.get("train_metrics"),
        )

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        l2: float = 1.0,
        lr: float = 0.1,
        epochs: int = 400,
        model_version: str = "logistic.v1",
        seed: int = 7,
    ) -> "LogisticModel":
        """Batch gradient descent on binary cross-entropy + L2."""
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"X must be (n, {N_FEATURES})")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X/y length mismatch")
        if X.shape[0] == 0:
            raise ValueError("empty training set")

        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
        stds = np.where(stds < 1e-12, 1.0, stds)
        Xn = (np.nan_to_num(X, nan=0.0) - means) / stds

        w = rng.normal(0.0, 0.01, size=N_FEATURES)
        b = 0.0
        n = float(X.shape[0])
        for _ in range(epochs):
            logits = Xn @ w + b
            # stable sigmoid
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            err = probs - y
            grad_w = (Xn.T @ err) / n + l2 * w
            grad_b = float(err.mean())
            w -= lr * grad_w
            b -= lr * grad_b

        # train accuracy (in-sample — only for artifact metadata, not claims)
        probs = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -50, 50)))
        preds = (probs >= 0.5).astype(np.float64)
        acc = float((preds == y).mean()) if n else 0.0
        return cls(
            weights=w.tolist(),
            bias=float(b),
            model_version=model_version,
            feature_means=means.tolist(),
            feature_stds=stds.tolist(),
            train_metrics={"in_sample_accuracy": acc, "n_train": int(n), "l2": l2, "epochs": epochs},
        )
