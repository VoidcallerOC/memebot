"""Metrics, calibration, and confusion matrix for the local decision model.

Trading expectancy matters more than generic accuracy. Never treat
confidence=0.95 as proof of 95% accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .models.base import LocalModel


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / n if n else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    @property
    def fnr(self) -> float:
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": self.precision, "recall": self.recall,
            "accuracy": self.accuracy, "false_positive_rate": self.fpr,
            "false_negative_rate": self.fnr,
        }


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_rate: float


@dataclass
class EvalReport:
    n: int
    threshold: float
    confusion: Confusion
    calibration: list[CalibrationBin] = field(default_factory=list)
    ece: float = 0.0  # expected calibration error
    brier: float = 0.0
    mean_probability: float = 0.0
    # Trading-oriented: average label among predicted BUYs (precision proxy)
    buy_hit_rate: float = 0.0
    # Expected value under unit stake if BUY when p>=threshold and payoff
    # assumes +1 on label=1 and -0.5 on label=0 (illustrative, not live PnL).
    illustrative_expectancy: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "threshold": self.threshold,
            "confusion": self.confusion.to_dict(),
            "calibration": [
                {
                    "lower": b.lower, "upper": b.upper, "count": b.count,
                    "mean_confidence": b.mean_confidence,
                    "empirical_rate": b.empirical_rate,
                }
                for b in self.calibration
            ],
            "ece": self.ece,
            "brier": self.brier,
            "mean_probability": self.mean_probability,
            "buy_hit_rate": self.buy_hit_rate,
            "illustrative_expectancy": self.illustrative_expectancy,
            "notes": list(self.notes),
        }


def predict_many(model: LocalModel, X: np.ndarray) -> np.ndarray:
    probs = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        probs[i] = model.predict_proba(X[i]).probability
    return probs


def evaluate(
    model: LocalModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.65,
    n_bins: int = 10,
) -> EvalReport:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    probs = predict_many(model, X)
    preds = (probs >= threshold).astype(np.float64)

    cm = Confusion()
    for p, t in zip(preds, y):
        if p == 1 and t == 1:
            cm.tp += 1
        elif p == 1 and t == 0:
            cm.fp += 1
        elif p == 0 and t == 0:
            cm.tn += 1
        else:
            cm.fn += 1

    # Calibration bins
    bins: list[CalibrationBin] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(CalibrationBin(lo, hi, 0, 0.0, 0.0))
            continue
        mean_c = float(probs[mask].mean())
        emp = float(y[mask].mean())
        bins.append(CalibrationBin(lo, hi, count, mean_c, emp))
        ece += (count / n) * abs(mean_c - emp)

    brier = float(np.mean((probs - y) ** 2)) if n else 0.0
    buy_mask = preds == 1
    buy_hit = float(y[buy_mask].mean()) if buy_mask.any() else 0.0
    # Illustrative EV only — NOT verified live profitability.
    if buy_mask.any():
        # +1 unit on hit, -0.5 unit on miss (asymmetric stop), no costs.
        ev = float(np.mean(np.where(y[buy_mask] == 1, 1.0, -0.5)))
    else:
        ev = 0.0

    return EvalReport(
        n=n,
        threshold=threshold,
        confusion=cm,
        calibration=bins,
        ece=ece,
        brier=brier,
        mean_probability=float(probs.mean()) if n else 0.0,
        buy_hit_rate=buy_hit,
        illustrative_expectancy=ev,
        notes=[
            "illustrative_expectancy uses synthetic payoff +1/-0.5 without fees; "
            "NOT live or shadow PnL evidence",
            "confidence is not accuracy; see calibration / ECE / Brier",
        ],
    )
