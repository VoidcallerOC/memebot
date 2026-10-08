"""Time-aware training of local decision models.

Never randomly shuffle all trades. Train on earlier windows, validate/test later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .dataset import (
    examples_to_xy,
    generate_synthetic,
    load_jsonl,
    save_jsonl,
    time_aware_split,
)
from .models.logistic import LogisticModel
from .models.rules import RulesModel
from .models.stumps import StumpEnsembleModel
from .schema import FEATURE_SCHEMA_VERSION, schema_manifest
from .validate import EvalReport, evaluate


@dataclass
class TrainResult:
    model_kind: str
    model_path: str
    model_version: str
    n_train: int
    n_valid: int
    n_test: int
    valid: EvalReport
    test: EvalReport
    synthetic: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "model_path": self.model_path,
            "model_version": self.model_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
            "valid": self.valid.to_dict(),
            "test": self.test.to_dict(),
            "synthetic": self.synthetic,
            "notes": list(self.notes),
            "schema": schema_manifest()["target"],
        }


def train_models(
    *,
    data_path: Optional[str] = None,
    out_dir: str = "artifacts/decision",
    n_synthetic: int = 500,
    seed: int = 42,
    buy_threshold: float = 0.65,
) -> dict[str, TrainResult]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if data_path:
        examples = load_jsonl(data_path)
        synthetic = all(e.meta.get("synthetic") for e in examples) if examples else False
        notes = [f"loaded {len(examples)} examples from {data_path}"]
    else:
        examples = generate_synthetic(n_synthetic, seed=seed)
        synthetic = True
        data_out = out / "synthetic_train.jsonl"
        save_jsonl(data_out, examples)
        notes = [
            f"generated {len(examples)} SYNTHETIC examples → {data_out}",
            "SYNTHETIC DATA IS NOT PROFITABILITY EVIDENCE",
        ]

    train, valid, test = time_aware_split(examples)
    notes.append(
        f"time-aware split: train={len(train)} valid={len(valid)} test={len(test)} "
        f"(chronological; no shuffle)"
    )
    if not train or not test:
        raise ValueError("insufficient examples for time-aware split")

    Xtr, ytr, _ = examples_to_xy(train)
    Xva, yva, _ = examples_to_xy(valid) if valid else (Xtr[:0], ytr[:0], np.array([]))
    Xte, yte, _ = examples_to_xy(test)

    results: dict[str, TrainResult] = {}

    # A: rules baseline (no fit)
    rules = RulesModel(model_version="rules.v1")
    rules_path = out / "model_rules.v1.json"
    rules.save(rules_path)
    results["rules"] = TrainResult(
        model_kind="rules",
        model_path=str(rules_path),
        model_version=rules.model_version,
        n_train=len(train), n_valid=len(valid), n_test=len(test),
        valid=evaluate(rules, Xva, yva, threshold=buy_threshold) if len(yva) else evaluate(rules, Xte, yte, threshold=buy_threshold),
        test=evaluate(rules, Xte, yte, threshold=buy_threshold),
        synthetic=synthetic,
        notes=notes + ["rules model has no learned weights"],
    )

    # B: logistic
    logistic = LogisticModel.train(
        Xtr, ytr, model_version="logistic.v1", seed=seed,
    )
    log_path = out / "model_logistic.v1.json"
    logistic.save(log_path)
    results["logistic"] = TrainResult(
        model_kind="logistic",
        model_path=str(log_path),
        model_version=logistic.model_version,
        n_train=len(train), n_valid=len(valid), n_test=len(test),
        valid=evaluate(logistic, Xva, yva, threshold=buy_threshold) if len(yva) else evaluate(logistic, Xte, yte, threshold=buy_threshold),
        test=evaluate(logistic, Xte, yte, threshold=buy_threshold),
        synthetic=synthetic,
        notes=notes + ["selected candidate: L2 logistic regression"],
    )

    # C: stump ensemble
    stumps = StumpEnsembleModel.train(
        Xtr, ytr, model_version="stumps.v1", seed=seed,
    )
    stump_path = out / "model_stumps.v1.json"
    stumps.save(stump_path)
    results["stumps"] = TrainResult(
        model_kind="stumps",
        model_path=str(stump_path),
        model_version=stumps.model_version,
        n_train=len(train), n_valid=len(valid), n_test=len(test),
        valid=evaluate(stumps, Xva, yva, threshold=buy_threshold) if len(yva) else evaluate(stumps, Xte, yte, threshold=buy_threshold),
        test=evaluate(stumps, Xte, yte, threshold=buy_threshold),
        synthetic=synthetic,
        notes=notes + ["stump ensemble comparison candidate"],
    )

    summary_path = out / "train_summary.json"
    summary_path.write_text(
        json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2),
        encoding="utf-8",
    )
    notes.append(f"wrote {summary_path}")
    return results


def choose_winner(results: dict[str, TrainResult]) -> str:
    """Pick the simplest model with useful out-of-sample signal.

    A model is "useful" only if test precision beats the positive base rate by
    at least 5 percentage points *and* it issues at least one BUY (tp+fp>0).
    If nobody clears that bar, default to logistic (calibrated, fail-closed)
    when present — not because it is profitable, but because it is the safest
    local classifier to keep in the shadow path while real labels accumulate.
    """
    # Base rate from any arm's test set (same split for all).
    sample = next(iter(results.values()))
    cm0 = sample.test.confusion
    n_pos = cm0.tp + cm0.fn
    n = sample.test.n or 1
    base_rate = n_pos / n

    useful: list[tuple[tuple[float, float, int], str]] = []
    order = {"logistic": 0, "rules": 1, "stumps": 2}
    for name, tr in results.items():
        cm = tr.test.confusion
        buys = cm.tp + cm.fp
        if buys <= 0:
            continue
        if cm.precision < base_rate + 0.05:
            continue
        # Higher precision, then recall; prefer simpler on ties.
        key = (cm.precision, cm.recall, -order.get(name, 9))
        useful.append((key, name))
    if useful:
        useful.sort(reverse=True)
        return useful[0][1]
    if "logistic" in results:
        return "logistic"
    return next(iter(results))
