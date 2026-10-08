"""Compare decision arms on the SAME dataset.

A = rules only (existing-threshold heuristics)
B = local learned model (logistic / stumps)
C = frontier LLM — UNAVAILABLE unless an offline score file is supplied.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .benchmark import run_benchmark
from .dataset import generate_synthetic, load_jsonl, time_aware_split, walk_forward_folds
from .models.base import LocalModel
from .models.logistic import LogisticModel
from .models.rules import RulesModel
from .models.stumps import StumpEnsembleModel
from .shadow_sim import simulate_arm
from .train import choose_winner, train_models
from .validate import evaluate
from .dataset import examples_to_xy


def compare_arms(
    *,
    data_path: Optional[str] = None,
    out_dir: str = "artifacts/decision",
    n_synthetic: int = 500,
    seed: int = 42,
    buy_threshold: float = 0.65,
) -> dict[str, Any]:
    results = train_models(
        data_path=data_path, out_dir=out_dir, n_synthetic=n_synthetic, seed=seed,
    )
    winner = choose_winner(results)

    if data_path:
        examples = load_jsonl(data_path)
        synthetic = all(e.meta.get("synthetic") for e in examples) if examples else False
    else:
        examples = generate_synthetic(n_synthetic, seed=seed)
        synthetic = True
    _, _, test = time_aware_split(examples)

    arms: dict[str, Any] = {}
    for name, tr in results.items():
        model = LocalModel.load(tr.model_path)
        sim = simulate_arm(model, test, buy_threshold=buy_threshold)
        # Micro latency for this arm on test features
        t0 = time.perf_counter()
        for ex in test:
            model.predict_proba(ex.features.values)
        t1 = time.perf_counter()
        n = max(1, len(test))
        arms[name] = {
            "cost_usd_per_decision": 0.0,
            "latency_ms_per_decision_mean": ((t1 - t0) * 1000.0) / n,
            "throughput_decisions_per_sec": n / (t1 - t0) if (t1 - t0) > 0 else None,
            "test_classification": tr.test.to_dict(),
            "valid_classification": tr.valid.to_dict(),
            "shadow_simulation": sim.to_dict(),
            "model_path": tr.model_path,
            "synthetic": tr.synthetic,
        }

    # Walk-forward on winner kind (logistic) for stability check
    wf_rows = []
    folds = walk_forward_folds(examples, n_folds=3, min_train=40)
    for i, (tr_ex, va_ex, te_ex) in enumerate(folds):
        Xtr, ytr, _ = examples_to_xy(tr_ex)
        Xte, yte, _ = examples_to_xy(te_ex)
        if winner == "stumps":
            m = StumpEnsembleModel.train(Xtr, ytr, model_version=f"stumps.wf{i}")
        elif winner == "rules":
            m = RulesModel(model_version=f"rules.wf{i}")
        else:
            m = LogisticModel.train(Xtr, ytr, model_version=f"logistic.wf{i}", epochs=200)
        ev = evaluate(m, Xte, yte, threshold=buy_threshold)
        sim = simulate_arm(m, te_ex, buy_threshold=buy_threshold)
        wf_rows.append({
            "fold": i,
            "n_train": len(tr_ex),
            "n_valid": len(va_ex),
            "n_test": len(te_ex),
            "classification": ev.to_dict(),
            "shadow_simulation": sim.to_dict(),
        })

    bench = run_benchmark(
        LocalModel.load(results[winner].model_path) if winner in results else None,
        n_warmup=20, n_iter=500,
        out_path=str(Path(out_dir) / "latency_benchmark.json"),
    )

    report = {
        "arms": arms,
        "winner": winner,
        "winner_reason": (
            "Useful OOS precision above base rate + 5pp if any; else calibrated "
            "fail-closed logistic"
        ),
        "walk_forward": wf_rows,
        "latency_benchmark": {
            "feature_extraction_p50_ms": bench["feature_extraction"]["p50_ms"],
            "feature_extraction_p99_ms": bench["feature_extraction"]["p99_ms"],
            "model_inference_p50_ms": bench["model_inference"]["p50_ms"],
            "model_inference_p99_ms": bench["model_inference"]["p99_ms"],
            "end_to_end_p50_ms": bench["end_to_end"]["p50_ms"],
            "end_to_end_p99_ms": bench["end_to_end"]["p99_ms"],
            "throughput_decisions_per_sec": bench["throughput_decisions_per_sec"],
            "rss_mb_after": bench["rss_mb_after"],
        },
        "arm_C_frontier_llm": {
            "status": "UNAVAILABLE",
            "reason": (
                "No frontier LLM is wired into the decision path by design. "
                "Offline comparison requires a separately supplied score file; none present."
            ),
            "cost_usd_per_decision": None,
        },
        "profitability": "NO VERIFIED PROFITABILITY",
        "dataset_synthetic": synthetic,
        "notes": [
            "All arms evaluated on the SAME chronological test slice",
            "Shadow PnL uses latency-adjusted entry + fee/slippage; not wallet fills",
            "Walk-forward folds retrain without future leakage",
            "SYNTHETIC metrics are not trading evidence",
        ],
    }
    path = Path(out_dir) / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["comparison_path"] = str(path)
    return report
