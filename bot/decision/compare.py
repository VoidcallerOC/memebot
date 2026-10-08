"""Compare decision arms on the SAME dataset.

A = rules only (existing-threshold heuristics)
B = rules features + local learned model (logistic / stumps)
C = frontier LLM — NOT implemented here (no remote API in critical path);
    reported as UNAVAILABLE unless an offline score file is supplied.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .dataset import examples_to_xy, generate_synthetic, load_jsonl, time_aware_split
from .models.base import LocalModel
from .train import choose_winner, train_models
from .validate import evaluate


def compare_arms(
    *,
    data_path: Optional[str] = None,
    out_dir: str = "artifacts/decision",
    n_synthetic: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    results = train_models(
        data_path=data_path, out_dir=out_dir, n_synthetic=n_synthetic, seed=seed,
    )
    winner = choose_winner(results)

    arms = {}
    for name, tr in results.items():
        arms[name] = {
            "latency_note": "see benchmark.py — model inference is local CPU",
            "cost_usd_per_decision": 0.0,
            "test": tr.test.to_dict(),
            "valid": tr.valid.to_dict(),
            "model_path": tr.model_path,
            "synthetic": tr.synthetic,
        }

    report = {
        "arms": arms,
        "winner": winner,
        "winner_reason": (
            "Highest out-of-sample precision with preference for simpler logistic "
            "when within 0.02 of stumps"
        ),
        "arm_C_frontier_llm": {
            "status": "UNAVAILABLE",
            "reason": (
                "No frontier LLM is wired into the decision path by design. "
                "Offline comparison requires a separately supplied score file; none present."
            ),
            "cost_usd_per_decision": None,
        },
        "profitability": "NO VERIFIED PROFITABILITY",
        "notes": [
            "All metrics on synthetic or user-supplied labeled JSONL",
            "Shadow expectancy with real chain data is NOT STARTED until observations accumulate",
            results[winner].notes[0] if winner in results else "",
        ],
    }
    path = Path(out_dir) / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["comparison_path"] = str(path)
    return report
