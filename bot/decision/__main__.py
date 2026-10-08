"""CLI: python -m bot.decision <command>

Commands:
  schema       print feature/target schema
  train        time-aware train + evaluate candidates
  benchmark    latency p50/p90/p95/p99 + throughput
  compare      A/B(/C) comparison on same dataset (+ walk-forward + shadow PnL)
  label-meta   build labeled JSONL from META observations (needs forward prices)
  decide-demo  one offline decision on a synthetic sample (NO TRADE)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..config import Config
from .benchmark import run_benchmark
from .compare import compare_arms
from .dataset import generate_synthetic
from .engine import DecisionEngine
from .label_meta import label_from_meta_observations
from .models.base import LocalModel
from .risk_firewall import RiskFirewall
from .schema import schema_manifest
from .shadow import ShadowJournal, record_from_engine
from .train import choose_winner, train_models


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "schema"
    out_dir = "artifacts/decision"

    if cmd == "schema":
        print(json.dumps(schema_manifest(), indent=2))
        return 0

    if cmd == "train":
        data = None
        if "--data" in argv:
            data = argv[argv.index("--data") + 1]
        results = train_models(data_path=data, out_dir=out_dir)
        winner = choose_winner(results)
        summary = {k: v.to_dict() for k, v in results.items()}
        summary["winner"] = winner
        print(json.dumps(summary, indent=2))
        print(f"\nWINNER={winner} (synthetic={results[winner].synthetic})", file=sys.stderr)
        if results[winner].synthetic:
            print("NO VERIFIED PROFITABILITY — metrics are on synthetic labels", file=sys.stderr)
        return 0

    if cmd == "benchmark":
        model_path = None
        if "--model" in argv:
            model_path = argv[argv.index("--model") + 1]
        model = LocalModel.load(model_path) if model_path else None
        n_iter = 1000
        if "--n" in argv:
            n_iter = int(argv[argv.index("--n") + 1])
        report = run_benchmark(
            model, n_iter=n_iter, out_path=f"{out_dir}/latency_benchmark.json",
        )
        print(json.dumps(report, indent=2))
        return 0

    if cmd == "compare":
        data = None
        if "--data" in argv:
            data = argv[argv.index("--data") + 1]
        report = compare_arms(data_path=data, out_dir=out_dir)
        print(json.dumps(report, indent=2))
        return 0

    if cmd == "label-meta":
        obs = "meta_observations.jsonl"
        if "--observations" in argv:
            obs = argv[argv.index("--observations") + 1]
        out = f"{out_dir}/labeled_from_meta.jsonl"
        if "--out" in argv:
            out = argv[argv.index("--out") + 1]
        summary = label_from_meta_observations(obs, out_path=out)
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("labeled", 0) > 0 else 1

    if cmd == "decide-demo":
        # Offline demo only — writes a shadow/counterfactual row, never trades.
        examples = generate_synthetic(5, seed=1)
        model_path = Path(out_dir) / "model_logistic.v1.json"
        if not model_path.exists():
            train_models(out_dir=out_dir, n_synthetic=200)
        model = LocalModel.load(model_path)
        cfg = Config(live_trading=False, bankroll_usd=20.0)
        engine = DecisionEngine(model, RiskFirewall(cfg))
        ex = examples[0]
        result = engine.decide_from_features(
            ex.features, skip_network_screen=True, now=ex.detection_ts,
        )
        rec = record_from_engine(
            result, path=ex.path, wallet_entry_price=ex.path[0].price * 0.99,
        )
        journal = ShadowJournal(f"{out_dir}/shadow_demo.jsonl")
        journal.append(rec)
        print(json.dumps({"engine": result.to_dict(), "shadow": rec.to_dict()}, indent=2))
        print("NO LIVE TRADE — shadow journal only", file=sys.stderr)
        return 0

    print(
        "usage: python -m bot.decision "
        "[schema|train|benchmark|compare|label-meta|decide-demo]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
