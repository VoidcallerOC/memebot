"""Latency and throughput benchmarks for the local decision engine.

Reports p50/p90/p95/p99/max — never average alone.
"""
from __future__ import annotations

import json
import random
import resource
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import Config
from .dataset import _make_signal, _make_snapshot, generate_synthetic
from .engine import DecisionEngine
from .features import extract_features
from .models.base import LocalModel
from .models.logistic import LogisticModel
from .risk_firewall import RiskFirewall
from .schema import N_FEATURES


@dataclass
class LatencyStats:
    name: str
    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples_ms.append(float(ms))

    def summary(self) -> dict[str, Any]:
        xs = sorted(self.samples_ms)
        n = len(xs)
        if n == 0:
            return {"name": self.name, "n": 0}

        def pct(p: float) -> float:
            if n == 1:
                return xs[0]
            k = (n - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, n - 1)
            return xs[f] + (xs[c] - xs[f]) * (k - f)

        return {
            "name": self.name,
            "n": n,
            "p50_ms": pct(50),
            "p90_ms": pct(90),
            "p95_ms": pct(95),
            "p99_ms": pct(99),
            "max_ms": xs[-1],
            "min_ms": xs[0],
            "mean_ms": statistics.fmean(xs),
        }


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_benchmark(
    model: Optional[LocalModel] = None,
    *,
    n_warmup: int = 50,
    n_iter: int = 1000,
    out_path: Optional[str] = None,
) -> dict[str, Any]:
    examples = generate_synthetic(max(n_iter, 200), seed=99)
    if model is None:
        X = np.array([e.features.values for e in examples[:300]], dtype=np.float64)
        y = np.array([e.label for e in examples[:300]], dtype=np.float64)
        model = LogisticModel.train(X, y, model_version="logistic.v1.bench", epochs=100)

    cfg = Config(live_trading=False, bankroll_usd=20.0, max_position_pct=10.0)
    engine = DecisionEngine(model, RiskFirewall(cfg, kill_switch=False))

    extract_stats = LatencyStats("feature_extraction")
    infer_stats = LatencyStats("model_inference")
    e2e_stats = LatencyStats("end_to_end_decision")

    rng = random.Random(0)
    snaps = []
    for i in range(min(200, n_iter)):
        ts = 1_700_000_000.0 + i
        snap = _make_snapshot(
            f"m{i}", ts, liquidity=50_000, volume_5m=5_000, volume_1h=40_000,
            buys_5m=40, sells_5m=30, top_holder=20, price=0.001, rng=rng,
        )
        signal = _make_signal(snap, 0.5)
        snaps.append((snap, signal, ts))

    for i in range(n_warmup):
        ex = examples[i % len(examples)]
        engine.decide_from_features(ex.features, skip_network_screen=True, now=ex.detection_ts)
        snap, signal, ts = snaps[i % len(snaps)]
        extract_features(snap, signal, in_active_meta=True, now=ts)

    rss_before = _rss_mb()
    t_cpu0 = time.process_time()
    wall0 = time.perf_counter()

    for i in range(n_iter):
        snap, signal, ts = snaps[i % len(snaps)]
        t0 = time.perf_counter()
        feats = extract_features(snap, signal, in_active_meta=True, now=ts)
        t1 = time.perf_counter()
        extract_stats.add((t1 - t0) * 1000.0)

        t2 = time.perf_counter()
        model.predict_proba(feats.values)
        t3 = time.perf_counter()
        infer_stats.add((t3 - t2) * 1000.0)

        ex = examples[i % len(examples)]
        t4 = time.perf_counter()
        engine.decide_from_features(ex.features, skip_network_screen=True, now=ex.detection_ts)
        t5 = time.perf_counter()
        e2e_stats.add((t5 - t4) * 1000.0)

    wall1 = time.perf_counter()
    t_cpu1 = time.process_time()
    rss_after = _rss_mb()
    elapsed = wall1 - wall0
    throughput = n_iter / elapsed if elapsed > 0 else 0.0

    report = {
        "n_warmup": n_warmup,
        "n_iter": n_iter,
        "model_version": model.model_version,
        "n_features": N_FEATURES,
        "feature_extraction": extract_stats.summary(),
        "model_inference": infer_stats.summary(),
        "end_to_end": e2e_stats.summary(),
        "throughput_decisions_per_sec": throughput,
        "cpu_process_time_sec": t_cpu1 - t_cpu0,
        "wall_time_sec": elapsed,
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "rss_mb_delta": rss_after - rss_before,
        "notes": [
            "feature_extraction = TokenSnapshot+MetaSignal → FeatureRecord (local, no RPC)",
            "end_to_end = model inference + risk firewall (offline, no RPC/DexScreener)",
            "Network I/O for META collection is outside the local decision critical path",
            "Do not compare these numbers to proprietary vendor claims",
        ],
    }

    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
