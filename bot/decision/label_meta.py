"""Build labeled decision examples from META observation JSONL + forward prices.

Required historical data for a real label:
  1. A decision-time market_snapshot (or signal) row with features
  2. A forward price path for that mint after detection_ts

Without (2), examples cannot be labeled — this module reports how many rows
were skipped for missing forward prices rather than inventing outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from ..meta.detector import score_snapshot
from ..meta.model import TokenSnapshot
from .dataset import LabeledExample, save_jsonl
from .features import extract_features
from .labels import PriceTick, label_path
from .schema import DEFAULT_DETECTION_LATENCY_SECONDS, TARGET_NAME


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def price_paths_from_snapshots(rows: Sequence[dict[str, Any]]) -> dict[str, list[PriceTick]]:
    """Build per-mint price paths from successive market_snapshot observations."""
    paths: dict[str, list[PriceTick]] = {}
    for row in rows:
        if row.get("kind") not in (None, "market_snapshot") and "price_usd" not in row:
            # Allow raw snapshot dicts without kind, or kind=market_snapshot
            if row.get("kind") not in ("market_snapshot",):
                continue
        mint = str(row.get("mint") or "")
        price = row.get("price_usd")
        ts = row.get("observed_at") or row.get("recorded_at")
        if not mint or price is None or ts is None:
            continue
        try:
            px = float(price)
            t = float(ts)
        except (TypeError, ValueError):
            continue
        if px <= 0:
            continue
        paths.setdefault(mint, []).append(PriceTick(t, px))
    for mint in paths:
        paths[mint].sort(key=lambda t: t.ts)
    return paths


def label_from_meta_observations(
    observations_path: str | Path,
    *,
    out_path: Optional[str | Path] = None,
    latency_seconds: float = DEFAULT_DETECTION_LATENCY_SECONDS,
    min_forward_ticks: int = 2,
) -> dict[str, Any]:
    """Create labeled JSONL from META observations when forward prices exist.

    Returns a summary; writes JSONL only when out_path is set and labels > 0.
    """
    rows = _load_jsonl(observations_path)
    market_rows = [
        r for r in rows
        if r.get("kind") == "market_snapshot" or (r.get("mint") and r.get("price_usd") is not None and "market" in r)
    ]
    paths = price_paths_from_snapshots(market_rows)
    examples: list[LabeledExample] = []
    skipped_no_path = 0
    skipped_bad = 0

    for row in market_rows:
        mint = str(row.get("mint") or "")
        ts = float(row.get("observed_at") or row.get("recorded_at") or 0.0)
        if not mint or ts <= 0:
            skipped_bad += 1
            continue
        path = [t for t in paths.get(mint, []) if t.ts >= ts]
        if len(path) < min_forward_ticks:
            skipped_no_path += 1
            continue
        try:
            snap = TokenSnapshot.from_dict(row)
        except Exception:
            skipped_bad += 1
            continue
        signal = score_snapshot(snap)
        feats = extract_features(snap, signal, in_active_meta=False, now=ts)
        outcome = label_path(path, ts, latency_seconds=latency_seconds)
        examples.append(
            LabeledExample(
                features=feats,
                label=outcome.label,
                detection_ts=ts,
                path=path,
                meta={
                    "synthetic": False,
                    "source": "meta_observations",
                    "target_name": TARGET_NAME,
                    "outcome": outcome.to_dict(),
                },
            )
        )

    if out_path and examples:
        save_jsonl(out_path, examples)

    return {
        "observations_path": str(observations_path),
        "market_rows": len(market_rows),
        "labeled": len(examples),
        "skipped_no_forward_path": skipped_no_path,
        "skipped_bad_row": skipped_bad,
        "out_path": str(out_path) if out_path and examples else None,
        "status": "COMPLETE" if examples else "BLOCKED",
        "blocker": None if examples else "No mint has enough forward price ticks to label",
        "profitability": "NO VERIFIED PROFITABILITY",
    }
