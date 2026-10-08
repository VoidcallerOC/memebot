"""Labeled dataset IO + synthetic generator for pipeline/benchmark testing.

Synthetic data is ONLY for verifying the engine, latency, and train loop.
It does NOT constitute profitability evidence.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from ..meta.model import (
    MarketWindow,
    MetaSignal,
    TokenSnapshot,
    ok,
    unverified,
)
from .features import FeatureRecord, extract_features
from .labels import PriceTick, label_path
from .schema import (
    DEFAULT_DETECTION_LATENCY_SECONDS,
    FEATURE_SCHEMA_VERSION,
    N_FEATURES,
    TARGET_NAME,
)


@dataclass
class LabeledExample:
    features: FeatureRecord
    label: int
    detection_ts: float
    path: list[PriceTick]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "target_name": TARGET_NAME,
            "label": self.label,
            "detection_ts": self.detection_ts,
            "features": self.features.to_dict(),
            "price_path": [{"ts": t.ts, "price": t.price} for t in self.path],
            "meta": dict(self.meta),
            "synthetic": bool(self.meta.get("synthetic", False)),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LabeledExample":
        path = [PriceTick(float(p["ts"]), float(p["price"])) for p in raw.get("price_path") or []]
        return cls(
            features=FeatureRecord.from_dict(raw["features"]),
            label=int(raw["label"]),
            detection_ts=float(raw.get("detection_ts") or 0.0),
            path=path,
            meta=dict(raw.get("meta") or {}),
        )


def save_jsonl(path: str | Path, rows: Sequence[LabeledExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict()) + "\n")


def load_jsonl(path: str | Path) -> list[LabeledExample]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(LabeledExample.from_dict(json.loads(line)))
    return out


def examples_to_xy(examples: Sequence[LabeledExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n, F), y (n,), timestamps (n,) sorted by detection_ts."""
    ordered = sorted(examples, key=lambda e: e.detection_ts)
    X = np.array([e.features.values for e in ordered], dtype=np.float64)
    y = np.array([e.label for e in ordered], dtype=np.float64)
    ts = np.array([e.detection_ts for e in ordered], dtype=np.float64)
    return X, y, ts


def time_aware_split(
    examples: Sequence[LabeledExample],
    *,
    train_frac: float = 0.6,
    valid_frac: float = 0.2,
) -> tuple[list[LabeledExample], list[LabeledExample], list[LabeledExample]]:
    """Chronological split. Never shuffle across time."""
    ordered = sorted(examples, key=lambda e: e.detection_ts)
    n = len(ordered)
    if n == 0:
        return [], [], []
    i_train = max(1, int(n * train_frac))
    i_valid = max(i_train + 1, int(n * (train_frac + valid_frac)))
    i_valid = min(i_valid, n - 1) if n > 2 else i_train
    train = ordered[:i_train]
    valid = ordered[i_train:i_valid]
    test = ordered[i_valid:]
    return train, valid, test


def _make_snapshot(
    mint: str,
    ts: float,
    *,
    liquidity: float,
    volume_5m: float,
    volume_1h: float,
    buys_5m: int,
    sells_5m: int,
    top_holder: float,
    price: float,
    rng: random.Random,
) -> TokenSnapshot:
    age_hours = rng.uniform(1.0, 72.0)
    return TokenSnapshot(
        mint=mint,
        symbol=mint[:4].upper(),
        observed_at=ts,
        price_usd=price,
        liquidity_usd=liquidity,
        market_cap_usd=liquidity * rng.uniform(2.0, 20.0),
        pair_created_at=ts - age_hours * 3600.0,
        market={
            "5m": MarketWindow(
                volume_usd=volume_5m, tx_buys=buys_5m, tx_sells=sells_5m,
                price_change_pct=rng.uniform(-20, 40), source="synthetic",
            ),
            "1h": MarketWindow(
                volume_usd=volume_1h,
                tx_buys=buys_5m * 8, tx_sells=sells_5m * 8,
                price_change_pct=rng.uniform(-30, 60), source="synthetic",
            ),
        },
        top_holder_pct=top_holder,
        top10_holder_pct=min(100.0, top_holder * 2.5),
        creator_pct=rng.uniform(0, 15),
        liquidity_removed=False,
        lp_change_pct=rng.uniform(-5, 5),
        source="synthetic",
    )


def _make_signal(snap: TokenSnapshot, quality: float) -> MetaSignal:
    """quality in [0,1] drives META-like scores."""
    def sc(center: float):
        return ok(max(0.0, min(100.0, center)), "synthetic")

    return MetaSignal(
        token=snap.mint,
        symbol=snap.symbol,
        observed_at=snap.observed_at,
        trading_velocity=sc(40 + 50 * quality),
        historical_trading_velocity=sc(40 + 40 * quality),
        liquidity_quality=sc(30 + 60 * quality),
        manipulation_risk=sc(70 - 50 * quality),
        wallet_velocity=sc(35 + 45 * quality) if quality > 0.3 else unverified("synth"),
        attention_velocity=sc(40 + 40 * quality),
        flow_alignment="ALIGNED" if quality > 0.55 else "UNVERIFIED",
        provider_freshness="FRESH",
        price_usd=snap.price_usd,
        liquidity_usd=snap.liquidity_usd,
        market_cap_usd=snap.market_cap_usd,
    )


def _price_path(
    start_ts: float,
    entry_price: float,
    *,
    will_hit_tp: bool,
    rng: random.Random,
    n_ticks: int = 60,
    step_seconds: float = 60.0,
) -> list[PriceTick]:
    """Generate a forward path that either hits +10% before -5% or the reverse."""
    ticks = [PriceTick(start_ts, entry_price)]
    price = entry_price
    # Prepend a couple ticks before detection for realism
    for i in range(1, n_ticks + 1):
        ts = start_ts + i * step_seconds
        if will_hit_tp:
            # drift up with noise; avoid stopping out
            shock = rng.uniform(-0.01, 0.035)
        else:
            shock = rng.uniform(-0.035, 0.01)
        price = max(1e-12, price * (1.0 + shock))
        ticks.append(PriceTick(ts, price))
        if will_hit_tp and price >= entry_price * 1.10:
            break
        if not will_hit_tp and price <= entry_price * 0.95:
            break
    return ticks


def generate_synthetic(
    n: int = 400,
    *,
    seed: int = 42,
    start_ts: float = 1_700_000_000.0,
    latency_seconds: float = DEFAULT_DETECTION_LATENCY_SECONDS,
) -> list[LabeledExample]:
    """Synthetic labeled examples with a planted linear signal for smoke tests.

    POSITIVE class correlates with: higher liquidity quality / accel / buy pressure,
    lower manipulation / holder concentration. This is NOT real market data.
    """
    rng = random.Random(seed)
    examples: list[LabeledExample] = []
    for i in range(n):
        ts = start_ts + i * 300.0  # 5-minute spaced opportunities
        quality = rng.random()
        # Label probability rises with quality (planted signal).
        p_pos = 0.15 + 0.7 * quality
        will_hit = rng.random() < p_pos
        mint = f"SynthMint{i:05d}{'x' * 20}"[:44]
        liq = 10_000 * math.exp(quality * 2.5) * rng.uniform(0.8, 1.2)
        vol_1h = liq * rng.uniform(0.2, 2.0)
        vol_5m = vol_1h * (0.05 + 0.3 * quality) * rng.uniform(0.5, 1.5)
        buys = int(20 + 200 * quality)
        sells = int(20 + 200 * (1.0 - quality))
        top = 10 + 40 * (1.0 - quality)
        price = 10 ** rng.uniform(-6, -2)
        snap = _make_snapshot(
            mint, ts, liquidity=liq, volume_5m=vol_5m, volume_1h=vol_1h,
            buys_5m=buys, sells_5m=sells, top_holder=top, price=price, rng=rng,
        )
        signal = _make_signal(snap, quality)
        feats = extract_features(snap, signal, in_active_meta=quality > 0.6, now=ts)
        # Price path starts at detection; label_path applies latency.
        path = _price_path(ts, price, will_hit_tp=will_hit, rng=rng)
        outcome = label_path(path, ts, latency_seconds=latency_seconds)
        examples.append(
            LabeledExample(
                features=feats,
                label=outcome.label,
                detection_ts=ts,
                path=path,
                meta={
                    "synthetic": True,
                    "planted_quality": quality,
                    "planted_will_hit": will_hit,
                    "outcome": outcome.to_dict(),
                },
            )
        )
    return examples
