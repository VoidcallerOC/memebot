"""Post-PR #15 data-integrity regressions.

Covers:
  1. Full META TokenSnapshot + MetaSignal into the shadow decision path
  2. Incomplete forward horizons are UNLABELED (never invented as label=0)
  3. Purge/embargo on time-aware and walk-forward splits
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot.config import Config
from bot.decision.bridge import DecisionShadowBridge, build_meta_context, snapshot_from_safety
from bot.decision.dataset import (
    LabeledExample,
    generate_synthetic,
    purge_overlapping_labels,
    time_aware_split,
    walk_forward_folds,
)
from bot.decision.engine import MAX_MISSING_FRAC_FOR_BUY
from bot.decision.features import extract_features
from bot.decision.label_meta import label_from_meta_observations
from bot.decision.labels import PriceTick, label_path
from bot.decision.models.logistic import LogisticModel
from bot.decision.schema import N_FEATURES, TARGET_HORIZON_SECONDS
from bot.decision.signal import BUY, REJECT
from bot.meta.detector import score_snapshot
from bot.meta.model import MarketWindow, TokenSnapshot
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.safety import TokenSafety


def _cfg(**kwargs) -> Config:
    base = dict(
        live_trading=False,
        bankroll_usd=20.0,
        max_position_pct=10.0,
        max_open_positions=3,
        min_liquidity_usd=1000.0,
        decision_shadow_enabled=True,
        decision_model_path="artifacts/decision/model_logistic.v1.json",
        decision_shadow_file="decision_shadow_test.jsonl",
    )
    base.update(kwargs)
    return Config(**base)


def _rich_snap(mint: str, *, now: float = 1_700_000_000.0) -> TokenSnapshot:
    return TokenSnapshot(
        mint=mint,
        symbol="META",
        observed_at=now,
        price_usd=0.02,
        liquidity_usd=80_000.0,
        market_cap_usd=400_000.0,
        pair_created_at=now - 10_800.0,
        market={
            "5m": MarketWindow(volume_usd=8_000, tx_buys=55, tx_sells=30, price_change_pct=4.0),
            "1h": MarketWindow(volume_usd=55_000, tx_buys=320, tx_sells=210, price_change_pct=12.0),
        },
        top_holder_pct=11.0,
        top10_holder_pct=38.0,
        creator_pct=4.0,
        source="test_meta",
    )


# ---------------------------------------------------------------------------
# 1. Full META → decision integration
# ---------------------------------------------------------------------------

def test_bridge_uses_meta_snapshot_and_signal_feature_schema(tmp_path: Path):
    model_path = tmp_path / "m.json"
    LogisticModel([0.0] * N_FEATURES, bias=10.0, model_version="force.buy").save(model_path)
    cfg = _cfg(
        decision_model_path=str(model_path),
        decision_shadow_file=str(tmp_path / "shadow.jsonl"),
    )
    bridge = DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio())
    mint = "MetaMint111111111111111111111111111111111"
    safety = TokenSafety(mint=mint, passed=True, liquidity_usd=80_000, price_usd=0.02, symbol="META")
    snap = _rich_snap(mint)
    signal = score_snapshot(snap)
    # Training-schema features from META
    trained = extract_features(snap, signal, now=snap.observed_at)

    result = bridge.evaluate_candidate(
        safety, now=snap.observed_at, snapshot=snap, signal=signal,
    )
    assert result.decision.action == BUY
    assert result.features.missing_frac < MAX_MISSING_FRAC_FOR_BUY
    # Same feature schema the model was trained against (values aligned).
    assert len(result.features.values) == N_FEATURES == len(trained.values)
    assert result.features.values == pytest.approx(trained.values)
    rec = json.loads(Path(cfg.decision_shadow_file).read_text(encoding="utf-8").strip())
    assert "meta_source=provided" in rec["notes"]


def test_sparse_safety_fallback_does_not_relax_missing_threshold(tmp_path: Path, monkeypatch):
    """Safety-only sparse snapshots must not BUY via relax_missing."""
    model_path = tmp_path / "m.json"
    LogisticModel([0.0] * N_FEATURES, bias=10.0, model_version="force.buy").save(model_path)
    cfg = _cfg(
        decision_model_path=str(model_path),
        decision_shadow_file=str(tmp_path / "s.jsonl"),
    )
    bridge = DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio())

    # Force META fetch to fail → safety_fallback path.
    monkeypatch.setattr(
        "bot.decision.bridge.snapshot_from_dexscreener",
        lambda *a, **k: None,
    )
    safety = TokenSafety(
        mint="SparseMint11111111111111111111111111111",
        passed=True, liquidity_usd=50_000, price_usd=0.01, symbol="SP",
    )
    result = bridge.evaluate_candidate(safety, now=1_700_000_000.0)
    assert result.decision.action == REJECT
    assert "MISSING_FEATURES_ABOVE_THRESHOLD" in result.decision.reasons
    assert "META_SNAPSHOT_UNAVAILABLE" in result.decision.reasons
    assert result.features.missing_frac > MAX_MISSING_FRAC_FOR_BUY
    # Risk firewall still present / evaluateable
    assert result.risk is not None


def test_build_meta_context_prefers_provided_snapshot():
    safety = TokenSafety(mint="m", passed=True, liquidity_usd=1.0, price_usd=1.0, symbol="X")
    snap = _rich_snap("m")
    out_snap, out_sig, source = build_meta_context(safety, snapshot=snap, now=snap.observed_at)
    assert source == "provided"
    assert out_sig is not None
    assert out_snap.market.get("5m") is not None
    sparse = snapshot_from_safety(safety, now=1.0)
    assert not sparse.market


# ---------------------------------------------------------------------------
# 2. Label completeness — incomplete horizon = UNLABELED
# ---------------------------------------------------------------------------

def test_incomplete_horizon_is_unlabeled_not_zero():
    """Stream ends before full horizon without TP/SL → UNLABELED, not label=0."""
    path = [
        PriceTick(0.0, 1.0),
        PriceTick(2.5, 1.0),
        PriceTick(60.0, 1.01),
        PriceTick(120.0, 1.02),  # far short of 3600s horizon, no TP/SL
    ]
    outcome = label_path(path, detection_ts=0.0, latency_seconds=2.5, horizon_seconds=3600.0)
    assert outcome.label is None
    assert outcome.exit_reason == "incomplete"
    assert outcome.is_labeled is False


def test_true_timeout_with_full_coverage_is_label_zero():
    """Full horizon observed, neither TP nor stop → legitimate label=0."""
    path = [PriceTick(0.0, 1.0)]
    # Ticks every 60s through past the deadline (entry ~2.5 + 3600).
    for i in range(1, 70):
        path.append(PriceTick(float(i * 60), 1.0 + 0.001 * (i % 3)))  # stays flat-ish
    outcome = label_path(path, detection_ts=0.0, latency_seconds=2.5, horizon_seconds=3600.0)
    assert outcome.label == 0
    assert outcome.exit_reason == "timeout"
    assert outcome.is_labeled is True


def test_stop_before_horizon_still_labeled_zero():
    path = [
        PriceTick(0.0, 1.0),
        PriceTick(2.5, 1.0),
        PriceTick(5.0, 0.94),  # -6% stop
    ]
    outcome = label_path(path, detection_ts=0.0, latency_seconds=2.5)
    assert outcome.label == 0
    assert outcome.exit_reason == "stop"


def test_no_path_is_unlabeled():
    outcome = label_path([], detection_ts=0.0)
    assert outcome.label is None
    assert outcome.exit_reason == "no_path"


def test_label_meta_skips_incomplete_horizons(tmp_path: Path):
    """Observation stream shorter than horizon without TP/SL → not written as 0."""
    obs = tmp_path / "obs.jsonl"
    mint = "MintIncomplete222222222222222222222222222"
    rows = []
    # Only ~5 minutes of flat prices — no TP/SL, incomplete vs 1h horizon.
    for i in range(6):
        rows.append({
            "kind": "market_snapshot",
            "mint": mint,
            "symbol": "INC",
            "observed_at": 1_700_000_000.0 + i * 60,
            "price_usd": 1.0,
            "liquidity_usd": 50_000,
            "market_cap_usd": 200_000,
            "market": {
                "5m": {"volume_usd": 1000, "tx_buys": 10, "tx_sells": 10},
                "1h": {"volume_usd": 5000, "tx_buys": 50, "tx_sells": 50},
            },
            "top_holder_pct": 12,
        })
    obs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "labeled.jsonl"
    summary = label_from_meta_observations(obs, out_path=out)
    assert summary["labeled"] == 0
    assert summary["skipped_incomplete_horizon"] >= 1
    assert not out.exists()
    assert summary["profitability"] == "NO VERIFIED PROFITABILITY"


# ---------------------------------------------------------------------------
# 3. Time-series leakage — purge / embargo
# ---------------------------------------------------------------------------

def _toy_examples(n: int, spacing: float = 300.0, start: float = 1_000_000.0) -> list[LabeledExample]:
    examples = generate_synthetic(n, seed=7, start_ts=start)
    # Re-stamp detection_ts to exact spacing for deterministic purge math.
    for i, e in enumerate(examples):
        e.detection_ts = start + i * spacing
    return examples


def test_purge_removes_train_labels_overlapping_valid():
    horizon = 3600.0
    embargo = 3600.0
    feats = generate_synthetic(3, seed=1)
    # right_start=1e6+10_000; cutoff=1e6+10_000-3600=1e6+6400
    # keep if detection_ts + 3600 <= 1e6+6400 → detection_ts <= 1e6+2800
    left = [
        LabeledExample(
            features=feats[0].features, label=1,
            detection_ts=1_000_000.0,  # safe: ends at 1e6+3600 < cutoff
            path=[], meta={},
        ),
        LabeledExample(
            features=feats[1].features, label=0,
            detection_ts=1_000_000.0 + 5000.0,  # leaks: ends at 1e6+8600 > cutoff
            path=[], meta={},
        ),
    ]
    right = [
        LabeledExample(
            features=feats[2].features, label=1,
            detection_ts=1_000_000.0 + 10_000.0,
            path=[], meta={},
        ),
    ]
    purged = purge_overlapping_labels(
        left, right, horizon_seconds=horizon, embargo_seconds=embargo,
    )
    assert len(purged) == 1
    assert purged[0].detection_ts == 1_000_000.0
    assert all(
        e.detection_ts + horizon <= min(r.detection_ts for r in right) - embargo
        for e in purged
    )


def test_time_aware_split_applies_purge_embargo():
    # Dense timeline so boundary overlap is unavoidable without purge.
    examples = _toy_examples(80, spacing=300.0)
    train, valid, test = time_aware_split(
        examples,
        purge_horizon_seconds=TARGET_HORIZON_SECONDS,
        embargo_seconds=TARGET_HORIZON_SECONDS,
    )
    assert train and test
    # Chronological
    if valid:
        assert max(e.detection_ts for e in train) < min(e.detection_ts for e in valid)
        assert max(e.detection_ts for e in valid) < min(e.detection_ts for e in test)
        # No train label window overlaps valid (with embargo)
        v0 = min(e.detection_ts for e in valid)
        for e in train:
            assert e.detection_ts + TARGET_HORIZON_SECONDS <= v0 - TARGET_HORIZON_SECONDS
    # Never shuffled: detection_ts non-decreasing inside each split
    for part in (train, valid, test):
        ts = [e.detection_ts for e in part]
        assert ts == sorted(ts)


def test_walk_forward_folds_purge_embargo():
    examples = _toy_examples(150, spacing=300.0)
    folds = walk_forward_folds(
        examples,
        n_folds=3,
        min_train=40,
        purge_horizon_seconds=TARGET_HORIZON_SECONDS,
        embargo_seconds=TARGET_HORIZON_SECONDS,
    )
    assert len(folds) >= 1
    for train, valid, test in folds:
        assert train and test
        t_test = min(e.detection_ts for e in test)
        for e in train:
            assert e.detection_ts + TARGET_HORIZON_SECONDS <= t_test - TARGET_HORIZON_SECONDS
        if valid:
            assert max(e.detection_ts for e in train) < min(e.detection_ts for e in valid)
            for e in valid:
                assert e.detection_ts + TARGET_HORIZON_SECONDS <= t_test - TARGET_HORIZON_SECONDS


def test_safety_gates_unchanged_live_trading_false_ceiling():
    cfg = _cfg(live_trading=False, bankroll_usd=20.0)
    assert cfg.live_trading is False
    assert cfg.is_armed is False
    assert cfg.bankroll_usd == 20.0
