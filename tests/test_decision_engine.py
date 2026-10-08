"""Local decision engine tests — no network, no live trading, no keys."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from bot.config import Config
from bot.decision.dataset import (
    generate_synthetic,
    save_jsonl,
    time_aware_split,
    examples_to_xy,
)
from bot.decision.engine import DecisionEngine, MAX_MISSING_FRAC_FOR_BUY
from bot.decision.features import FeatureRecord, extract_features
from bot.decision.labels import PriceTick, label_path
from bot.decision.models.logistic import LogisticModel
from bot.decision.models.rules import RulesModel
from bot.decision.models.stumps import StumpEnsembleModel
from bot.decision.models.base import LocalModel
from bot.decision.risk_firewall import BLOCK, ALLOW, RiskFirewall
from bot.decision.schema import FEATURE_SCHEMA_VERSION, N_FEATURES, empty_vector
from bot.decision.shadow import ShadowJournal, record_from_engine
from bot.decision.signal import BUY, REJECT, WATCH
from bot.decision.train import choose_winner, train_models
from bot.decision.benchmark import run_benchmark
from bot.decision.validate import evaluate
from bot.meta.model import MarketWindow, TokenSnapshot, ok
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
    )
    base.update(kwargs)
    return Config(**base)


def _snap(**kwargs) -> TokenSnapshot:
    raw = dict(
        mint="So11111111111111111111111111111111111111112",
        symbol="TEST",
        observed_at=1_700_000_000.0,
        price_usd=0.01,
        liquidity_usd=50_000.0,
        market_cap_usd=200_000.0,
        pair_created_at=1_700_000_000.0 - 3600,
        market={
            "5m": MarketWindow(volume_usd=5000, tx_buys=40, tx_sells=20, price_change_pct=5),
            "1h": MarketWindow(volume_usd=40000, tx_buys=300, tx_sells=200, price_change_pct=10),
        },
        top_holder_pct=15.0,
        top10_holder_pct=40.0,
        creator_pct=5.0,
    )
    raw.update(kwargs)
    return TokenSnapshot(**raw)


def test_feature_vector_length_and_missing_behavior():
    snap = _snap(liquidity_usd=None, top_holder_pct=None)
    signal = None
    rec = extract_features(snap, signal, now=snap.observed_at)
    assert len(rec.values) == N_FEATURES
    assert rec.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert rec.missing_frac > 0
    # UNVERIFIED scores encoded as -1
    from bot.decision.schema import FEATURE_INDEX
    assert rec.values[FEATURE_INDEX["vol_accel_live"]] == -1.0
    assert rec.values[FEATURE_INDEX["top_holder_pct"]] == -1.0


def test_label_uses_latency_adjusted_entry_not_wallet_price():
    # Wallet supposedly entered at t=0 price=1.0; we detect at t=0 but latency=3s
    # At t=3 price has already moved to 1.20 — our entry must be 1.20.
    path = [
        PriceTick(0.0, 1.0),
        PriceTick(3.0, 1.20),
        PriceTick(10.0, 1.35),  # +12.5% from 1.20 → hits +10%
    ]
    outcome = label_path(path, detection_ts=0.0, latency_seconds=3.0)
    assert outcome.entry_price == pytest.approx(1.20)
    assert outcome.label == 1
    assert outcome.exit_reason == "take_profit"


def test_label_stop_before_tp():
    path = [
        PriceTick(0.0, 1.0),
        PriceTick(2.5, 1.0),
        PriceTick(5.0, 0.94),  # -6%
    ]
    outcome = label_path(path, detection_ts=0.0, latency_seconds=2.5)
    assert outcome.label == 0
    assert outcome.exit_reason == "stop"


def test_time_aware_split_is_chronological():
    examples = generate_synthetic(50, seed=1)
    train, valid, test = time_aware_split(examples)
    assert train and test
    assert max(e.detection_ts for e in train) <= min(e.detection_ts for e in (valid or test))
    if valid and test:
        assert max(e.detection_ts for e in valid) <= min(e.detection_ts for e in test)


def test_logistic_trains_and_predicts():
    examples = generate_synthetic(120, seed=2)
    train, _, test = time_aware_split(examples)
    X, y, _ = examples_to_xy(train)
    model = LogisticModel.train(X, y, epochs=150, model_version="logistic.test")
    pred = model.predict_proba(test[0].features.values)
    assert 0.0 <= pred.probability <= 1.0
    assert pred.model_version == "logistic.test"


def test_risk_firewall_blocks_model_buy():
    cfg = _cfg()
    risk = RiskManager(cfg)
    # Force daily halt
    risk._halted_today = True
    fw = RiskFirewall(cfg, risk=risk, portfolio=Portfolio())
    model = RulesModel()
    engine = DecisionEngine(model, fw, buy_threshold=0.0)  # force BUY if model says anything
    # Craft features that rules will score high
    examples = generate_synthetic(5, seed=3)
    # Use a model that always returns high p
    class AlwaysBuy(RulesModel):
        def predict_proba(self, features):
            from bot.decision.models.base import ModelPrediction
            return ModelPrediction(0.99, 99.0, "always", ["FORCE"])

    engine.model = AlwaysBuy()
    result = engine.decide_from_features(
        examples[0].features, skip_network_screen=True, now=examples[0].detection_ts,
    )
    assert result.decision.action == BUY
    assert result.risk.status == BLOCK
    assert any("HALT" in r or "MAX_OPEN" in r for r in result.risk.reasons)


def test_risk_firewall_blocks_safety_fail():
    cfg = _cfg()
    fw = RiskFirewall(cfg)
    class AlwaysBuy(LogisticModel):
        def __init__(self):
            super().__init__([0.0] * N_FEATURES, 10.0, model_version="ab")
    engine = DecisionEngine(AlwaysBuy(), fw, buy_threshold=0.5)
    safety = TokenSafety(mint="x", passed=False, reasons=["honeypot"], liquidity_usd=1e6, price_usd=1.0)
    examples = generate_synthetic(3, seed=4)
    # Make features fresh
    feats = examples[0].features
    feats.observed_at = examples[0].detection_ts
    result = engine.decide_from_features(
        feats, safety=safety, skip_network_screen=True, now=feats.observed_at,
    )
    assert result.decision.action == BUY
    assert result.risk.status == BLOCK
    assert any("SAFETY" in r for r in result.risk.reasons)


def test_model_unavailable_defaults_to_reject():
    cfg = _cfg()
    engine = DecisionEngine(None, RiskFirewall(cfg))
    examples = generate_synthetic(2, seed=5)
    result = engine.decide_from_features(
        examples[0].features, skip_network_screen=True, now=examples[0].detection_ts,
    )
    assert result.decision.action == REJECT
    assert "MODEL_UNAVAILABLE" in result.decision.reasons


def test_nan_features_reject():
    cfg = _cfg()
    model = RulesModel()
    engine = DecisionEngine(model, RiskFirewall(cfg))
    examples = generate_synthetic(2, seed=6)
    feats = examples[0].features
    feats.values[0] = float("nan")
    result = engine.decide_from_features(
        feats, skip_network_screen=True, now=feats.observed_at,
    )
    assert result.decision.action == REJECT
    assert "NAN_OR_INF_FEATURES" in result.decision.reasons


def test_stale_features_reject():
    cfg = _cfg()
    engine = DecisionEngine(RulesModel(), RiskFirewall(cfg))
    examples = generate_synthetic(2, seed=7)
    feats = examples[0].features
    feats.observed_at = 1_000_000.0
    result = engine.decide_from_features(feats, skip_network_screen=True, now=1_000_000.0 + 999)
    assert result.decision.action == REJECT
    assert any(r.startswith("STALE_FEATURES") for r in result.decision.reasons)


def test_unknown_token_reject():
    cfg = _cfg()
    engine = DecisionEngine(RulesModel(), RiskFirewall(cfg))
    values = empty_vector()
    feats = FeatureRecord(
        mint="", observed_at=1_700_000_000.0, values=values,
        missing_mask=[False] * N_FEATURES, missing_frac=0.0,
    )
    result = engine.decide_from_features(feats, skip_network_screen=True, now=feats.observed_at)
    assert result.decision.action == REJECT
    assert "UNKNOWN_TOKEN" in result.decision.reasons


def test_kill_switch_blocks():
    cfg = _cfg()
    fw = RiskFirewall(cfg, kill_switch=True)
    class AlwaysBuy(LogisticModel):
        def __init__(self):
            super().__init__([0.0] * N_FEATURES, 10.0, model_version="ab")
    engine = DecisionEngine(AlwaysBuy(), fw, buy_threshold=0.1)
    examples = generate_synthetic(2, seed=8)
    result = engine.decide_from_features(
        examples[0].features, skip_network_screen=True, now=examples[0].detection_ts,
    )
    assert result.risk.status == BLOCK
    assert "KILL_SWITCH" in result.risk.reasons


def test_shadow_ignores_wallet_entry_price(tmp_path: Path):
    cfg = _cfg()
    examples = generate_synthetic(5, seed=9)
    X, y, _ = examples_to_xy(examples[:40] if len(examples) >= 40 else examples)
    # small train
    examples = generate_synthetic(80, seed=9)
    train, _, _ = time_aware_split(examples)
    X, y, _ = examples_to_xy(train)
    model = LogisticModel.train(X, y, epochs=50)
    engine = DecisionEngine(model, RiskFirewall(cfg))
    ex = examples[-1]
    result = engine.decide_from_features(ex.features, skip_network_screen=True, now=ex.detection_ts)
    wallet_px = 999.0  # absurd — must not be used
    rec = record_from_engine(result, path=ex.path, wallet_entry_price=wallet_px)
    assert rec.simulated_entry_price != wallet_px
    assert any("IGNORED_FOR_SIMULATED_FILL" in n for n in rec.notes)
    journal = ShadowJournal(str(tmp_path / "shadow.jsonl"))
    journal.append(rec)
    rows = journal.load()
    assert len(rows) == 1
    assert "model_version" in rows[0]
    assert "feature_schema_version" in rows[0]
    assert "decision_engine_version" in rows[0]
    assert "risk_engine_version" in rows[0]


def test_counterfactual_kind_for_rejects(tmp_path: Path):
    cfg = _cfg()
    engine = DecisionEngine(None, RiskFirewall(cfg))  # always reject
    examples = generate_synthetic(3, seed=10)
    result = engine.decide_from_features(
        examples[0].features, skip_network_screen=True, now=examples[0].detection_ts,
    )
    rec = record_from_engine(result, path=examples[0].path)
    assert rec.kind == "counterfactual"
    assert rec.outcome_label in (0, 1)
    assert rec.rejection_reason


def test_model_save_load_roundtrip(tmp_path: Path):
    examples = generate_synthetic(60, seed=11)
    X, y, _ = examples_to_xy(examples)
    model = LogisticModel.train(X, y, epochs=40, model_version="logistic.roundtrip")
    path = tmp_path / "m.json"
    model.save(path)
    loaded = LocalModel.load(path)
    assert loaded.model_version == "logistic.roundtrip"
    p1 = model.predict_proba(X[0]).probability
    p2 = loaded.predict_proba(X[0]).probability
    assert p1 == pytest.approx(p2)


def test_train_compare_smoke(tmp_path: Path):
    out = tmp_path / "art"
    results = train_models(out_dir=str(out), n_synthetic=150, seed=12)
    assert "logistic" in results and "rules" in results and "stumps" in results
    winner = choose_winner(results)
    assert winner in results
    assert results[winner].synthetic is True
    # Explicit non-claim
    assert results[winner].test.n > 0


def test_benchmark_reports_percentiles(tmp_path: Path):
    report = run_benchmark(n_warmup=5, n_iter=50, out_path=str(tmp_path / "lat.json"))
    for key in ("feature_extraction", "model_inference", "end_to_end"):
        s = report[key]
        assert "p50_ms" in s and "p90_ms" in s and "p95_ms" in s and "p99_ms" in s
        assert "max_ms" in s
        assert s["p50_ms"] <= s["p90_ms"] <= s["p95_ms"] <= s["p99_ms"] <= s["max_ms"]
    assert report["throughput_decisions_per_sec"] > 0


def test_corrupted_model_weights_reject_or_safe():
    # Wrong weight length should fail at construction
    with pytest.raises(ValueError):
        LogisticModel(weights=[0.0, 1.0], bias=0.0)


def test_evaluate_calibration_keys():
    examples = generate_synthetic(100, seed=13)
    X, y, _ = examples_to_xy(examples)
    model = RulesModel()
    rep = evaluate(model, X, y, threshold=0.65)
    d = rep.to_dict()
    assert "confusion" in d and "calibration" in d and "ece" in d and "brier" in d
    assert "illustrative_expectancy" in d
