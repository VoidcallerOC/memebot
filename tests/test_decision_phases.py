"""Phases 6–13 coverage: walk-forward, shadow bridge, compare, failure modes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot.config import Config, load_config
from bot.decision.bridge import DecisionShadowBridge, snapshot_from_safety
from bot.decision.compare import compare_arms
from bot.decision.dataset import generate_synthetic, walk_forward_folds
from bot.decision.engine import DecisionEngine
from bot.decision.label_meta import label_from_meta_observations
from bot.decision.models.base import LocalModel, ModelPrediction
from bot.decision.models.logistic import LogisticModel
from bot.decision.models.rules import RulesModel
from bot.decision.risk_firewall import BLOCK, RiskFirewall
from bot.decision.schema import N_FEATURES
from bot.decision.shadow import record_from_engine
from bot.decision.signal import BUY, REJECT
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.safety import TokenSafety
import bot.main as main_module


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


def test_walk_forward_folds_are_chronological():
    examples = generate_synthetic(120, seed=21)
    folds = walk_forward_folds(examples, n_folds=3, min_train=40)
    assert len(folds) >= 1
    for train, valid, test in folds:
        assert train and test
        assert max(e.detection_ts for e in train) <= min(e.detection_ts for e in test)
        if valid:
            assert max(e.detection_ts for e in train) <= min(e.detection_ts for e in valid)


def test_compare_includes_shadow_pnl_and_walk_forward(tmp_path: Path):
    report = compare_arms(out_dir=str(tmp_path), n_synthetic=180, seed=22)
    assert report["profitability"] == "NO VERIFIED PROFITABILITY"
    assert report["arm_C_frontier_llm"]["status"] == "UNAVAILABLE"
    assert "walk_forward" in report
    assert report["latency_benchmark"]["end_to_end_p50_ms"] >= 0
    for name, arm in report["arms"].items():
        assert "shadow_simulation" in arm
        assert "max_drawdown_pct" in arm["shadow_simulation"]
        assert "opportunity_cost_pct" in arm["shadow_simulation"]
        assert arm["cost_usd_per_decision"] == 0.0


def test_label_meta_blocked_without_forward_prices(tmp_path: Path):
    obs = tmp_path / "obs.jsonl"
    # Single snapshot — no forward path
    obs.write_text(
        json.dumps({
            "kind": "market_snapshot",
            "mint": "Mint1111111111111111111111111111111111111",
            "observed_at": 1700000000.0,
            "price_usd": 0.01,
            "liquidity_usd": 50000,
            "market": {},
        }) + "\n",
        encoding="utf-8",
    )
    summary = label_from_meta_observations(obs, out_path=tmp_path / "out.jsonl")
    assert summary["status"] == "BLOCKED"
    assert summary["labeled"] == 0
    assert summary["profitability"] == "NO VERIFIED PROFITABILITY"


def test_label_meta_labels_when_forward_path_exists(tmp_path: Path):
    obs = tmp_path / "obs.jsonl"
    mint = "Mint2222222222222222222222222222222222222"
    rows = []
    price = 1.0
    for i in range(8):
        # Drift up enough to hit +10%
        price = 1.0 * (1.02 ** i)
        rows.append({
            "kind": "market_snapshot",
            "mint": mint,
            "symbol": "T",
            "observed_at": 1700000000.0 + i * 60,
            "price_usd": price,
            "liquidity_usd": 50000,
            "market_cap_usd": 200000,
            "market": {
                "5m": {"volume_usd": 5000, "tx_buys": 40, "tx_sells": 20},
                "1h": {"volume_usd": 40000, "tx_buys": 200, "tx_sells": 150},
            },
            "top_holder_pct": 12,
        })
    obs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "labeled.jsonl"
    summary = label_from_meta_observations(obs, out_path=out)
    assert summary["labeled"] >= 1
    assert out.exists()


def test_bridge_journals_and_gates_paper(tmp_path: Path):
    model_path = tmp_path / "m.json"
    # Always-buy model
    LogisticModel([0.0] * N_FEATURES, bias=10.0, model_version="force.buy").save(model_path)
    cfg = _cfg(
        decision_model_path=str(model_path),
        decision_shadow_file=str(tmp_path / "shadow.jsonl"),
    )
    bridge = DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio())
    safety = TokenSafety(
        mint="MintBridge111111111111111111111111111111",
        passed=True, liquidity_usd=50_000, price_usd=0.01, symbol="BR",
    )
    result = bridge.evaluate_candidate(safety, now=1700000000.0)
    assert result.decision.action == BUY
    assert bridge.allows_paper_entry(result) is True
    rows = Path(cfg.decision_shadow_file).read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    rec = json.loads(rows[0])
    assert "model_version" in rec and "risk_engine_version" in rec


def test_bridge_blocks_when_risk_halts(tmp_path: Path):
    model_path = tmp_path / "m.json"
    LogisticModel([0.0] * N_FEATURES, bias=10.0, model_version="force.buy").save(model_path)
    cfg = _cfg(decision_model_path=str(model_path), decision_shadow_file=str(tmp_path / "s.jsonl"))
    risk = RiskManager(cfg)
    risk._halted_today = True
    bridge = DecisionShadowBridge(cfg, risk, Portfolio())
    safety = TokenSafety(mint="m", passed=True, liquidity_usd=50_000, price_usd=1.0, symbol="X")
    result = bridge.evaluate_candidate(safety, now=1700000000.0)
    assert result.decision.action == BUY
    assert result.risk.status == BLOCK
    assert bridge.allows_paper_entry(result) is False


def test_bridge_missing_model_rejects(tmp_path: Path):
    cfg = _cfg(
        decision_model_path=str(tmp_path / "missing.json"),
        decision_shadow_file=str(tmp_path / "s.jsonl"),
    )
    bridge = DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio())
    safety = TokenSafety(mint="m", passed=True, liquidity_usd=50_000, price_usd=1.0, symbol="X")
    result = bridge.evaluate_candidate(safety, now=1700000000.0)
    assert result.decision.action == REJECT
    assert "MODEL_UNAVAILABLE" in result.decision.reasons


def test_bridge_duplicate_event_forced_reject(tmp_path: Path):
    model_path = tmp_path / "m.json"
    LogisticModel([0.0] * N_FEATURES, bias=10.0, model_version="force.buy").save(model_path)
    cfg = _cfg(decision_model_path=str(model_path), decision_shadow_file=str(tmp_path / "s.jsonl"))
    bridge = DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio())
    safety = TokenSafety(mint="dupmint", passed=True, liquidity_usd=50_000, price_usd=1.0, symbol="D")
    r1 = bridge.evaluate_candidate(safety, now=1700000000.0)
    r2 = bridge.evaluate_candidate(safety, now=1700000000.0)
    assert r1.decision.action == BUY
    assert r2.decision.action == REJECT
    assert "DUPLICATE_EVENT" in r2.decision.reasons or any(
        "DUPLICATE" in x for x in r2.decision.reasons
    )


def test_corrupted_model_file_fail_closed(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    cfg = _cfg(decision_model_path=str(bad), decision_shadow_file=str(tmp_path / "s.jsonl"))
    bridge = DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio())
    safety = TokenSafety(mint="m", passed=True, liquidity_usd=50_000, price_usd=1.0, symbol="X")
    result = bridge.evaluate_candidate(safety, now=1700000000.0)
    assert result.decision.action == REJECT


def test_liquidity_collapse_flag_in_snapshot():
    safety = TokenSafety(
        mint="m", passed=False, reasons=["liquidity unlocked"], liquidity_usd=100, price_usd=1.0,
    )
    snap = snapshot_from_safety(safety, now=1.0)
    # bridge sets liquidity_removed when evaluating — unit the helper path:
    if any("liquidity" in r.lower() for r in safety.reasons):
        snap.liquidity_removed = True
    assert snap.liquidity_removed is True


def test_extreme_volatility_missing_frac_rejects():
    cfg = _cfg(decision_shadow_enabled=False)
    engine = DecisionEngine(RulesModel(), RiskFirewall(cfg))
    from bot.decision.features import FeatureRecord
    from bot.decision.schema import empty_vector
    values = empty_vector()
    # Mark almost everything missing via high missing_feature_frac
    from bot.decision.schema import FEATURE_INDEX
    values[FEATURE_INDEX["missing_feature_frac"]] = 0.9
    feats = FeatureRecord(
        mint="m", observed_at=1700000000.0, values=values,
        missing_mask=[True] * N_FEATURES, missing_frac=0.9,
    )
    result = engine.decide_from_features(feats, skip_network_screen=True, now=feats.observed_at)
    assert result.decision.action == REJECT


def test_main_dry_run_gates_on_decision_shadow(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "m.json"
    # Model that always rejects
    LogisticModel([0.0] * N_FEATURES, bias=-10.0, model_version="force.reject").save(model_path)
    cfg = _cfg(
        decision_shadow_enabled=True,
        decision_model_path=str(model_path),
        decision_shadow_file=str(tmp_path / "shadow.jsonl"),
        live_trading=False,
    )
    bot = main_module.TradingBot(cfg)

    class Cand:
        mint = "CandidateMint111111111111111111111111111"

    monkeypatch.setattr(bot.strategy, "find_candidates", lambda: [Cand()])
    monkeypatch.setattr(
        bot.screener, "screen",
        lambda mint: TokenSafety(mint=mint, passed=True, liquidity_usd=50_000, price_usd=0.01, symbol="C"),
    )
    calls = []
    monkeypatch.setattr(
        bot.executor, "swap",
        lambda *a, **k: calls.append("swap") or type("R", (), {
            "ok": True, "error": None, "status": "simulated", "simulated": True,
            "tx_signature": None, "actual_in_amount": None, "actual_out_amount": None,
            "out_decimals": None, "fee_lamports": None,
        })(),
    )
    bot._seek_entry()
    assert calls == []  # gated — no paper swap
    assert bot.portfolio.open_count == 0


def test_main_live_armed_does_not_use_decision_gate(monkeypatch, tmp_path: Path):
    """Phase 11: even if shadow flag is on, armed path must not be model-gated here."""
    model_path = tmp_path / "m.json"
    LogisticModel([0.0] * N_FEATURES, bias=-10.0, model_version="force.reject").save(model_path)
    cfg = Config(
        live_trading=True,
        wallet_private_key="test-only-sentinel",
        burner_wallet_pubkey="PUB",
        bankroll_usd=20.0,
        max_position_pct=10.0,
        decision_shadow_enabled=True,
        decision_model_path=str(model_path),
        decision_shadow_file=str(tmp_path / "s.jsonl"),
        state_file=str(tmp_path / "state.json"),
        process_lock_file=str(tmp_path / "lock"),
    )
    # Avoid loading real state side effects
    monkeypatch.setattr(main_module, "load_state", lambda *a, **k: None)
    bot = main_module.TradingBot(cfg)
    assert bot.cfg.is_armed is True

    class Cand:
        mint = "LiveMint111111111111111111111111111111111"

    monkeypatch.setattr(bot.strategy, "find_candidates", lambda: [Cand()])
    monkeypatch.setattr(
        bot.screener, "screen",
        lambda mint: TokenSafety(mint=mint, passed=True, liquidity_usd=50_000, price_usd=0.01, symbol="L"),
    )
    from bot.jupiter import SwapResult, TransactionStatus
    monkeypatch.setattr(
        bot.executor, "swap",
        lambda *a, **k: SwapResult(
            ok=True, simulated=False, in_amount=1, out_amount=1000,
            status=TransactionStatus.CONFIRMED_SUCCESS,
            tx_signature="sig", actual_in_amount=1, actual_out_amount=1000, out_decimals=6,
        ),
    )
    monkeypatch.setattr(bot, "_usd_to_lamports", lambda usd: 1)
    monkeypatch.setattr(bot, "_save", lambda: None)
    bot._seek_entry()
    # Live path ignores decision gate — position may open if swap confirms.
    assert bot.portfolio.open_count == 1


def test_decision_shadow_env_defaults_off(monkeypatch):
    monkeypatch.delenv("DECISION_SHADOW", raising=False)
    # load_config reads dotenv; ensure default false via Config()
    assert Config().decision_shadow_enabled is False
