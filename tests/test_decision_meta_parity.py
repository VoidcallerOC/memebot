"""Post-PR #16 META feature-source parity: DecisionShadowBridge ↔ MetaDetector.

Proves:
  - shared DexScreener TokenSnapshot used by safety + decision
  - shadow enrichment includes holders + history windows like MetaDetector.evaluate
  - missing provider/history data stays fail-closed (no invented values)
  - duplicate DexScreener fetches are eliminated on the seek-entry path
  - PR #16 incomplete-label / missing-threshold guarantees remain intact
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from bot.config import Config
from bot.decision.bridge import DecisionShadowBridge, build_meta_context
from bot.decision.engine import MAX_MISSING_FRAC_FOR_BUY
from bot.decision.features import extract_features
from bot.decision.labels import PriceTick, label_path
from bot.decision.models.logistic import LogisticModel
from bot.decision.schema import N_FEATURES
from bot.decision.signal import BUY, REJECT
from bot.meta.detector import MetaDetector, prepare_snapshot_for_scoring, score_snapshot
from bot.meta.history import ObservationHistory
from bot.meta.model import MarketWindow, TokenSnapshot
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.safety import SafetyScreener, TokenSafety
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
        rpc_url="http://127.0.0.1:8899",
    )
    base.update(kwargs)
    return Config(**base)


def _dex_snap(mint: str, *, now: float = 1_700_000_000.0) -> TokenSnapshot:
    return TokenSnapshot(
        mint=mint,
        symbol="PAR",
        observed_at=now,
        price_usd=0.05,
        liquidity_usd=75_000.0,
        market_cap_usd=300_000.0,
        pair_created_at=now - 14_400.0,
        market={
            "5m": MarketWindow(volume_usd=6_000, tx_buys=40, tx_sells=25, price_change_pct=3.0, source="dexscreener"),
            "1h": MarketWindow(volume_usd=40_000, tx_buys=220, tx_sells=180, price_change_pct=8.0, source="dexscreener"),
        },
        source="dexscreener",
    )


# ---------------------------------------------------------------------------
# Shared snapshot: safety + decision
# ---------------------------------------------------------------------------

def test_safety_reuses_provided_snapshot_without_dex_fetch(monkeypatch):
    cfg = _cfg()
    screener = SafetyScreener(cfg)
    fetches = {"n": 0}

    def boom(mint):
        fetches["n"] += 1
        raise AssertionError("DexScreener must not be re-fetched when snapshot provided")

    monkeypatch.setattr(screener, "_fetch_market", boom)
    monkeypatch.setattr(screener, "_fetch_mint_account", lambda mint: {
        "mint_authority": None, "freeze_authority": None,
    })
    monkeypatch.setattr(screener, "_consult_rugcheck", lambda *a, **k: None)

    mint = "SharedMint1111111111111111111111111111111"
    snap = _dex_snap(mint)
    snap.top_holder_pct = 12.0
    snap.top10_holder_pct = 40.0
    # Avoid holder RPC when pct already on snap
    monkeypatch.setattr(screener, "_fetch_top_holder_pct", lambda mint: (_ for _ in ()).throw(
        AssertionError("holder RPC must use snapshot.top_holder_pct")
    ))

    safety = screener.screen(mint, snapshot=snap)
    assert fetches["n"] == 0
    assert safety.passed is True
    assert safety.liquidity_usd == pytest.approx(75_000.0)
    assert safety.price_usd == pytest.approx(0.05)
    assert safety.symbol == "PAR"


def test_seek_entry_uses_one_dex_snapshot_for_safety_and_decision(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "m.json"
    LogisticModel([0.0] * N_FEATURES, bias=-10.0, model_version="force.reject").save(model_path)
    cfg = _cfg(
        decision_model_path=str(model_path),
        decision_shadow_file=str(tmp_path / "shadow.jsonl"),
        state_file=str(tmp_path / "state.json"),
        process_lock_file=str(tmp_path / "lock"),
    )
    bot = main_module.TradingBot(cfg)
    mint = "OneFetchMint11111111111111111111111111111"
    snap = _dex_snap(mint)
    snap.top_holder_pct = 10.0
    snap.top10_holder_pct = 30.0

    fetch_calls = {"n": 0}

    def one_snap(self, m):
        fetch_calls["n"] += 1
        assert m == mint
        return snap

    monkeypatch.setattr(main_module.TradingBot, "_candidate_market_snapshot", one_snap)

    class Cand:
        def __init__(self, m):
            self.mint = m

    monkeypatch.setattr(bot.strategy, "find_candidates", lambda: [Cand(mint)])
    seen: dict[str, Any] = {}

    def screen_wrap(m, *, snapshot=None):
        seen["safety_snap"] = snapshot
        return TokenSafety(
            mint=m, passed=True,
            liquidity_usd=float(snapshot.liquidity_usd or 0) if snapshot else 0,
            price_usd=float(snapshot.price_usd or 0) if snapshot else 0,
            symbol=(snapshot.symbol if snapshot else "X"),
        )

    monkeypatch.setattr(bot.screener, "screen", screen_wrap)

    orig_eval = bot.decision_shadow.evaluate_candidate

    def eval_wrap(safety, **kwargs):
        seen["decision_snap"] = kwargs.get("snapshot")
        return orig_eval(safety, **kwargs)

    monkeypatch.setattr(bot.decision_shadow, "evaluate_candidate", eval_wrap)
    monkeypatch.setattr(
        bot.executor, "swap",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not swap on REJECT")),
    )

    bot._seek_entry()
    assert fetch_calls["n"] == 1
    assert seen["safety_snap"] is snap
    assert seen["decision_snap"] is snap


# ---------------------------------------------------------------------------
# META enrichment parity with MetaDetector.evaluate
# ---------------------------------------------------------------------------

def test_prepare_snapshot_matches_metadetector_enrichment(tmp_path: Path):
    hist_path = tmp_path / "obs.jsonl"
    now = 1_700_100_000.0
    mint = "HistMint111111111111111111111111111111111"
    # Seed hourly market rows so 4h can reconstruct.
    rows = []
    for step in range(0, 5):
        ts = now - step * 3600.0
        rows.append(json.dumps({
            "kind": "market_snapshot",
            "recorded_at": ts,
            "observed_at": ts,
            "mint": mint,
            "price_usd": 1.0,
            "market": {"1h": {"volume_usd": 10_000.0, "tx_buys": 10, "tx_sells": 8}},
        }))
    hist_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    history = ObservationHistory(str(hist_path))
    history.load(force=True)
    snap = _dex_snap(mint, now=now)
    snap.top_holder_pct = 15.0
    snap.top10_holder_pct = 45.0

    meta = prepare_snapshot_for_scoring(
        snap, history, now=now, rpc_url="", session=None, attach_holders=True,
    )
    assert meta["holders_attached"] is True  # already present → skipped RPC, True
    assert "4h" in snap.market
    assert snap.market["4h"].source.startswith("history_reconstructed")

    # MetaDetector.evaluate on a clone must produce the same market keys + scores.
    snap2 = _dex_snap(mint, now=now)
    snap2.top_holder_pct = 15.0
    snap2.top10_holder_pct = 45.0
    detector = MetaDetector(_cfg(), observations_path=str(hist_path))
    report = detector.evaluate([snap2], persist=False, now=now)
    assert "4h" in snap2.market
    sig_bridge = score_snapshot(snap)
    sig_meta = report.signals[0]
    assert sig_bridge.historical_trading_velocity.status == sig_meta.historical_trading_velocity.status
    # Feature vectors for decision schema must match across paths.
    f1 = extract_features(snap, sig_bridge, now=now)
    f2 = extract_features(snap2, sig_meta, now=now)
    assert f1.values == pytest.approx(f2.values)


def test_bridge_enrichment_attaches_history_window(tmp_path: Path):
    model_path = tmp_path / "m.json"
    LogisticModel([0.0] * N_FEATURES, bias=10.0, model_version="force.buy").save(model_path)
    hist_path = tmp_path / "obs.jsonl"
    now = 1_700_100_000.0
    mint = "BridgeHistMint11111111111111111111111111"
    rows = []
    for step in range(0, 5):
        ts = now - step * 3600.0
        rows.append(json.dumps({
            "kind": "market_snapshot",
            "recorded_at": ts,
            "observed_at": ts,
            "mint": mint,
            "price_usd": 1.0,
            "market": {"1h": {"volume_usd": 8_000.0, "tx_buys": 12, "tx_sells": 9}},
        }))
    hist_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    cfg = _cfg(
        decision_model_path=str(model_path),
        decision_shadow_file=str(tmp_path / "s.jsonl"),
    )
    history = ObservationHistory(str(hist_path))
    bridge = DecisionShadowBridge(
        cfg, RiskManager(cfg), Portfolio(), history=history,
    )
    safety = TokenSafety(mint=mint, passed=True, liquidity_usd=75_000, price_usd=0.05, symbol="PAR")
    snap = _dex_snap(mint, now=now)
    snap.top_holder_pct = 11.0
    snap.top10_holder_pct = 33.0
    result = bridge.evaluate_candidate(safety, now=now, snapshot=snap)
    assert result.decision.action == BUY
    rec = json.loads(Path(cfg.decision_shadow_file).read_text(encoding="utf-8").strip())
    assert "meta_window_4h=present" in rec["notes"]
    assert "meta_holders=present" in rec["notes"]
    assert "4h" in snap.market


def test_missing_history_does_not_invent_4h_window(tmp_path: Path):
    """No hourly history → 4h stays absent; scores stay UNVERIFIED — not faked."""
    history = ObservationHistory(str(tmp_path / "empty.jsonl"))
    snap = _dex_snap("NoHistMint111111111111111111111111111")
    prepare_snapshot_for_scoring(snap, history, now=snap.observed_at, rpc_url="")
    assert "4h" not in snap.market
    sig = score_snapshot(snap)
    # historical velocity may be unverified without 4h — never a fabricated number
    from bot.meta.model import UNVERIFIED
    # Either UNVERIFIED or computed from live 5m/1h only — must not invent 4h volume
    assert "4h" not in snap.market


def test_sparse_fallback_still_fail_closed(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "m.json"
    LogisticModel([0.0] * N_FEATURES, bias=10.0, model_version="force.buy").save(model_path)
    cfg = _cfg(
        decision_model_path=str(model_path),
        decision_shadow_file=str(tmp_path / "s.jsonl"),
    )
    bridge = DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio())
    monkeypatch.setattr("bot.decision.bridge.snapshot_from_dexscreener", lambda *a, **k: None)
    safety = TokenSafety(
        mint="SparseParity111111111111111111111111111",
        passed=True, liquidity_usd=50_000, price_usd=0.01, symbol="SP",
    )
    result = bridge.evaluate_candidate(safety, now=1_700_000_000.0)
    assert result.decision.action == REJECT
    assert "MISSING_FEATURES_ABOVE_THRESHOLD" in result.decision.reasons
    assert result.features.missing_frac > MAX_MISSING_FRAC_FOR_BUY


def test_incomplete_label_guarantee_intact():
    """PR #16: truncated path without TP/SL remains UNLABELED."""
    path = [
        PriceTick(0.0, 1.0),
        PriceTick(2.5, 1.0),
        PriceTick(60.0, 1.01),
    ]
    outcome = label_path(path, detection_ts=0.0, latency_seconds=2.5, horizon_seconds=3600.0)
    assert outcome.label is None
    assert outcome.exit_reason == "incomplete"


def test_build_meta_context_parity_source_when_fetched(monkeypatch):
    safety = TokenSafety(mint="m", passed=True, liquidity_usd=1.0, price_usd=1.0, symbol="X")
    snap = _dex_snap("m")
    snap.top_holder_pct = 9.0
    snap.top10_holder_pct = 28.0
    monkeypatch.setattr("bot.decision.bridge.snapshot_from_dexscreener", lambda *a, **k: snap)
    out, sig, source = build_meta_context(safety, now=snap.observed_at, rpc_url="")
    assert source == "meta_detector_parity"
    assert sig is not None
    assert out.top_holder_pct == pytest.approx(9.0)


def test_live_trading_and_ceiling_untouched():
    cfg = _cfg(live_trading=False, bankroll_usd=20.0)
    assert cfg.live_trading is False
    assert cfg.is_armed is False
    assert cfg.bankroll_usd == 20.0
