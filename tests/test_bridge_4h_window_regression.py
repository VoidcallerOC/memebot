"""Regression: the shadow bridge's 4h META window must keep updating while the
collector (another process) appends to the observation file.

On a81e9e2 the bridge cached ObservationHistory rows on first use and never
reloaded them, so the reconstructed 4h window disappeared ~70 min after the
first evaluation. Synthetic rows only, tmp_path only, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

from bot.config import Config
from bot.decision.bridge import DecisionShadowBridge
from bot.decision.models.logistic import LogisticModel
from bot.decision.schema import N_FEATURES
from bot.meta.history import ObservationHistory
from bot.meta.model import MarketWindow, TokenSnapshot
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.safety import TokenSafety

MINT = "Regr4hMint1111111111111111111111111111111"
T0 = 1_700_200_000.0
STEP = 300.0  # collector tick


class _NoNetworkSession:
    def __getattr__(self, name):
        raise AssertionError(f"network access attempted via session.{name}")


def _market_row(ts: float) -> str:
    return json.dumps({
        "kind": "market_snapshot",
        "recorded_at": ts,
        "observed_at": ts,
        "mint": MINT,
        "symbol": "RGR",
        "price_usd": 0.05,
        "market": {"1h": {"volume_usd": 9_000.0, "tx_buys": 30, "tx_sells": 20}},
    })


class _FixtureCollector:
    """Plays the role of `bot.meta run`: appends rows every 300 s (simulated)."""

    def __init__(self, path: Path, start: float):
        self.path = path
        self.next_ts = start

    def advance_to(self, until: float) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            while self.next_ts <= until:
                fh.write(_market_row(self.next_ts) + "\n")
                self.next_ts += STEP


def _live_snap(now: float) -> TokenSnapshot:
    snap = TokenSnapshot(
        mint=MINT, symbol="RGR", observed_at=now, price_usd=0.05,
        liquidity_usd=75_000.0, market_cap_usd=300_000.0, pair_created_at=now - 86_400.0,
        market={
            "5m": MarketWindow(volume_usd=1_000, tx_buys=10, tx_sells=8, price_change_pct=1.0, source="dexscreener"),
            "1h": MarketWindow(volume_usd=9_000, tx_buys=30, tx_sells=20, price_change_pct=2.0, source="dexscreener"),
        },
        source="dexscreener",
    )
    snap.top_holder_pct = 12.0   # already present -> holder RPC skipped
    snap.top10_holder_pct = 35.0
    return snap


def test_bridge_4h_window_survives_beyond_70_min(tmp_path: Path):
    obs = tmp_path / "meta_observations.jsonl"
    shadow = tmp_path / "decision_shadow.jsonl"
    model_path = tmp_path / "model.json"
    LogisticModel([0.0] * N_FEATURES, bias=0.0, model_version="regression.test").save(model_path)

    collector = _FixtureCollector(obs, start=T0 - 4 * 3600.0)
    collector.advance_to(T0)  # >= 4h of genuine history before the bridge starts

    cfg = Config(
        live_trading=False, bankroll_usd=20.0, max_position_pct=10.0,
        decision_shadow_enabled=True, decision_model_path=str(model_path),
        decision_shadow_file=str(shadow), rpc_url="http://127.0.0.1:9",
    )
    # Built ONCE, exactly like TradingBot does.
    bridge = DecisionShadowBridge(
        cfg, RiskManager(cfg), Portfolio(), session=_NoNetworkSession(),
        history=ObservationHistory(str(obs)),
    )
    safety = TokenSafety(mint=MINT, passed=True, liquidity_usd=75_000, price_usd=0.05, symbol="RGR")

    results = {}
    for label, offset in (("+0", 0.0), ("+75min", 75 * 60.0), ("+3h", 3 * 3600.0), ("+6h", 6 * 3600.0)):
        now = T0 + offset
        collector.advance_to(now)  # collector keeps appending between evaluations
        snap = _live_snap(now)
        bridge.evaluate_candidate(safety, now=now, snapshot=snap)
        rec = json.loads(shadow.read_text(encoding="utf-8").strip().splitlines()[-1])
        results[label] = ("4h" in snap.market, "meta_window_4h=present" in rec["notes"])

    print("4h window per checkpoint (snap.market has 4h, journal note present):", results)
    assert results == {k: (True, True) for k in ("+0", "+75min", "+3h", "+6h")}, results
