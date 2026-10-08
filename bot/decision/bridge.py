"""Dry-run / paper bridge: decision engine → risk firewall → shadow journal.

Never authorizes live execution. When enabled, dry-run entries may be gated on
BUY + risk ALLOW. Live (armed) trading ignores this gate (Phase 11).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from ..config import Config
from ..portfolio import Portfolio
from ..risk import RiskManager
from ..safety import SafetyScreener, TokenSafety
from ..meta.model import MarketWindow, TokenSnapshot
from .engine import DecisionEngine, EngineResult
from .models.base import LocalModel
from .risk_firewall import RiskFirewall
from .shadow import ShadowJournal, record_from_engine
from .signal import BUY, REJECT

log = logging.getLogger(__name__)


def snapshot_from_safety(safety: TokenSafety, *, now: Optional[float] = None) -> TokenSnapshot:
    """Minimal TokenSnapshot from SafetyScreener output (sparse features OK)."""
    now = now if now is not None else time.time()
    return TokenSnapshot(
        mint=safety.mint,
        symbol=safety.symbol or "",
        observed_at=now,
        price_usd=safety.price_usd or None,
        liquidity_usd=safety.liquidity_usd or None,
        market={
            # No window data from safety alone → missing encodings apply.
        },
        source="safety_bridge",
    )


class DecisionShadowBridge:
    """Optional hook used by TradingBot in dry-run when decision_shadow_enabled."""

    def __init__(
        self,
        cfg: Config,
        risk: RiskManager,
        portfolio: Portfolio,
        screener: Optional[SafetyScreener] = None,
    ):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "decision_shadow_enabled", False))
        self.journal = ShadowJournal(getattr(cfg, "decision_shadow_file", None))
        self._engine: Optional[DecisionEngine] = None
        self._seen_events: set[str] = set()  # mint+bucket for duplicate suppression
        if not self.enabled:
            return
        model_path = getattr(cfg, "decision_model_path", "") or ""
        model: Optional[LocalModel] = None
        if model_path and Path(model_path).exists():
            try:
                model = LocalModel.load(model_path)
                log.info("decision shadow: loaded model %s", model_path)
            except Exception as exc:
                log.error("decision shadow: corrupted/unreadable model (%s) — NO TRADE mode", exc)
                model = None
        else:
            log.warning(
                "decision shadow enabled but model missing at %r — decisions REJECT",
                model_path,
            )
        firewall = RiskFirewall(cfg, risk=risk, screener=screener, portfolio=portfolio)
        self._engine = DecisionEngine(model, firewall)

    def evaluate_candidate(
        self,
        safety: TokenSafety,
        *,
        now: Optional[float] = None,
        wallet_entry_price: Optional[float] = None,
    ) -> EngineResult:
        if self._engine is None:
            raise RuntimeError("DecisionShadowBridge not enabled")
        now = now if now is not None else time.time()
        # Duplicate event suppression: same mint within the same second.
        key = f"{safety.mint}:{int(now)}"
        if key in self._seen_events:
            snap = snapshot_from_safety(safety, now=now)
            # Force reject path for duplicate without re-inferring as BUY.
            result = self._engine.decide(
                snap, None, safety=safety, skip_network_screen=True, now=now,
                relax_missing_threshold=bool(safety.passed),
            )
            result.decision.reasons.append("DUPLICATE_EVENT")
            if result.decision.action == BUY:
                result.decision.action = REJECT
                result.decision.reasons.append("DUPLICATE_FORCED_REJECT")
            rec = record_from_engine(result, wallet_entry_price=wallet_entry_price, now=now)
            rec.notes.append("duplicate_event_suppressed")
            self.journal.append(rec)
            return result
        self._seen_events.add(key)

        snap = snapshot_from_safety(safety, now=now)
        # Liquidity collapse / extreme flags from safety reasons.
        if any("liquidity" in r.lower() for r in safety.reasons):
            snap.liquidity_removed = True
        result = self._engine.decide(
            snap, None, safety=safety, skip_network_screen=True, now=now,
            # Safety already ran; allow model to score sparse bridge features.
            # Risk firewall still enforces safety/risk hard rules.
            relax_missing_threshold=bool(safety.passed),
        )
        rec = record_from_engine(result, wallet_entry_price=wallet_entry_price, now=now)
        self.journal.append(rec)
        return result

    def allows_paper_entry(self, result: EngineResult) -> bool:
        """Dry-run paper open only when model BUY and risk ALLOW."""
        return result.decision.action == BUY and result.risk.allowed
