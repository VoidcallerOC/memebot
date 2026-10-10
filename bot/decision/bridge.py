"""Dry-run / paper bridge: decision engine → risk firewall → shadow journal.

Never authorizes live execution. When enabled, dry-run entries may be gated on
BUY + risk ALLOW. Live (armed) trading ignores this gate (Phase 11).

Shadow decisions use the same META enrichment path as MetaDetector.evaluate():
DexScreener TokenSnapshot → holder concentration → ObservationHistory windows
→ score_snapshot → MetaSignal. Sparse safety-only snapshots are fail-closed
(missing-feature threshold is never relaxed for scoring).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import requests

from ..config import Config
from ..meta.adapters import snapshot_from_dexscreener
from ..meta.detector import prepare_snapshot_for_scoring, score_snapshot
from ..meta.history import ObservationHistory
from ..meta.model import MetaSignal, TokenSnapshot
from ..paths import REPO_ROOT, resolve_repo_path
from ..portfolio import Portfolio
from ..risk import RiskManager
from ..safety import SafetyScreener, TokenSafety
from .engine import DecisionEngine, EngineResult
from .models.base import LocalModel
from .risk_firewall import RiskFirewall
from .shadow import ShadowJournal, record_from_engine
from .signal import BUY, REJECT

log = logging.getLogger(__name__)


def snapshot_from_safety(safety: TokenSafety, *, now: Optional[float] = None) -> TokenSnapshot:
    """Minimal TokenSnapshot from SafetyScreener output (sparse features).

    Used only as a fail-closed fallback when META market data is unavailable.
    Callers must NOT relax the missing-feature BUY threshold for these snaps.
    """
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


def build_meta_context(
    safety: TokenSafety,
    *,
    now: Optional[float] = None,
    session: Optional[requests.Session] = None,
    snapshot: Optional[TokenSnapshot] = None,
    signal: Optional[MetaSignal] = None,
    history: Optional[ObservationHistory] = None,
    rpc_url: str = "",
    enrich: bool = True,
) -> tuple[TokenSnapshot, Optional[MetaSignal], str]:
    """Resolve TokenSnapshot + MetaSignal with MetaDetector feature-source parity.

    Preference order:
      1. Caller-provided snapshot (enriched like MetaDetector unless enrich=False)
      2. Live DexScreener META snapshot + MetaDetector enrichment + score_snapshot
      3. Sparse safety fallback (no MetaSignal) — fail-closed on missing features

    Enrichment (when enrich=True and a market snapshot exists):
      holder concentration → ObservationHistory reconstructed windows → score

    Never invents market windows, social/wallet providers, or META scores.
    """
    now = now if now is not None else time.time()
    hist = history if history is not None else ObservationHistory()

    def _apply_safety_flags(snap: TokenSnapshot) -> TokenSnapshot:
        if any("liquidity" in r.lower() for r in safety.reasons):
            snap.liquidity_removed = True
        return snap

    def _enrich_and_score(snap: TokenSnapshot, source: str) -> tuple[TokenSnapshot, MetaSignal, str]:
        snap = _apply_safety_flags(snap)
        if snap.observed_at <= 0:
            snap.observed_at = now
        if enrich:
            prepare_snapshot_for_scoring(
                snap, hist, now=now, rpc_url=rpc_url, session=session,
                attach_holders=True,
            )
        # After enrichment, always re-score so MetaSignal matches the snap.
        # When enrich=False, honor a caller-provided signal (tests / replay).
        if signal is not None and not enrich:
            sig = signal
        else:
            sig = score_snapshot(snap)
        return snap, sig, source

    if snapshot is not None:
        return _enrich_and_score(snapshot, "provided")

    try:
        fetched = snapshot_from_dexscreener(safety.mint, session=session, now=now)
    except Exception as exc:
        log.warning("decision shadow: META snapshot fetch failed for %s: %s", safety.mint[:8], exc)
        fetched = None

    if fetched is not None:
        # Prefer safety screener price/liq when present (already screened).
        if safety.price_usd and (fetched.price_usd is None or fetched.price_usd <= 0):
            fetched.price_usd = safety.price_usd
        if safety.liquidity_usd and (fetched.liquidity_usd is None or fetched.liquidity_usd <= 0):
            fetched.liquidity_usd = safety.liquidity_usd
        if safety.symbol and not fetched.symbol:
            fetched.symbol = safety.symbol
        snap, sig, _ = _enrich_and_score(fetched, "meta_detector_parity")
        return snap, sig, "meta_detector_parity"

    snap = _apply_safety_flags(snapshot_from_safety(safety, now=now))
    return snap, None, "safety_fallback"


class DecisionShadowBridge:
    """Optional hook used by TradingBot in dry-run when decision_shadow_enabled."""

    def __init__(
        self,
        cfg: Config,
        risk: RiskManager,
        portfolio: Portfolio,
        screener: Optional[SafetyScreener] = None,
        *,
        session: Optional[requests.Session] = None,
        history: Optional[ObservationHistory] = None,
    ):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "decision_shadow_enabled", False))
        self.journal = ShadowJournal(getattr(cfg, "decision_shadow_file", None))
        self._engine: Optional[DecisionEngine] = None
        self._seen_events: set[str] = set()  # mint+bucket for duplicate suppression
        self._session = session if session is not None else requests.Session()
        self._history = history if history is not None else ObservationHistory()
        if not self.enabled:
            return
        configured = (getattr(cfg, "decision_model_path", "") or "").strip()
        model_path = resolve_repo_path(configured) if configured else ""
        model: Optional[LocalModel] = None
        if model_path and Path(model_path).exists():
            try:
                model = LocalModel.load(model_path)
                log.info("decision shadow: loaded model %s", model_path)
            except Exception as exc:
                log.error(
                    "decision shadow: corrupted/unreadable model at %s (%s) — NO TRADE mode, "
                    "all decisions REJECT. Fix: point DECISION_MODEL_PATH at a valid model "
                    "file (absolute, or relative to the repo root %s).",
                    model_path, exc, REPO_ROOT,
                )
                model = None
        else:
            log.error(
                "decision shadow enabled but model file not found at %s "
                "(DECISION_MODEL_PATH=%r) — NO TRADE mode, all decisions REJECT. "
                "Fix: set DECISION_MODEL_PATH to an existing model file, either absolute "
                "or relative to the repo root %s (not the CWD or MEMEBOT_DATA_DIR).",
                model_path or "<empty>", configured, REPO_ROOT,
            )
        firewall = RiskFirewall(cfg, risk=risk, screener=screener, portfolio=portfolio)
        self._engine = DecisionEngine(model, firewall)

    def evaluate_candidate(
        self,
        safety: TokenSafety,
        *,
        now: Optional[float] = None,
        wallet_entry_price: Optional[float] = None,
        snapshot: Optional[TokenSnapshot] = None,
        signal: Optional[MetaSignal] = None,
        enrich: bool = True,
    ) -> EngineResult:
        if self._engine is None:
            raise RuntimeError("DecisionShadowBridge not enabled")
        now = now if now is not None else time.time()
        # The collector (bot.meta run) appends to the observation file from
        # another process. Pick up its new rows every evaluation so the
        # reconstructed 4h window keeps working beyond the first ~70 minutes.
        try:
            self._history.refresh()
        except OSError as exc:
            log.warning("decision shadow: observation history refresh failed: %s", exc)
        snap, meta_signal, source = build_meta_context(
            safety,
            now=now,
            session=self._session,
            snapshot=snapshot,
            signal=signal,
            history=self._history,
            rpc_url=getattr(self.cfg, "rpc_url", "") or "",
            enrich=enrich,
        )
        # Duplicate event suppression: same mint within the same second.
        key = f"{safety.mint}:{int(now)}"
        if key in self._seen_events:
            # Force reject path for duplicate without re-inferring as BUY.
            # Never relax missing-feature safety — even for duplicates.
            result = self._engine.decide(
                snap, meta_signal, safety=safety, skip_network_screen=True, now=now,
                relax_missing_threshold=False,
            )
            result.decision.reasons.append("DUPLICATE_EVENT")
            if result.decision.action == BUY:
                result.decision.action = REJECT
                result.decision.reasons.append("DUPLICATE_FORCED_REJECT")
            rec = record_from_engine(result, wallet_entry_price=wallet_entry_price, now=now)
            rec.notes.append("duplicate_event_suppressed")
            rec.notes.append(f"meta_source={source}")
            self.journal.append(rec)
            return result
        self._seen_events.add(key)

        # Full META features when available. Sparse fallback keeps missing-feature
        # BUY threshold intact (RiskFirewall still enforces safety/risk hard rules).
        result = self._engine.decide(
            snap, meta_signal, safety=safety, skip_network_screen=True, now=now,
            relax_missing_threshold=False,
        )
        if source == "safety_fallback":
            result.decision.reasons.append("META_SNAPSHOT_UNAVAILABLE")
        rec = record_from_engine(result, wallet_entry_price=wallet_entry_price, now=now)
        rec.notes.append(f"meta_source={source}")
        if "4h" in snap.market:
            rec.notes.append("meta_window_4h=present")
        else:
            rec.notes.append("meta_window_4h=absent")
        if snap.top_holder_pct is not None:
            rec.notes.append("meta_holders=present")
        else:
            rec.notes.append("meta_holders=absent")
        self.journal.append(rec)
        return result

    def allows_paper_entry(self, result: EngineResult) -> bool:
        """Dry-run paper open only when model BUY and risk ALLOW."""
        return result.decision.action == BUY and result.risk.allowed
