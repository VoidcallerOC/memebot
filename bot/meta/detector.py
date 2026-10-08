"""META DETECTOR orchestration. Read-only; never touches the executor.

Reconstructed implementation. ``score_snapshot`` turns one TokenSnapshot into
a MetaSignal; ``MetaDetector.evaluate`` scores a universe, attaches history-
derived windows, builds narrative clusters and persists observations.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

from ..config import Config
from .adapters import attach_holder_concentration, fetch_boosted_mints, snapshot_from_dexscreener
from .clusters import build_clusters
from .history import ObservationHistory, SIGNAL_ROW, THEME_ROW
from .model import MetaReport, MetaSignal, TokenSnapshot, UNVERIFIED
from .narrative import classify_narratives, score_narrative
from .scoring import (
    flow_alignment,
    provider_freshness,
    score_attention,
    score_creator,
    score_historical_trading,
    score_liquidity,
    score_manipulation,
    score_trading,
    score_wallets,
)

log = logging.getLogger(__name__)

DEFAULT_UNIVERSE_LIMIT = 25
PROVIDER_NOTES = (
    "social velocity: UNVERIFIED unless a social provider populates SocialWindow",
    "unique buyer/seller flow: UNVERIFIED unless a wallet indexer populates WalletWindow",
    "creator-linked wallet graph: UNVERIFIED (top-holder share via RPC is a proxy only)",
    "read-only observation; nothing here is a buy, sell, or hold instruction",
)


def prepare_snapshot_for_scoring(
    snap: TokenSnapshot,
    history: ObservationHistory,
    *,
    now: Optional[float] = None,
    rpc_url: str = "",
    session: Optional[requests.Session] = None,
    attach_holders: bool = True,
) -> dict:
    """Apply the same per-token enrichment MetaDetector.evaluate uses before scoring.

    Order matches production META:
      1. optional holder concentration (RPC; skipped if already on snap)
      2. ObservationHistory reconstructed windows (4h; never interpolated)
      3. caller runs score_snapshot()

    Does not invent social/wallet windows — absent providers stay UNVERIFIED.
    """
    now = now if now is not None else time.time()
    holders_attached = False
    if attach_holders:
        holders_attached = attach_holder_concentration(snap, rpc_url, session)
    windows = history.attach_reconstructed_windows(snap, now)
    return {
        "holders_attached": bool(holders_attached),
        "windows_attached": dict(windows),
    }


def score_snapshot(snap: TokenSnapshot) -> MetaSignal:
    themes = classify_narratives(snap)
    narrative_vel, narrative_fresh, catalyst = score_narrative(snap, themes)
    trading = score_trading(snap)
    historical = score_historical_trading(snap)
    wallets = score_wallets(snap)
    attention = score_attention(snap)
    liquidity = score_liquidity(snap)
    creator = score_creator(snap)
    alignment = flow_alignment(trading, wallets)
    manipulation = score_manipulation(snap, trading, wallets, attention, liquidity, creator, alignment)

    codes: list[str] = []
    for component in (attention, trading, historical, wallets, liquidity, creator, manipulation, narrative_vel):
        for code in component.reason_codes:
            if code not in codes:
                codes.append(code)
    if attention.details.get("social_concentration") and "SOCIAL_CONCENTRATION" not in codes:
        codes.append("SOCIAL_CONCENTRATION")
    if alignment == "VOLUME_SPIKE_WITHOUT_WALLET_GROWTH" and "VOLUME_WITHOUT_WALLET_GROWTH" not in codes:
        codes.append("VOLUME_WITHOUT_WALLET_GROWTH")
    if attention.verified and attention.details.get("state") in ("ACCELERATING", "EXTREME_ACCELERATION") \
            and wallets.status == UNVERIFIED:
        codes.append("ATTENTION_WITHOUT_WALLET_DATA")

    freshness = provider_freshness(snap)
    if freshness == "STALE":
        codes.append("PROVIDER_STALE")
    elif freshness == "CONFLICTING":
        codes.append("PROVIDER_CONFLICTING")

    return MetaSignal(
        token=snap.mint,
        symbol=snap.symbol,
        observed_at=snap.observed_at,
        narratives=themes,
        attention_velocity=attention,
        trading_velocity=trading,
        wallet_velocity=wallets,
        historical_trading_velocity=historical,
        liquidity_quality=liquidity,
        manipulation_risk=manipulation,
        creator_concentration=creator,
        narrative_velocity=narrative_vel,
        narrative_freshness=narrative_fresh,
        external_catalyst=catalyst,
        flow_alignment=alignment,
        provider_freshness=freshness,
        reason_codes=codes,
        price_usd=snap.price_usd,
        liquidity_usd=snap.liquidity_usd,
        market_cap_usd=snap.market_cap_usd,
    )


class MetaDetector:
    def __init__(self, cfg: Optional[Config] = None, session: Optional[requests.Session] = None,
                 observations_path: Optional[str] = None, universe_limit: Optional[int] = None):
        self.cfg = cfg or Config()
        self.session = session or requests.Session()
        self.history = ObservationHistory(observations_path)
        if universe_limit is None:
            try:
                universe_limit = int(os.getenv("META_UNIVERSE_LIMIT", str(DEFAULT_UNIVERSE_LIMIT)))
            except ValueError:
                universe_limit = DEFAULT_UNIVERSE_LIMIT
        self.universe_limit = max(1, universe_limit)

    # -- live ---------------------------------------------------------------

    def collect_live(self, now: Optional[float] = None) -> list[TokenSnapshot]:
        now = now if now is not None else time.time()
        snapshots: list[TokenSnapshot] = []
        for mint in fetch_boosted_mints(self.session)[: self.universe_limit]:
            snap = snapshot_from_dexscreener(mint, self.session, now=now)
            if snap is None:
                continue
            attach_holder_concentration(snap, self.cfg.rpc_url, self.session)
            snapshots.append(snap)
        return snapshots

    def scan_live(self, now: Optional[float] = None) -> MetaReport:
        now = now if now is not None else time.time()
        snapshots = self.collect_live(now)
        return self.evaluate(snapshots, persist=True, persist_raw=False, now=now)

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, snapshots: list[TokenSnapshot], persist: bool = True,
                 persist_raw: bool = False, now: Optional[float] = None) -> MetaReport:
        now = now if now is not None else time.time()
        self.history.load(force=True)
        prior_rows = len(self.history.load())

        coverage = {w: 0 for w in ("5m", "1h", "4h", "6h", "24h")}
        signals: list[MetaSignal] = []
        for snap in snapshots:
            # Holders are attached in collect_live / pipeline before evaluate.
            # Re-attach is a no-op when fields already present (shared-context safe).
            prepare_snapshot_for_scoring(
                snap, self.history, now=now,
                rpc_url=self.cfg.rpc_url, session=self.session,
                attach_holders=True,
            )
            for window, item in snap.market.items():
                if window in coverage and item.volume_usd is not None:
                    coverage[window] += 1
            signals.append(score_snapshot(snap))

        first_seen = self.history.theme_first_seen()
        clusters = build_clusters(signals, first_seen, now)
        counts = {c.theme: len(c.tokens) for c in clusters}
        for cluster in clusters:
            cluster.narrative_acceleration = self.history.theme_acceleration(cluster.theme, len(cluster.tokens), now)
            for code in cluster.narrative_acceleration.reason_codes:
                if code not in cluster.reason_codes:
                    cluster.reason_codes.append(code)

        metas = [c for c in clusters if c.is_meta]
        notes = [
            f"universe: {len(snapshots)} token(s) observed at {now:.0f}",
            f"history: {prior_rows} prior row(s) in {self.history.path}",
            f"clusters: {len(clusters)} theme(s), {len(metas)} with >= 2 tokens (single-token themes are not metas)",
            f"window coverage (tokens with verified volume): "
            + ", ".join(f"{w}={n}" for w, n in coverage.items())
            + " (4h is reconstructed from stored hourly windows, never interpolated)",
            *PROVIDER_NOTES,
        ]
        report = MetaReport(
            generated_at=now,
            signals=signals,
            clusters=clusters,
            notes=notes,
            window_coverage={**coverage, "tokens": len(snapshots)},
        )

        if persist_raw:
            for snap in snapshots:
                self.history.append_market_snapshot(snap, recorded_at=now)
        if persist:
            for signal in signals:
                self.history.append(SIGNAL_ROW, signal.to_dict(), recorded_at=now)
            self.history.append(THEME_ROW, {"counts": counts}, recorded_at=now)
        return report
