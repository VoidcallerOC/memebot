"""Data model for the read-only META DETECTOR.

Reconstructed implementation. The original Phase 3 source for this module was
never committed and could not be recovered; this file is derived from the
interfaces that the surviving package (``clusters.py``, ``narrative.py``,
``report.py``, ``pipeline.py``, ``__main__.py``) and the tests require.

Every score is 0-100 with an explicit status. A component the providers cannot
verify stays ``UNVERIFIED`` with ``value=None``; it is never coerced to zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

OK = "OK"
UNVERIFIED = "UNVERIFIED"
SCORE_STATES = frozenset({OK, UNVERIFIED})

FRESH = "FRESH"
STALE = "STALE"
CONFLICTING = "CONFLICTING"
PROVIDER_FRESHNESS_STATES = frozenset({FRESH, STALE, CONFLICTING, UNVERIFIED})

CLUSTER_STATES = frozenset({"NEW", "EMERGING", "ACCELERATING", "PEAKING", "DECAYING", "DORMANT"})

FLOW_ALIGNED = "ALIGNED"
FLOW_VOLUME_WITHOUT_WALLETS = "VOLUME_SPIKE_WITHOUT_WALLET_GROWTH"
FLOW_WALLETS_WITHOUT_VOLUME = "WALLET_GROWTH_WITHOUT_VOLUME"
FLOW_ALIGNMENTS = frozenset({FLOW_ALIGNED, FLOW_VOLUME_WITHOUT_WALLETS, FLOW_WALLETS_WITHOUT_VOLUME, UNVERIFIED})

LIVE_WINDOWS = ("5m", "1h", "6h", "24h")
HISTORY_WINDOWS = ("4h",)


@dataclass
class Score:
    status: str = UNVERIFIED
    value: Optional[float] = None
    formula: str = ""
    reason_codes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.status == OK and self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "formula": self.formula,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }


def ok(value: float, formula: str, reason_codes: Optional[list[str]] = None, **details: Any) -> Score:
    return Score(OK, float(value), formula, list(reason_codes or []), dict(details))


def unverified(reason: str, **details: Any) -> Score:
    details = dict(details)
    details["reason"] = reason
    return Score(UNVERIFIED, None, "", [], details)


@dataclass
class MarketWindow:
    volume_usd: Optional[float] = None
    tx_buys: Optional[int] = None
    tx_sells: Optional[int] = None
    price_change_pct: Optional[float] = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MarketWindow":
        return cls(
            volume_usd=_opt_float(raw.get("volume_usd")),
            tx_buys=_opt_int(raw.get("tx_buys")),
            tx_sells=_opt_int(raw.get("tx_sells")),
            price_change_pct=_opt_float(raw.get("price_change_pct")),
            source=str(raw.get("source") or ""),
        )


@dataclass
class SocialWindow:
    mentions: Optional[int] = None
    unique_contributors: Optional[int] = None
    engagement: Optional[float] = None
    top_account_share: Optional[float] = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SocialWindow":
        return cls(
            mentions=_opt_int(raw.get("mentions")),
            unique_contributors=_opt_int(raw.get("unique_contributors")),
            engagement=_opt_float(raw.get("engagement")),
            top_account_share=_opt_float(raw.get("top_account_share")),
            source=str(raw.get("source") or ""),
        )


@dataclass
class WalletWindow:
    new_holders: Optional[int] = None
    unique_buyers: Optional[int] = None
    unique_sellers: Optional[int] = None
    active_wallets: Optional[int] = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WalletWindow":
        return cls(
            new_holders=_opt_int(raw.get("new_holders")),
            unique_buyers=_opt_int(raw.get("unique_buyers")),
            unique_sellers=_opt_int(raw.get("unique_sellers")),
            active_wallets=_opt_int(raw.get("active_wallets")),
            source=str(raw.get("source") or ""),
        )


@dataclass
class TokenSnapshot:
    """One timestamped observation of one token. Missing provider data is None."""

    mint: str
    symbol: str = ""
    name: str = ""
    description: str = ""
    observed_at: float = 0.0
    price_usd: Optional[float] = None
    liquidity_usd: Optional[float] = None
    market_cap_usd: Optional[float] = None
    pair_address: str = ""
    pair_created_at: Optional[float] = None
    texts: list[str] = field(default_factory=list)
    market: dict[str, MarketWindow] = field(default_factory=dict)
    social: dict[str, SocialWindow] = field(default_factory=dict)
    wallets: dict[str, WalletWindow] = field(default_factory=dict)
    liquidity_removed: bool = False
    lp_change_pct: Optional[float] = None
    creator_pct: Optional[float] = None
    top_holder_pct: Optional[float] = None
    top10_holder_pct: Optional[float] = None
    external_catalyst_verified: bool = False
    stale: bool = False
    conflicting: bool = False
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "name": self.name,
            "description": self.description,
            "observed_at": self.observed_at,
            "price_usd": self.price_usd,
            "liquidity_usd": self.liquidity_usd,
            "market_cap_usd": self.market_cap_usd,
            "pair_address": self.pair_address,
            "pair_created_at": self.pair_created_at,
            "texts": list(self.texts),
            "market": {k: v.to_dict() for k, v in self.market.items()},
            "social": {k: v.to_dict() for k, v in self.social.items()},
            "wallets": {k: v.to_dict() for k, v in self.wallets.items()},
            "liquidity_removed": self.liquidity_removed,
            "lp_change_pct": self.lp_change_pct,
            "creator_pct": self.creator_pct,
            "top_holder_pct": self.top_holder_pct,
            "top10_holder_pct": self.top10_holder_pct,
            "external_catalyst_verified": self.external_catalyst_verified,
            "stale": self.stale,
            "conflicting": self.conflicting,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TokenSnapshot":
        return cls(
            mint=str(raw.get("mint") or ""),
            symbol=str(raw.get("symbol") or ""),
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            observed_at=float(raw.get("observed_at") or 0.0),
            price_usd=_opt_float(raw.get("price_usd")),
            liquidity_usd=_opt_float(raw.get("liquidity_usd")),
            market_cap_usd=_opt_float(raw.get("market_cap_usd")),
            pair_address=str(raw.get("pair_address") or ""),
            pair_created_at=_opt_float(raw.get("pair_created_at")),
            texts=[str(t) for t in (raw.get("texts") or [])],
            market={k: MarketWindow.from_dict(v) for k, v in (raw.get("market") or {}).items() if isinstance(v, dict)},
            social={k: SocialWindow.from_dict(v) for k, v in (raw.get("social") or {}).items() if isinstance(v, dict)},
            wallets={k: WalletWindow.from_dict(v) for k, v in (raw.get("wallets") or {}).items() if isinstance(v, dict)},
            liquidity_removed=bool(raw.get("liquidity_removed")),
            lp_change_pct=_opt_float(raw.get("lp_change_pct")),
            creator_pct=_opt_float(raw.get("creator_pct")),
            top_holder_pct=_opt_float(raw.get("top_holder_pct")),
            top10_holder_pct=_opt_float(raw.get("top10_holder_pct")),
            external_catalyst_verified=bool(raw.get("external_catalyst_verified")),
            stale=bool(raw.get("stale")),
            conflicting=bool(raw.get("conflicting")),
            source=str(raw.get("source") or ""),
        )


@dataclass
class MetaSignal:
    token: str
    symbol: str = ""
    observed_at: float = 0.0
    narratives: list[str] = field(default_factory=list)
    attention_velocity: Score = field(default_factory=Score)
    trading_velocity: Score = field(default_factory=Score)
    wallet_velocity: Score = field(default_factory=Score)
    historical_trading_velocity: Score = field(default_factory=Score)
    liquidity_quality: Score = field(default_factory=Score)
    manipulation_risk: Score = field(default_factory=Score)
    creator_concentration: Score = field(default_factory=Score)
    narrative_velocity: Score = field(default_factory=Score)
    narrative_freshness: Score = field(default_factory=Score)
    external_catalyst: str = UNVERIFIED
    flow_alignment: str = UNVERIFIED
    provider_freshness: str = UNVERIFIED
    reason_codes: list[str] = field(default_factory=list)
    price_usd: Optional[float] = None
    liquidity_usd: Optional[float] = None
    market_cap_usd: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "symbol": self.symbol,
            "observed_at": self.observed_at,
            "narratives": list(self.narratives),
            "attention_velocity": self.attention_velocity.to_dict(),
            "trading_velocity": self.trading_velocity.to_dict(),
            "wallet_velocity": self.wallet_velocity.to_dict(),
            "historical_trading_velocity": self.historical_trading_velocity.to_dict(),
            "liquidity_quality": self.liquidity_quality.to_dict(),
            "manipulation_risk": self.manipulation_risk.to_dict(),
            "creator_concentration": self.creator_concentration.to_dict(),
            "narrative_velocity": self.narrative_velocity.to_dict(),
            "narrative_freshness": self.narrative_freshness.to_dict(),
            "external_catalyst": self.external_catalyst,
            "flow_alignment": self.flow_alignment,
            "provider_freshness": self.provider_freshness,
            "reason_codes": list(self.reason_codes),
            "price_usd": self.price_usd,
            "liquidity_usd": self.liquidity_usd,
            "market_cap_usd": self.market_cap_usd,
        }


@dataclass
class MetaCluster:
    theme: str
    tokens: list[str]
    attention_velocity: Score
    volume_velocity: Score
    wallet_velocity: Score
    narrative_confidence: Score
    first_detected: float
    last_updated: float
    status: str
    reason_codes: list[str] = field(default_factory=list)
    narrative_acceleration: Score = field(default_factory=Score)

    @property
    def is_meta(self) -> bool:
        """A theme carried by a single token is not a market-wide meta."""
        return len(self.tokens) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "tokens": list(self.tokens),
            "member_count": len(self.tokens),
            "is_meta": self.is_meta,
            "attention_velocity": self.attention_velocity.to_dict(),
            "volume_velocity": self.volume_velocity.to_dict(),
            "wallet_velocity": self.wallet_velocity.to_dict(),
            "narrative_confidence": self.narrative_confidence.to_dict(),
            "narrative_acceleration": self.narrative_acceleration.to_dict(),
            "first_detected": self.first_detected,
            "last_updated": self.last_updated,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class MetaReport:
    generated_at: float
    signals: list[MetaSignal] = field(default_factory=list)
    clusters: list[MetaCluster] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    window_coverage: dict[str, Any] = field(default_factory=dict)

    def candidate_mints(self) -> list[str]:
        """Mints that belong to a multi-token cluster that is not decaying.

        Observational only: every mint still goes through safety.py and
        risk.py before the trading loop may act on it.
        """
        by_token = {s.token: s for s in self.signals}
        picked: list[str] = []
        for cluster in self.clusters:
            if not cluster.is_meta or cluster.status not in {"EMERGING", "ACCELERATING", "PEAKING"}:
                continue
            for mint in cluster.tokens:
                signal = by_token.get(mint)
                if signal is None or mint in picked:
                    continue
                if signal.provider_freshness in {STALE, CONFLICTING}:
                    continue
                if "LIQUIDITY_REMOVAL" in signal.reason_codes:
                    continue
                picked.append(mint)
        return picked

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "notes": list(self.notes),
            "window_coverage": dict(self.window_coverage),
            "clusters": [c.to_dict() for c in self.clusters],
            "signals": [s.to_dict() for s in self.signals],
        }


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
