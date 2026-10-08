"""Feature schema for the local decision model.

Every feature has: name, source, type, units/range, missing-value behavior.
Do not feed arbitrary prose into the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .versions import FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    dtype: str  # float | int | binary
    units: str
    range_hint: str
    missing: str  # how missing values are encoded
    description: str


# Ordered feature vector. Index position is part of the schema contract.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "log_liquidity_usd", "TokenSnapshot.liquidity_usd", "float",
        "log1p(USD)", "[0, ~20]", "0.0 if missing",
        "Log liquidity depth at decision time",
    ),
    FeatureSpec(
        "log_market_cap_usd", "TokenSnapshot.market_cap_usd", "float",
        "log1p(USD)", "[0, ~25]", "0.0 if missing",
        "Log market cap at decision time",
    ),
    FeatureSpec(
        "token_age_hours", "pair_created_at vs observed_at", "float",
        "hours", "[0, inf)", "0.0 if missing",
        "Hours since pair creation",
    ),
    FeatureSpec(
        "vol_5m_usd", "MarketWindow[5m].volume_usd", "float",
        "USD", "[0, inf)", "0.0 if missing",
        "5m trailing volume",
    ),
    FeatureSpec(
        "vol_1h_usd", "MarketWindow[1h].volume_usd", "float",
        "USD", "[0, inf)", "0.0 if missing",
        "1h trailing volume",
    ),
    FeatureSpec(
        "vol_accel_live", "MetaSignal.trading_velocity.value", "float",
        "score 0-100", "[0, 100]", "-1.0 if UNVERIFIED",
        "Live trading velocity score from META",
    ),
    FeatureSpec(
        "vol_accel_hist", "MetaSignal.historical_trading_velocity.value", "float",
        "score 0-100", "[0, 100]", "-1.0 if UNVERIFIED",
        "Historical (5m vs 4h) trading velocity",
    ),
    FeatureSpec(
        "buy_sell_ratio_5m", "MarketWindow[5m] tx_buys/(buys+sells)", "float",
        "ratio", "[0, 1]", "0.5 if missing",
        "Buy pressure in 5m window",
    ),
    FeatureSpec(
        "buy_sell_ratio_1h", "MarketWindow[1h] tx_buys/(buys+sells)", "float",
        "ratio", "[0, 1]", "0.5 if missing",
        "Buy pressure in 1h window",
    ),
    FeatureSpec(
        "tx_velocity_5m", "MarketWindow[5m] buys+sells", "float",
        "count", "[0, inf)", "0.0 if missing",
        "Transaction count in 5m",
    ),
    FeatureSpec(
        "top_holder_pct", "TokenSnapshot.top_holder_pct", "float",
        "percent", "[0, 100]", "-1.0 if missing",
        "Largest holder share",
    ),
    FeatureSpec(
        "top10_holder_pct", "TokenSnapshot.top10_holder_pct", "float",
        "percent", "[0, 100]", "-1.0 if missing",
        "Top-10 holder share",
    ),
    FeatureSpec(
        "creator_pct", "TokenSnapshot.creator_pct", "float",
        "percent", "[0, 100]", "-1.0 if missing",
        "Creator allocation share",
    ),
    FeatureSpec(
        "liquidity_quality", "MetaSignal.liquidity_quality.value", "float",
        "score 0-100", "[0, 100]", "-1.0 if UNVERIFIED",
        "META liquidity quality score",
    ),
    FeatureSpec(
        "manipulation_risk", "MetaSignal.manipulation_risk.value", "float",
        "score 0-100", "[0, 100]", "-1.0 if UNVERIFIED",
        "META manipulation risk (higher = worse)",
    ),
    FeatureSpec(
        "wallet_velocity", "MetaSignal.wallet_velocity.value", "float",
        "score 0-100", "[0, 100]", "-1.0 if UNVERIFIED",
        "META wallet velocity",
    ),
    FeatureSpec(
        "attention_velocity", "MetaSignal.attention_velocity.value", "float",
        "score 0-100", "[0, 100]", "-1.0 if UNVERIFIED",
        "META attention velocity",
    ),
    FeatureSpec(
        "flow_aligned", "MetaSignal.flow_alignment", "binary",
        "0/1", "{0,1}", "0 if not ALIGNED",
        "Volume and wallet growth aligned",
    ),
    FeatureSpec(
        "provider_fresh", "MetaSignal.provider_freshness", "binary",
        "0/1", "{0,1}", "0 if not FRESH",
        "Provider data marked FRESH",
    ),
    FeatureSpec(
        "liquidity_removed", "TokenSnapshot.liquidity_removed", "binary",
        "0/1", "{0,1}", "0",
        "Liquidity removal flag",
    ),
    FeatureSpec(
        "lp_change_pct", "TokenSnapshot.lp_change_pct", "float",
        "percent", "(-100, +inf)", "0.0 if missing",
        "LP change percent",
    ),
    FeatureSpec(
        "in_active_meta", "MetaReport.candidate_mints membership", "binary",
        "0/1", "{0,1}", "0",
        "Token in EMERGING/ACCELERATING/PEAKING multi-token cluster",
    ),
    FeatureSpec(
        "price_change_5m_pct", "MarketWindow[5m].price_change_pct", "float",
        "percent", "(-100, +inf)", "0.0 if missing",
        "5m price change",
    ),
    FeatureSpec(
        "price_change_1h_pct", "MarketWindow[1h].price_change_pct", "float",
        "percent", "(-100, +inf)", "0.0 if missing",
        "1h price change",
    ),
    FeatureSpec(
        "missing_feature_frac", "derived", "float",
        "fraction", "[0, 1]", "computed",
        "Fraction of primary features that were missing/UNVERIFIED",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_SPECS)

# Decision target — chosen from data that can actually be labeled from
# price paths (META snapshots + forward prices). Not "should I buy?"
#
# Label = 1 iff, after a latency-adjusted entry, price reaches +TAKE_PROFIT_PCT
# before -STOP_PCT within HORIZON_SECONDS. Timeout / neither hit → 0.
TARGET_NAME = "hit_plus10_before_minus5"
TARGET_TAKE_PROFIT_PCT = 10.0
TARGET_STOP_PCT = 5.0
TARGET_HORIZON_SECONDS = 3600.0  # 1h observation window for the label
DEFAULT_DETECTION_LATENCY_SECONDS = 2.5  # realistic detect→execute lag


def schema_manifest() -> dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "n_features": N_FEATURES,
        "features": [
            {
                "name": s.name,
                "source": s.source,
                "dtype": s.dtype,
                "units": s.units,
                "range_hint": s.range_hint,
                "missing": s.missing,
                "description": s.description,
            }
            for s in FEATURE_SPECS
        ],
        "target": {
            "name": TARGET_NAME,
            "definition": (
                f"1 iff price reaches +{TARGET_TAKE_PROFIT_PCT:g}% before "
                f"-{TARGET_STOP_PCT:g}% within {TARGET_HORIZON_SECONDS:g}s "
                "after latency-adjusted entry; else 0"
            ),
            "take_profit_pct": TARGET_TAKE_PROFIT_PCT,
            "stop_pct": TARGET_STOP_PCT,
            "horizon_seconds": TARGET_HORIZON_SECONDS,
            "entry_rule": (
                "Simulated entry uses price at detection_ts + latency_seconds, "
                "never the observed wallet entry price"
            ),
            "leakage_risks": [
                "Must not use future META scores or post-entry volumes as features",
                "Must not use wallet entry price as our fill",
                "Random shuffle train/test is forbidden; use time-aware splits",
            ],
            "unavailable_features": [
                "wallet historical PnL / win rate (smart_money tracks mints only)",
                "copyability after realistic latency (no fill-time price journal yet)",
                "SOL regime / broad memecoin regime (not collected)",
                "dev transaction graph (creator_pct is a proxy only)",
                "social velocity (META marks UNVERIFIED without social provider)",
            ],
        },
    }


def empty_vector() -> list[float]:
    """Return a zero-ish vector with schema-default missing encodings.

    Score features that use -1 for UNVERIFIED are left at -1 so the model
    can learn "unknown" separately from "zero".
    """
    vec = [0.0] * N_FEATURES
    for name in (
        "vol_accel_live", "vol_accel_hist", "top_holder_pct", "top10_holder_pct",
        "creator_pct", "liquidity_quality", "manipulation_risk",
        "wallet_velocity", "attention_velocity",
    ):
        vec[FEATURE_INDEX[name]] = -1.0
    for name in ("buy_sell_ratio_5m", "buy_sell_ratio_1h"):
        vec[FEATURE_INDEX[name]] = 0.5
    return vec
