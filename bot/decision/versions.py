"""Immutable version identifiers for decision-engine audit trails.

Every decision record must carry these so we can answer months later:
"Why did the bot make this decision?" Never silently replace a model.
"""
from __future__ import annotations

FEATURE_SCHEMA_VERSION = "features.v1"
DECISION_ENGINE_VERSION = "decision_engine.v1"
RISK_ENGINE_VERSION = "risk_firewall.v1"  # wraps existing bot.risk + bot.safety
# Model artifact files embed their own model_version; this is the package API.
MODEL_API_VERSION = "local_model.v1"

# Default action thresholds (probability of target=1). Tunable via config later.
BUY_CONFIDENCE_THRESHOLD = 0.65
WATCH_CONFIDENCE_THRESHOLD = 0.45
