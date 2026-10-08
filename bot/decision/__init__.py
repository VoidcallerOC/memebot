"""Local ultra-low-latency decision engine for memebot.

CHAIN → FEATURES → LOCAL MODEL → BUY/WATCH/REJECT → DETERMINISTIC RISK → SHADOW

The model never authorizes a transaction. Live trading is out of scope for this
package until shadow gates are met. See docs/WAR_ROOM_DECISION_ENGINE.md.
"""
from __future__ import annotations

from .engine import DecisionEngine, EngineResult
from .signal import BUY, REJECT, WATCH, TypedDecision
from .versions import (
    DECISION_ENGINE_VERSION,
    FEATURE_SCHEMA_VERSION,
    RISK_ENGINE_VERSION,
)

__all__ = [
    "DecisionEngine",
    "EngineResult",
    "TypedDecision",
    "BUY",
    "WATCH",
    "REJECT",
    "FEATURE_SCHEMA_VERSION",
    "DECISION_ENGINE_VERSION",
    "RISK_ENGINE_VERSION",
]
