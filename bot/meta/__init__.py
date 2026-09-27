"""Read-only Solana memecoin META DETECTOR.

This package identifies emerging narratives and whether attention is
accompanied by market/on-chain flow. It never executes trades, never
mutates wallet or portfolio state, and never bypasses ``safety.py`` /
``risk.py``.
"""

from .detector import MetaDetector
from .model import MetaReport, MetaSignal, Score
from .report import format_report

__all__ = ["MetaDetector", "MetaReport", "MetaSignal", "Score", "format_report"]
