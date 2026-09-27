"""Optional candidate source. Still only emits mints.

The existing loop continues to run safety.py and risk.py before any
execution path. This module never calls SwapExecutor.
"""
from __future__ import annotations

from typing import Optional

import requests

from ..config import Config
from ..strategy import Candidate
from .detector import MetaDetector


class MetaCandidateStrategy:
    def __init__(self, cfg: Config, session: Optional[requests.Session] = None):
        self.detector = MetaDetector(cfg, session)

    def find_candidates(self) -> list[Candidate]:
        report = self.detector.scan_live()
        return [Candidate(mint=mint) for mint in report.candidate_mints()]
