"""Observational pipeline:

DexScreener live universe
    → snapshot every N minutes
    → JSONL historical observations
    → 5m / 1h / 4h / 24h comparisons
    → cross-token narrative acceleration
    → META CLUSTER

Never calls the swap executor.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import requests

from ..config import Config
from .adapters import attach_holder_concentration, fetch_boosted_mints, snapshot_from_dexscreener
from .detector import MetaDetector
from .model import MetaReport

log = logging.getLogger(__name__)


class MetaPipeline:
    def __init__(self, cfg: Optional[Config] = None, session: Optional[requests.Session] = None,
                 observations_path: Optional[str] = None, universe_limit: int = 25):
        self.cfg = cfg or Config()
        self.session = session or requests.Session()
        self.detector = MetaDetector(self.cfg, self.session, observations_path)
        self.universe_limit = universe_limit
        interval = getattr(self.cfg, "meta_snapshot_interval_seconds", 300)
        self.interval = max(60, int(interval or 300))

    def snapshot_universe(self, now: Optional[float] = None,
                          fetch_mints: Optional[Callable[[], list[str]]] = None) -> MetaReport:
        now = now if now is not None else time.time()
        mints = (fetch_mints() if fetch_mints else fetch_boosted_mints(self.session))[:self.universe_limit]
        snapshots = []
        for mint in mints:
            snap = snapshot_from_dexscreener(mint, self.session, now=now)
            if snap is None:
                continue
            attach_holder_concentration(snap, self.cfg.rpc_url, self.session)
            snapshots.append(snap)
        log.info("meta pipeline snapshot: %d/%d tokens persisted", len(snapshots), len(mints))
        return self.detector.evaluate(snapshots, persist=True, persist_raw=True, now=now)

    def run_forever(self, cycles: Optional[int] = None) -> None:
        """Blocking observational loop. Does not start the trading bot."""
        seen = 0
        while cycles is None or seen < cycles:
            started = time.time()
            try:
                report = self.snapshot_universe(now=started)
                log.info("meta pipeline tick clusters=%d signals=%d coverage=%s",
                         len(report.clusters), len(report.signals), report.window_coverage)
            except Exception:
                log.exception("meta pipeline tick failed")
            seen += 1
            if cycles is not None and seen >= cycles:
                break
            sleep_for = self.interval - (time.time() - started)
            if sleep_for > 0:
                time.sleep(sleep_for)
