"""CLI: python -m bot.meta [report [--persist] [--json]|snapshot|run]

Read-only. Never talks to the executor.

  * ``run``      long-running collector. Single instance (flock on
                 ``<MEMEBOT_DATA_DIR>/meta.lock``, override META_PROCESS_LOCK_FILE;
                 exits 3 if held). SIGTERM/SIGINT finish the current tick,
                 release the lock and exit 0.
  * ``snapshot`` one collector tick (persists rows; refused while ``run`` holds
                 the lock).
  * ``report``   ad-hoc report. Does NOT write observation rows unless
                 ``--persist`` is given (then refused while the lock is held).
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import threading
from typing import Any, Callable, Optional

from ..config import load_config
from ..paths import log_resolved_paths, meta_lock_path
from ..process_lock import ProcessLock
from .detector import MetaDetector
from .history import default_observations_path
from .pipeline import MetaPipeline
from .report import format_report

log = logging.getLogger(__name__)

EXIT_LOCK_HELD = 3


def make_stop_handler(stop: threading.Event) -> Callable[[int, Any], None]:
    """Signal handler that only requests a stop; the current tick finishes."""
    def _handle(signum: int, _frame: Any) -> None:
        log.info("meta collector received signal %s; stopping after current tick", signum)
        stop.set()
    return _handle


def run_collector(cfg: Any, *, pipeline: Optional[MetaPipeline] = None,
                  lock_path: Optional[str] = None,
                  stop_event: Optional[threading.Event] = None,
                  install_signal_handlers: bool = True) -> int:
    lock = ProcessLock(lock_path or meta_lock_path())
    if not lock.acquire():
        log.error("another bot.meta collector is running or the lock is unavailable: %s", lock.path)
        return EXIT_LOCK_HELD
    stop = stop_event if stop_event is not None else threading.Event()
    previous: dict[int, Any] = {}
    try:
        if install_signal_handlers:
            handler = make_stop_handler(stop)
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, handler)
        observations = default_observations_path()
        log_resolved_paths(meta_lock=lock.path, meta_observations=observations)
        pipe = pipeline if pipeline is not None else MetaPipeline(cfg, observations_path=observations)
        pipe.run_forever(stop_event=stop)
        return 0
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)
        lock.release()


def _with_writer_lock(fn: Callable[[], Any]) -> tuple[Optional[Any], int]:
    lock = ProcessLock(meta_lock_path())
    if not lock.acquire():
        log.error("bot.meta collector holds %s; refusing to write observation rows", lock.path)
        return None, EXIT_LOCK_HELD
    try:
        return fn(), 0
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config()
        meta_lock_path()  # validate MEMEBOT_DATA_DIR early (must be absolute)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    command = argv[0] if argv and not argv[0].startswith("-") else "report"
    flags = [a for a in argv if a.startswith("-")]
    if command == "run":
        return run_collector(cfg)
    if command == "snapshot":
        report, code = _with_writer_lock(lambda: MetaPipeline(cfg).snapshot_universe())
    elif "--persist" in flags:
        report, code = _with_writer_lock(lambda: MetaDetector(cfg).scan_live(persist=True))
    else:
        report, code = MetaDetector(cfg).scan_live(persist=False), 0
    if code:
        return code
    if "--json" in flags:
        print(json.dumps({
            "generated_at": report.generated_at,
            "notes": report.notes,
            "window_coverage": report.window_coverage,
            "clusters": [c.to_dict() for c in report.clusters],
            "signals": [s.to_dict() for s in report.signals],
        }, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
