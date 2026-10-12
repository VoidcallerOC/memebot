"""Persistence — save/restore open positions, realized PnL, and the daily
circuit-breaker state so the bot survives a restart without losing track of
live positions.

Writes are atomic (temp file + fsync + os.replace + directory fsync) so a
crash mid-write can't corrupt the state file and orphan real on-chain
positions; ``save_state`` reports whether the write is durable (see its
contract).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from .portfolio import Portfolio
from .risk import RiskManager

log = logging.getLogger(__name__)

STATE_VERSION = 1


# Durable saves need a directory fsync after the rename. That is defined on
# POSIX (Linux, macOS). Elsewhere (e.g. Windows: directories cannot be opened
# and fsynced) a save is written best effort but never reported durable, and
# armed (live) startup refuses (see ``TradingBot.run``); paper mode, which
# ignores the result, keeps working.
DURABLE_DIR_FSYNC_SUPPORTED = os.name == "posix"


def durable_saves_supported() -> bool:
    return DURABLE_DIR_FSYNC_SUPPORTED


def save_state(path: str, portfolio: Portfolio, risk: RiskManager) -> bool:
    """Atomically write state; True ONLY when the new state is durable.

    Durability contract. Returns True only after, in order: the temp file is
    fully written, flushed and fsynced; ``os.replace`` put it at ``path``;
    and the containing directory was opened and fsynced (and closed). Never
    raises; every failure is logged and returns False:

    * failure before ``os.replace`` (serialize, mkstemp, write, flush, file
      fsync, replace): ``path`` still holds the previous state; the temp file
      is removed (best effort);
    * failure after ``os.replace`` (directory open / fsync / close): the NEW
      state is visible now, but after a crash or power loss the disk may hold
      either the old or the new version. Durability is uncertain, so the
      caller must treat the write as not durable (no send; roll back an
      in-memory booking). Recovery is correct from either version;
    * platforms without directory fsync (non-POSIX): written best effort,
      reported not durable.
    """
    try:
        data = {
            "version": STATE_VERSION,
            "portfolio": portfolio.to_dict(),
            "risk": risk.to_dict(),
        }
    except Exception as exc:
        log.error("failed to serialize state for %s: %s", path, exc)
        return False
    tmp = None
    replaced = False
    try:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            f = os.fdopen(fd, "w")
        except Exception:
            os.close(fd)
            raise
        with f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # data on disk before it becomes visible
        os.replace(tmp, path)  # atomic on POSIX and Windows
        tmp = None
        replaced = True
        if not durable_saves_supported():
            log.error(
                "state written to %s but NOT durable: directory fsync unsupported on "
                "this platform", path,
            )
            return False
        _fsync_dir(directory)  # make the rename itself durable; raises on failure
        return True
    except Exception as exc:
        if replaced:
            log.error(
                "state write to %s NOT durable: replaced but directory fsync failed (%s); "
                "the disk may hold the old or the new state", path, exc,
            )
        else:
            log.error("failed to save state to %s: %s", path, exc)
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def _fsync_dir(directory: str) -> None:
    """fsync a directory so a rename inside it survives power loss (POSIX).

    Raises OSError if the directory cannot be opened, fsynced or closed.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    dfd = os.open(directory, flags)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def load_state(path: str, portfolio: Portfolio, risk: RiskManager) -> bool:
    """Restore state in place. Returns True if a state file was loaded."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        log.error("failed to read state from %s: %s — starting fresh", path, exc)
        return False

    if data.get("version") != STATE_VERSION:
        log.warning("state file version mismatch (%s); ignoring", data.get("version"))
        return False

    portfolio.load_dict(data.get("portfolio", {}))
    risk.load_dict(data.get("risk", {}))
    log.info("Restored %d open position(s); realized PnL $%.2f; daily PnL $%.2f",
             portfolio.open_count, portfolio.realized_pnl, risk.realized_pnl_today)
    return True
