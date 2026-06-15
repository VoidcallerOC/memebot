"""Persistence — save/restore open positions, realized PnL, and the daily
circuit-breaker state so the bot survives a restart without losing track of
live positions.

Writes are atomic (temp file + os.replace) so a crash mid-write can't corrupt
the state file and orphan real on-chain positions.
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


def save_state(path: str, portfolio: Portfolio, risk: RiskManager) -> None:
    data = {
        "version": STATE_VERSION,
        "portfolio": portfolio.to_dict(),
        "risk": risk.to_dict(),
    }
    try:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)  # atomic on POSIX and Windows
    except Exception as exc:
        log.error("failed to save state to %s: %s", path, exc)


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
