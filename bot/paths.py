"""Persistent data-path resolution (MEMEBOT_DATA_DIR).

Every relative persistence path (state, process locks, META observation JSONL,
decision shadow journal) resolves under one data directory so that processes
started from different working directories share the same files and the same
single-instance locks.

  * ``MEMEBOT_DATA_DIR`` set   -> must be an absolute path; used as-is.
  * ``MEMEBOT_DATA_DIR`` unset -> the repository root (the directory that
    contains the ``bot`` package), i.e. the historical layout when the bot is
    launched from the repo root, but no longer dependent on the CWD.

Absolute paths given via STATE_FILE / PROCESS_LOCK_FILE / ... are never
rewritten.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DATA_DIR_ENV = "MEMEBOT_DATA_DIR"
REPO_ROOT = Path(__file__).resolve().parent.parent

META_LOCK_ENV = "META_PROCESS_LOCK_FILE"
DEFAULT_META_LOCK_FILE = "meta.lock"


def data_dir() -> Path:
    raw = (os.getenv(DATA_DIR_ENV) or "").strip()
    if not raw:
        return REPO_ROOT
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{DATA_DIR_ENV} must be an absolute path (got a relative path)")
    return path


def resolve_data_path(path: str) -> str:
    """Return ``path`` unchanged if absolute, else anchored under data_dir()."""
    p = Path(os.path.expanduser(str(path)))
    if p.is_absolute():
        return str(p)
    return str(data_dir() / p)


def meta_lock_path() -> str:
    raw = (os.getenv(META_LOCK_ENV) or "").strip() or DEFAULT_META_LOCK_FILE
    return resolve_data_path(raw)


def with_resolved_data_paths(cfg: Any) -> Any:
    """Return a copy of a (frozen) Config with absolute persistence paths.

    Only persistence paths are touched: state_file, process_lock_file,
    decision_shadow_file. Trading / risk / live settings are copied verbatim.
    """
    changes = {}
    for name in ("state_file", "process_lock_file", "decision_shadow_file"):
        value = getattr(cfg, name, None)
        if value:
            changes[name] = resolve_data_path(value)
    return dataclasses.replace(cfg, **changes) if changes else cfg


def log_resolved_paths(**paths: str) -> None:
    log.info("data dir: %s", data_dir())
    for name, value in paths.items():
        log.info("  %s = %s", name, value)
