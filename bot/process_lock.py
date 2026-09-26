"""Single-instance advisory lock for the memebot runtime.

The lock uses the host OS's advisory file-locking primitive.  The lock file
contains no credentials or process metadata; its existence alone is not used
to infer liveness, so stale files do not block startup.  A lock-access error
fails closed.
"""
from __future__ import annotations

import fcntl
import os
from typing import Optional


class ProcessLock:
    """Hold an exclusive, non-blocking lock for one memebot process."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._handle: Optional[object] = None

    def acquire(self) -> bool:
        """Acquire the lock, returning False on conflict or lock failure."""
        if self._handle is not None:
            return True
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            handle = open(self.path, "a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
            self._handle = handle
            return True
        except (OSError, ValueError):
            return False

    def release(self) -> None:
        """Release the lock and close the descriptor; safe to call repeatedly."""
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass

    def __enter__(self) -> "ProcessLock":
        if not self.acquire():
            raise RuntimeError("process lock unavailable")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def process_lock_available(path: str) -> bool:
    """Return whether a memebot process lock can be acquired right now.

    A return value of False means either another instance holds the lock or the
    lock state could not be established.  Both cases must fail closed.
    """
    lock = ProcessLock(path)
    if not lock.acquire():
        return False
    lock.release()
    return True
