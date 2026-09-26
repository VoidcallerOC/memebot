"""Cross-platform single-instance advisory lock for the memebot runtime.

The lock file contains no credentials or process metadata. Its existence alone
is not used to infer liveness, so stale files do not block startup. Lock-access
errors fail closed.
"""
from __future__ import annotations

import os
from typing import Optional

_IS_WINDOWS = os.name == "nt"
_LOCK_BYTES = 1


def _ensure_lock_byte(handle: object) -> None:
    """Ensure the lock file has one byte for Windows region locking."""
    handle.seek(0, os.SEEK_END)  # type: ignore[attr-defined]
    if handle.tell() == 0:  # type: ignore[attr-defined]
        handle.write(b"\0")  # type: ignore[attr-defined]
        handle.flush()  # type: ignore[attr-defined]
    handle.seek(0)  # type: ignore[attr-defined]


def _acquire_native(handle: object) -> None:
    if _IS_WINDOWS:
        # Import only on Windows: importing fcntl on Windows breaks test
        # collection and normal module import.
        import msvcrt

        _ensure_lock_byte(handle)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_BYTES)  # type: ignore[attr-defined]
        return

    # Import only on POSIX hosts; Linux remains fully supported without a
    # third-party dependency.
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]


def _release_native(handle: object) -> None:
    if _IS_WINDOWS:
        import msvcrt

        handle.seek(0)  # type: ignore[attr-defined]
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_BYTES)  # type: ignore[attr-defined]
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


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
            handle = open(self.path, "a+b")
            try:
                _acquire_native(handle)
            except (ImportError, OSError, ValueError):
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
            _release_native(handle)
        except (ImportError, OSError, ValueError):
            pass
        try:
            handle.close()  # type: ignore[attr-defined]
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

    False means either another instance holds the lock or lock state could not
    be established. Both cases fail closed.
    """
    lock = ProcessLock(path)
    if not lock.acquire():
        return False
    lock.release()
    return True
