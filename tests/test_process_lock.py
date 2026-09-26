"""Cross-platform process-lock tests without wallet or transaction dependencies."""
from __future__ import annotations

import builtins
import os
from pathlib import Path
import subprocess
import sys

import pytest

import bot.process_lock as process_lock
from bot.process_lock import ProcessLock, process_lock_available


POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="requires POSIX fcntl")
WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="requires native Windows msvcrt")


class FakeMSVCRT:
    LK_NBLCK = 1
    LK_UNLCK = 2
    locked_inodes: set[int] = set()

    @classmethod
    def locking(cls, fd, mode, size):
        assert size == 1
        inode = os.fstat(fd).st_ino
        if mode == cls.LK_NBLCK:
            if inode in cls.locked_inodes:
                raise OSError("simulated Windows lock conflict")
            cls.locked_inodes.add(inode)
        elif mode == cls.LK_UNLCK:
            cls.locked_inodes.discard(inode)
        else:
            raise AssertionError(f"unexpected lock mode: {mode}")


@pytest.fixture
def fake_windows(monkeypatch):
    FakeMSVCRT.locked_inodes.clear()
    monkeypatch.setattr(process_lock, "_IS_WINDOWS", True)
    monkeypatch.setitem(sys.modules, "msvcrt", FakeMSVCRT)
    yield
    FakeMSVCRT.locked_inodes.clear()


@pytest.fixture
def native_linux(monkeypatch):
    monkeypatch.setattr(process_lock, "_IS_WINDOWS", False)


def test_windows_lock_acquisition_and_release(tmp_path, fake_windows):
    lock = ProcessLock(str(tmp_path / "windows.lock"))
    assert lock.acquire()
    lock.release()
    assert process_lock_available(str(tmp_path / "windows.lock"))


def test_windows_conflicting_process_fails(tmp_path, fake_windows):
    path = str(tmp_path / "windows.lock")
    holder = ProcessLock(path)
    contender = ProcessLock(path)
    assert holder.acquire()
    try:
        assert not contender.acquire()
    finally:
        holder.release()
        contender.release()
    assert process_lock_available(path)


def test_windows_import_path_does_not_import_fcntl(tmp_path, fake_windows, monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "fcntl":
            raise AssertionError("Windows backend must not import fcntl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    lock = ProcessLock(str(tmp_path / "windows-import.lock"))
    assert lock.acquire()
    lock.release()


@POSIX_ONLY
def test_linux_lock_acquisition_and_release(tmp_path, native_linux):
    lock = ProcessLock(str(tmp_path / "linux.lock"))
    assert lock.acquire()
    lock.release()
    lock.release()
    assert process_lock_available(str(tmp_path / "linux.lock"))


@POSIX_ONLY
def test_linux_conflicting_process_fails(tmp_path, native_linux):
    path = str(tmp_path / "linux.lock")
    holder = ProcessLock(path)
    contender = ProcessLock(path)
    assert holder.acquire()
    try:
        assert not contender.acquire()
    finally:
        holder.release()
        contender.release()


@POSIX_ONLY
def test_stale_lock_file_does_not_block(tmp_path, native_linux):
    path = tmp_path / "stale.lock"
    path.write_text("stale state")
    assert process_lock_available(str(path))


@POSIX_ONLY
def test_lock_access_failure_fails_closed(tmp_path, native_linux):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocker")
    assert not process_lock_available(str(blocker / "lock"))


@POSIX_ONLY
def test_release_is_idempotent_and_allows_reacquisition(tmp_path, native_linux):
    path = str(tmp_path / "release.lock")
    lock = ProcessLock(path)
    lock.release()
    assert lock.acquire()
    lock.release()
    lock.release()
    replacement = ProcessLock(path)
    assert replacement.acquire()
    replacement.release()


@WINDOWS_ONLY
def test_native_windows_lock_acquisition_and_release(tmp_path):
    path = str(tmp_path / "windows-native.lock")
    lock = ProcessLock(path)
    assert lock.acquire()
    lock.release()
    lock.release()
    assert process_lock_available(path)


@WINDOWS_ONLY
def test_native_windows_conflicting_process_fails(tmp_path):
    path = str(tmp_path / "windows-native-conflict.lock")
    holder = ProcessLock(path)
    contender = ProcessLock(path)
    assert holder.acquire()
    try:
        assert not contender.acquire()
    finally:
        holder.release()
        contender.release()


@WINDOWS_ONLY
def test_native_windows_stale_lock_file_does_not_block(tmp_path):
    path = tmp_path / "windows-native-stale.lock"
    path.write_text("stale state")
    assert process_lock_available(str(path))


@WINDOWS_ONLY
def test_native_windows_lock_access_failure_fails_closed(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocker")
    assert not process_lock_available(str(blocker / "lock"))


@WINDOWS_ONLY
def test_native_windows_release_is_idempotent_and_allows_reacquisition(tmp_path):
    path = str(tmp_path / "windows-native-release.lock")
    lock = ProcessLock(path)
    lock.release()
    assert lock.acquire()
    lock.release()
    lock.release()
    replacement = ProcessLock(path)
    assert replacement.acquire()
    replacement.release()


@WINDOWS_ONLY
def test_native_windows_second_process_contention(tmp_path):
    """Verify contention across real Windows processes, not just handles."""
    path = str(tmp_path / "windows-native-process.lock")
    project_root = Path(__file__).resolve().parents[1]
    child_code = (
        "import sys, time; "
        "from bot.process_lock import ProcessLock; "
        "lock = ProcessLock(sys.argv[1]); "
        "raise SystemExit(2) if not lock.acquire() else print('READY', flush=True); "
        "time.sleep(15)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, path],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        assert not process_lock_available(path)
    finally:
        child.terminate()
        child.wait(timeout=10)
