"""P1: ``save_state`` reports True only when the new state is durable.

Durability contract (``bot.state.save_state``): True only after the temp file
is written, flushed and fsynced, ``os.replace`` is done, and the containing
directory was opened, fsynced and closed. Any failure returns False:
before the replace the old file is intact (temp removed); after the replace
the new file is visible but either version may survive a crash, so callers
treat it as not durable (no send; booking rolled back to confirmed_unbooked).

Faults are injected at the real OS boundary inside ``bot.state`` (``os.open``
/ ``os.fsync`` / ``os.close`` / ``os.replace`` / the temp file's write) through
a proxy ``os`` module; ``_fsync_dir`` and ``save_state`` themselves run
unmodified. Everything else is the offline world of the native-SOL e2e tests:
nothing is signed or sent, no network.
"""
from __future__ import annotations

import json
import os
import types

import pytest

import bot.main as main_module
import bot.state as state_module
from bot.jupiter import SOL_MINT
from bot.portfolio import Portfolio
from bot.state import save_state
from tests.test_p1_native_sol_settlement_e2e import (  # noqa: F401  (world is a fixture)
    FULL, IN_RAW, MINT, SIG, SL_MARK, SL_PNL, assert_booked_once,
    assert_never_terminal_unbooked, boot, intents_of, lands, open_position, persisted,
    position, run, show, world,
)

REAL_OS = os


class Crash(BaseException):
    """The process dies at the injected point (not caught by save_state)."""


class FaultyOS:
    """Proxy for ``os`` inside bot.state. Each ``save_state`` call is one
    'save' (counted at ``fdopen``); ``faults`` maps save number -> step that
    fails: write, file_fsync, replace, dir_open, dir_fsync, dir_close. With
    ``crash=True`` the failing step kills the process instead of raising
    OSError (after the replace, the new file is then visible on disk)."""

    def __init__(self, faults, crash=False):
        self.faults, self.crash = dict(faults), crash
        self.save_no = 0
        self.dir_fds = set()
        self.events = []
        self.open_flags = []
        self.before = {}          # save number -> state file bytes before that save
        self.state_path = None

    def __getattr__(self, name):
        return getattr(REAL_OS, name)

    def _hit(self, step):
        self.events.append(step)
        if self.faults.get(self.save_no) == step:
            if self.crash:
                raise Crash(step)
            raise OSError(5, f"injected {step} failure")

    def fdopen(self, fd, mode="r", *a, **kw):
        self.save_no += 1
        if self.state_path and REAL_OS.path.exists(self.state_path):
            with open(self.state_path, "rb") as f:
                self.before[self.save_no] = f.read()
        f = REAL_OS.fdopen(fd, mode, *a, **kw)
        if self.faults.get(self.save_no) == "write":
            proxy = self

            class FailingWrite:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    f.close()

                def write(self, s):
                    proxy._hit("write")

                def __getattr__(self, n):
                    return getattr(f, n)
            return FailingWrite()
        return f

    def fsync(self, fd):
        self._hit("dir_fsync" if fd in self.dir_fds else "file_fsync")
        return REAL_OS.fsync(fd)

    def replace(self, src, dst):
        self._hit("replace")
        return REAL_OS.replace(src, dst)

    def open(self, path, flags, *a, **kw):
        self.open_flags.append(flags)
        self._hit("dir_open")
        fd = REAL_OS.open(path, flags, *a, **kw)
        self.dir_fds.add(fd)
        return fd

    def close(self, fd):
        is_dir = fd in self.dir_fds
        self.dir_fds.discard(fd)
        REAL_OS.close(fd)
        if is_dir:
            self._hit("dir_close")


def install(monkeypatch, faults=None, crash=False, path=None):
    fos = FaultyOS(faults or {}, crash=crash)
    fos.state_path = path
    monkeypatch.setattr(state_module, "os", fos)
    return fos


def risk(pnl=0.0):
    return types.SimpleNamespace(to_dict=lambda: {"realized_pnl_today": pnl})


def disk_pnl(path):
    with open(path) as f:
        return json.load(f)["portfolio"]["realized_pnl"]


def tmp_leftovers(d):
    return [p for p in os.listdir(d) if p.endswith(".tmp")]


# -- save_state unit: every step must succeed before True ----------------------------

def test_success_path_is_durable_in_order(monkeypatch, tmp_path):
    path = str(tmp_path / "state.json")
    fos = install(monkeypatch)
    pf = Portfolio()
    pf.realized_pnl = 1.5
    assert save_state(path, pf, risk(1.5)) is True
    assert fos.events == ["file_fsync", "replace", "dir_open", "dir_fsync", "dir_close"]
    assert fos.open_flags == [REAL_OS.O_RDONLY | getattr(REAL_OS, "O_DIRECTORY", 0)]
    assert fos.dir_fds == set()                              # directory fd closed
    assert disk_pnl(path) == 1.5
    assert tmp_leftovers(tmp_path) == []


def _old_then_fault(monkeypatch, tmp_path, step):
    path = str(tmp_path / "state.json")
    pf = Portfolio()
    assert save_state(path, pf, risk()) is True               # old: realized 0.0
    old = open(path, "rb").read()
    fos = install(monkeypatch, {1: step})
    pf.realized_pnl = 9.0
    ok = save_state(path, pf, risk(9.0))
    return path, old, fos, ok


@pytest.mark.parametrize("step", ["write", "file_fsync", "replace"])
def test_failure_before_replace_returns_false_old_file_intact_tmp_removed(monkeypatch, tmp_path, step):
    path, old, fos, ok = _old_then_fault(monkeypatch, tmp_path, step)
    assert ok is False
    assert open(path, "rb").read() == old                     # previous state untouched
    assert tmp_leftovers(tmp_path) == []                      # temp cleaned up
    assert "dir_open" not in fos.events and "dir_fsync" not in fos.events


@pytest.mark.parametrize("step", ["dir_open", "dir_fsync", "dir_close"])
def test_directory_durability_failure_after_replace_returns_false(monkeypatch, tmp_path, step, caplog):
    """The rename happened (new file visible) but it is not known durable."""
    path, old, fos, ok = _old_then_fault(monkeypatch, tmp_path, step)
    assert ok is False                                        # never reported durable
    assert disk_pnl(path) == 9.0                         # new version is what is visible
    assert open(path, "rb").read() != old
    assert tmp_leftovers(tmp_path) == []                      # nothing left behind
    assert fos.dir_fds == set()                               # dir fd never leaked
    assert "old or the new state" in caplog.text


def test_failure_creating_temp_file_returns_false(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")                                   # a file where a dir is needed
    assert save_state(str(blocker / "state.json"), Portfolio(), risk()) is False


def test_serialization_failure_returns_false_without_touching_disk(tmp_path):
    path = tmp_path / "state.json"
    bad = types.SimpleNamespace(to_dict=lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert save_state(str(path), Portfolio(), bad) is False
    assert not path.exists() and tmp_leftovers(tmp_path) == []


def test_fsync_dir_raises_on_open_and_fsync_failure(monkeypatch, tmp_path):
    install(monkeypatch, {0: "dir_open"})                     # save_no stays 0: direct call
    with pytest.raises(OSError):
        state_module._fsync_dir(str(tmp_path))
    fos = install(monkeypatch, {0: "dir_fsync"})
    with pytest.raises(OSError):
        state_module._fsync_dir(str(tmp_path))
    assert fos.dir_fds == set()
    install(monkeypatch)
    assert state_module._fsync_dir(str(tmp_path)) is None


# -- unsupported platforms: never reported durable; armed startup refuses --------------

def test_unsupported_platform_writes_best_effort_but_is_never_durable(monkeypatch, tmp_path):
    monkeypatch.setattr(state_module, "DURABLE_DIR_FSYNC_SUPPORTED", False)
    path = str(tmp_path / "state.json")
    pf = Portfolio()
    pf.realized_pnl = 2.0
    assert save_state(path, pf, risk()) is False
    assert disk_pnl(path) == 2.0            # paper: best effort persisted


def test_unsupported_platform_armed_startup_refuses_before_any_rpc(world, monkeypatch):
    bot = boot(world)
    open_position(bot, world)
    world.marks[MINT] = SL_MARK
    world.rpc.clear()
    monkeypatch.setattr(state_module, "DURABLE_DIR_FSYNC_SUPPORTED", False)
    assert run(bot, 2) == 0
    assert world.rpc == [] and bot.jup.quotes == [] and world.signs == [] and world.sends == []
    assert bot.entry_scans == []


def test_supported_platform_is_posix():
    assert state_module.durable_saves_supported() is (os.name == "posix")


# -- pre-send: a non-durable intent means zero quote / sign / send ---------------------

@pytest.mark.parametrize("step", ["write", "file_fsync", "replace", "dir_open", "dir_fsync", "dir_close"])
def test_presend_not_durable_sends_nothing_then_retries_once(world, monkeypatch, step):
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL)
    world.marks[MINT] = SL_MARK
    install(monkeypatch, {1: step}, path=world.state)         # the pre-send intent save
    bot._tick()
    assert bot.jup.quotes == [] and world.signs == [] and world.sends == []
    (intent,) = bot.portfolio.pending_intents
    assert intent["status"] == "blocked" and "nothing sent" in intent["error"]
    assert intent["raw_amount"] == IN_RAW
    assert bot.portfolio.positions[MINT].raw_tokens == IN_RAW
    bot._tick()                                               # durable now: sold once
    assert bot.jup.quotes == [(MINT, SOL_MINT, IN_RAW)]
    assert [i["status"] for i in intents_of(persisted(world))] == ["failed", "settled"]
    assert_booked_once(world)
    assert len(bot.notifier.sells) == 1


@pytest.mark.parametrize("visible", ["old", "new"])
def test_presend_crash_after_replace_restarts_from_either_version_once(world, monkeypatch, visible):
    """Dies inside the dir fsync of the pre-send save: the new file (pending
    intent, no signature) is visible; after power loss the old one (no intent)
    may be. Nothing was sent; a restart from either sells and books once."""
    bot = boot(world)
    open_position(bot, world)
    world.sig_queue = [SIG]                                   # a second send would fail
    world.on_send[SIG] = lands(MINT, FULL)
    world.marks[MINT] = SL_MARK
    fos = install(monkeypatch, {1: "dir_fsync"}, crash=True, path=world.state)
    with pytest.raises(Crash):
        bot._tick()
    assert world.signs == [] and world.sends == [] and bot.jup.quotes == []
    (intent,) = intents_of(persisted(world))                  # new version is visible
    assert (intent["status"], intent["raw_amount"], intent["tx_signature"]) == ("pending", IN_RAW, None)
    if visible == "old":
        with open(world.state, "wb") as f:
            f.write(fos.before[1])
        assert intents_of(persisted(world)) == []
    install(monkeypatch)                                      # storage healthy again
    world.now += main_module.SELL_INTENT_EXPIRY_SECONDS + 1
    run(boot(world), 4)
    assert world.sends == [SIG]
    assert_never_terminal_unbooked(world)
    assert_booked_once(world)


# -- booking: non-durable booking stays confirmed_unbooked, books exactly once ---------

BOOKING_SAVE = {"on_time": 4, "delayed": 2}   # [pending, signature,] confirmed_unbooked, booking


def _to_booking_tick(world, bot, path):
    open_position(bot, world)
    world.sig_queue = [SIG]
    world.on_send[SIG] = lands(MINT, FULL, visible=path == "on_time")
    world.marks[MINT] = SL_MARK
    if path == "delayed":
        bot._tick()
        show(world)


@pytest.mark.parametrize("path", ["on_time", "delayed"])
@pytest.mark.parametrize("step", ["file_fsync", "replace", "dir_open", "dir_fsync"])
def test_booking_not_durable_rolls_back_then_books_once_in_process(world, monkeypatch, path, step):
    bot = boot(world)
    _to_booking_tick(world, bot, path)
    install(monkeypatch, {BOOKING_SAVE[path]: step}, path=world.state)
    bot._tick()
    (intent,) = bot.portfolio.pending_intents                 # memory rolled back exactly
    assert intent["status"] == "confirmed_unbooked" and "not durable" in intent["resolution"]
    assert "booked_at" not in intent and intent["raw_amount"] == IN_RAW
    pos = bot.portfolio.positions[MINT]
    assert (pos.raw_tokens, pos.tokens) == (IN_RAW, FULL)
    assert bot.portfolio.realized_pnl == 0.0 and bot.risk.realized_pnl_today == 0.0
    assert bot.notifier.sells == []
    state = persisted(world)
    if step in ("dir_open", "dir_fsync"):
        # after the replace: the booked version is already visible on disk
        assert [i["status"] for i in intents_of(state)] == ["settled"]
        assert state["portfolio"]["realized_pnl"] == pytest.approx(SL_PNL)
    else:
        assert [i["status"] for i in intents_of(state)] == ["confirmed_unbooked"]
        assert state["portfolio"]["realized_pnl"] == 0.0
    assert_never_terminal_unbooked(world)

    for _ in range(3):
        bot._tick()      # memory books again; its save replaces the disk version
    assert len(bot.notifier.sells) == 1
    assert bot.portfolio.realized_pnl == pytest.approx(SL_PNL)
    assert_booked_once(world)                                 # booked once, not twice
    bot2 = boot(world)
    run(bot2, 2)
    assert bot2.notifier.sells == []
    assert_booked_once(world)


@pytest.mark.parametrize("path", ["on_time", "delayed"])
@pytest.mark.parametrize("visible", ["old", "new"])
def test_booking_crash_after_replace_restarts_from_either_version_once(world, monkeypatch, path, visible):
    """Dies inside the dir fsync of the booking save. Disk holds either the
    new version (settled + booked) or, after power loss, the old one
    (confirmed_unbooked evidence). Restart books exactly once either way."""
    bot = boot(world)
    _to_booking_tick(world, bot, path)
    n = BOOKING_SAVE[path]
    fos = install(monkeypatch, {n: "dir_fsync"}, crash=True, path=world.state)
    with pytest.raises(Crash):
        bot._tick()
    assert world.sends == [SIG]
    assert_booked_once(world)                                 # new version visible
    if visible == "old":
        with open(world.state, "wb") as f:
            f.write(fos.before[n])
        state = persisted(world)
        assert [i["status"] for i in intents_of(state)] == ["confirmed_unbooked"]
        assert state["portfolio"]["realized_pnl"] == 0.0
        assert position(state)["raw_tokens"] == IN_RAW
    assert_never_terminal_unbooked(world)
    install(monkeypatch)
    world.rpc.clear()
    bot2 = boot(world)
    run(bot2, 3)
    assert "getTransaction" not in world.rpc                  # booked from stored evidence
    assert len(bot2.notifier.sells) == (1 if visible == "old" else 0)
    assert_booked_once(world)
    bot3 = boot(world)
    run(bot3, 2)
    assert bot3.notifier.sells == []
    assert_booked_once(world)
