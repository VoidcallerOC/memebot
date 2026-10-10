"""PR1 data-integrity tests: incremental history refresh, meta single-instance
lock + clean SIGTERM shutdown, MEMEBOT_DATA_DIR resolution, fsync durability,
and `report` not persisting rows.

Synthetic fixtures under tmp_path only. No network: the real collector loop is
never started (a fake tick replaces snapshot_universe), and no real state or
observation files are read or written.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import bot.main as main_module
import bot.meta.__main__ as meta_cli
import bot.meta.store as store_module
from bot.config import Config
from bot.decision.shadow import ShadowJournal, ShadowRecord
from bot.meta.detector import MetaDetector
from bot.meta.history import ObservationHistory
from bot.meta.model import MarketWindow, TokenSnapshot
from bot.meta.pipeline import MetaPipeline
from bot.paths import REPO_ROOT, data_dir, meta_lock_path, resolve_data_path, with_resolved_data_paths
from bot.portfolio import Portfolio
from bot.process_lock import ProcessLock
from bot.risk import RiskManager
from bot.state import load_state, save_state

REPO = Path(__file__).resolve().parent.parent


class _NoNetworkSession:
    def __getattr__(self, name):
        raise AssertionError(f"network access attempted via session.{name}")


def _row(i: int, mint: str = "M") -> str:
    return json.dumps({"kind": "market_snapshot", "recorded_at": float(i), "observed_at": float(i),
                       "mint": mint, "market": {"1h": {"volume_usd": 1.0}}})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("MEMEBOT_DATA_DIR", "META_OBSERVATIONS_FILE", "META_PROCESS_LOCK_FILE",
                 "DECISION_SHADOW_FILE", "STATE_FILE", "PROCESS_LOCK_FILE"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# 1. Incremental refresh
# ---------------------------------------------------------------------------

def test_history_refresh_reads_only_new_bytes(tmp_path, monkeypatch):
    path = tmp_path / "obs.jsonl"
    path.write_text(_row(1) + "\n" + _row(2) + "\n", encoding="utf-8")
    hist = ObservationHistory(str(path))
    assert len(hist.load()) == 2
    offsets = []
    real = store_module.read_observations_from
    monkeypatch.setattr("bot.meta.history.read_observations_from",
                        lambda p, off: (offsets.append(off), real(p, off))[1])
    size_before = path.stat().st_size
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_row(3) + "\n")
    assert [r["recorded_at"] for r in hist.refresh()] == [1.0, 2.0, 3.0]
    assert offsets == [size_before]          # incremental, not a full re-read
    assert hist.refresh() is hist.load()     # nothing new -> no read at all
    assert offsets == [size_before]


def test_history_refresh_incremental_partial_line_and_truncation(tmp_path):
    path = tmp_path / "obs.jsonl"
    path.write_text(_row(1) + "\n", encoding="utf-8")
    hist = ObservationHistory(str(path))
    assert len(hist.load()) == 1

    full = _row(2) + "\n"
    torn, rest = full[:25], full[25:]
    with path.open("a", encoding="utf-8") as fh:
        fh.write(torn)                       # writer mid-append
    assert len(hist.refresh()) == 1          # torn tail not parsed
    with path.open("a", encoding="utf-8") as fh:
        fh.write(rest)                       # line completed
    rows = hist.refresh()
    assert [r["recorded_at"] for r in rows] == [1.0, 2.0]
    assert len(hist.refresh()) == 2          # no double counting

    # Truncation (size < offset) -> full reload.
    path.write_text(_row(9) + "\n", encoding="utf-8")
    assert [r["recorded_at"] for r in hist.refresh()] == [9.0]

    # Copy-truncate then regrowth past the old offset -> head mismatch -> reload.
    path.write_text("".join(_row(100 + i) + "\n" for i in range(5)), encoding="utf-8")
    assert [r["recorded_at"] for r in hist.refresh()] == [100.0 + i for i in range(5)]

    # Rotation (new inode) -> full reload; row count equals a fresh full read.
    rotated = tmp_path / "obs.jsonl.1"
    os.replace(path, rotated)
    path.write_text(_row(7) + "\n" + _row(8) + "\n", encoding="utf-8")
    assert [r["recorded_at"] for r in hist.refresh()] == [7.0, 8.0]
    assert len(hist.refresh()) == len(ObservationHistory(str(path)).load())

    # File removed -> empty, then recreated -> picked up.
    path.unlink()
    assert hist.refresh() == []
    path.write_text(_row(42) + "\n", encoding="utf-8")
    assert [r["recorded_at"] for r in hist.refresh()] == [42.0]


def test_history_append_then_refresh_does_not_duplicate(tmp_path):
    path = tmp_path / "obs.jsonl"
    hist = ObservationHistory(str(path))
    hist.load()
    hist.append("signal", {"token": "A"}, recorded_at=1.0)
    hist.append("signal", {"token": "B"}, recorded_at=2.0)
    assert [r["token"] for r in hist.refresh()] == ["A", "B"]
    assert len(hist.refresh()) == 2
    assert len(hist.load(force=True)) == 2


# ---------------------------------------------------------------------------
# 2. bot.meta run: single instance + clean shutdown
# ---------------------------------------------------------------------------

class _FakePipeline:
    def __init__(self):
        self.calls = 0

    def run_forever(self, cycles=None, stop_event=None):
        self.calls += 1


def test_meta_run_second_instance_refused(tmp_path):
    lock_path = str(tmp_path / "meta.lock")
    holder = ProcessLock(lock_path)
    assert holder.acquire()
    try:
        fake = _FakePipeline()
        rc = meta_cli.run_collector(Config(), pipeline=fake, lock_path=lock_path,
                                    install_signal_handlers=False)
        assert rc == 3
        assert fake.calls == 0
    finally:
        holder.release()
    fake = _FakePipeline()
    assert meta_cli.run_collector(Config(), pipeline=fake, lock_path=lock_path,
                                  install_signal_handlers=False) == 0
    assert fake.calls == 1
    assert ProcessLock(lock_path).acquire()  # released after a normal exit


def test_meta_run_second_instance_refused_across_processes(tmp_path):
    lock_path = str(tmp_path / "meta.lock")
    holder = ProcessLock(lock_path)
    assert holder.acquire()
    try:
        script = textwrap.dedent(f"""
            import bot.meta.__main__ as m
            from bot.config import Config
            class P:
                def run_forever(self, **kw):
                    raise SystemExit("pipeline must not run")
            raise SystemExit(m.run_collector(Config(), pipeline=P(), lock_path={lock_path!r},
                                             install_signal_handlers=False))
        """)
        proc = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=_child_env(tmp_path),
                              capture_output=True, text=True, timeout=30)
        assert proc.returncode == 3, proc.stderr
    finally:
        holder.release()


def test_meta_stop_handler_finishes_current_tick(tmp_path):
    obs = tmp_path / "obs.jsonl"
    stop = threading.Event()
    handler = meta_cli.make_stop_handler(stop)
    pipe = MetaPipeline(Config(), session=_NoNetworkSession(), observations_path=str(obs))
    ticks = []

    def fake_tick(now=None, fetch_mints=None):
        handler(signal.SIGTERM, None)        # signal arrives mid-tick
        store_module.append_observation(str(obs), {"kind": "theme_counts", "recorded_at": now, "counts": {}})
        ticks.append(now)

    pipe.snapshot_universe = fake_tick
    started = time.monotonic()
    pipe.run_forever(stop_event=stop)
    assert len(ticks) == 1                   # tick completed, no further ticks
    assert time.monotonic() - started < 5    # did not sleep the 300 s interval
    assert len(obs.read_text(encoding="utf-8").splitlines()) == 1


def _child_env(tmp_path: Path) -> dict:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(REPO),
        "HOME": str(tmp_path),
        "MEMEBOT_DATA_DIR": str(tmp_path),
        "LIVE_TRADING": "false",
        "WALLET_PRIVATE_KEY": "",
    }


def test_meta_run_sigterm_subprocess_exits_cleanly(tmp_path):
    """Harmless test entrypoint: real run_collector + real signal handlers, but
    the tick is a synthetic local append (no network, no real data)."""
    script = textwrap.dedent("""
        import logging, sys
        logging.basicConfig(level=logging.INFO)
        import bot.meta.__main__ as m
        from bot.meta.model import MetaReport
        from bot.config import Config
        from bot.meta.history import default_observations_path
        from bot.meta.pipeline import MetaPipeline
        from bot.meta.store import append_observation

        class NoNet:
            def __getattr__(self, name):
                raise AssertionError("network")

        obs = default_observations_path()
        pipe = MetaPipeline(Config(), session=NoNet(), observations_path=obs)

        def tick(now=None, fetch_mints=None):
            for i in range(51):
                append_observation(obs, {"kind": "market_snapshot", "recorded_at": now, "i": i})
            print("TICK", flush=True)
            return MetaReport(generated_at=now)

        pipe.snapshot_universe = tick
        sys.exit(m.run_collector(Config(), pipeline=pipe))
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=tmp_path, env=_child_env(tmp_path),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "TICK"
        assert not ProcessLock(str(tmp_path / "meta.lock")).acquire()  # held while running
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    stderr = proc.stderr.read()
    assert rc == 0, stderr
    assert "stopping after current tick" in stderr
    assert "tick failed" not in stderr
    lines = (tmp_path / "meta_observations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 51 and all(json.loads(line)["kind"] == "market_snapshot" for line in lines)
    lock = ProcessLock(str(tmp_path / "meta.lock"))
    assert lock.acquire()                    # released on shutdown
    lock.release()


# ---------------------------------------------------------------------------
# 3. MEMEBOT_DATA_DIR
# ---------------------------------------------------------------------------

def test_data_dir_defaults_to_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert data_dir() == REPO_ROOT == REPO
    assert resolve_data_path("state.json") == str(REPO / "state.json")
    assert meta_lock_path() == str(REPO / "meta.lock")
    assert ObservationHistory().path == str(REPO / "meta_observations.jsonl")
    assert ShadowJournal().path == str(REPO / "decision_shadow.jsonl")


def test_data_dir_rejects_relative(monkeypatch):
    monkeypatch.setenv("MEMEBOT_DATA_DIR", "relative/dir")
    with pytest.raises(ValueError):
        data_dir()


def test_data_dir_resolves_absolute_paths(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("MEMEBOT_DATA_DIR", str(data))
    monkeypatch.setenv("META_OBSERVATIONS_FILE", "obs/meta.jsonl")
    monkeypatch.setenv("DECISION_SHADOW_FILE", "shadow.jsonl")
    absolute_state = str(tmp_path / "elsewhere" / "state.json")
    cfg = Config(state_file=absolute_state, process_lock_file=".memebot.lock",
                 decision_shadow_file="decision_shadow.jsonl")
    locks = []
    for cwd in (tmp_path / "a", tmp_path / "b"):
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        resolved = with_resolved_data_paths(cfg)
        assert resolved.state_file == absolute_state          # absolute untouched
        assert resolved.process_lock_file == str(data / ".memebot.lock")
        assert resolved.decision_shadow_file == str(data / "decision_shadow.jsonl")
        assert resolved.live_trading is False and resolved.wallet_private_key == ""
        assert ObservationHistory().path == str(data / "obs" / "meta.jsonl")
        assert ShadowJournal().path == str(data / "shadow.jsonl")
        assert meta_lock_path() == str(data / "meta.lock")
        locks.append(resolved.process_lock_file)
    assert locks[0] == locks[1]
    first = ProcessLock(locks[0])
    assert first.acquire()
    try:
        assert not ProcessLock(locks[1]).acquire()            # cwd no longer splits the lock
    finally:
        first.release()


def test_bot_main_uses_absolute_lock_and_refuses_relative_data_dir(tmp_path, monkeypatch):
    seen = []

    class FakeLock:
        def __init__(self, path):
            seen.append(path)

        def acquire(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(main_module, "ProcessLock", FakeLock)
    monkeypatch.setattr(main_module, "load_config", lambda: Config(process_lock_file=".memebot.lock"))
    monkeypatch.setenv("MEMEBOT_DATA_DIR", str(tmp_path))
    assert main_module.main() == 3
    assert seen == [str(tmp_path / ".memebot.lock")]
    monkeypatch.setenv("MEMEBOT_DATA_DIR", "not/absolute")
    seen.clear()
    assert main_module.main() == 2
    assert seen == []


# ---------------------------------------------------------------------------
# 4. Durable writes
# ---------------------------------------------------------------------------

@pytest.fixture
def fsync_calls(monkeypatch):
    calls = []
    real = os.fsync

    def spy(fd):
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return calls


def test_append_observation_fsyncs(tmp_path, fsync_calls):
    store_module.append_observation(str(tmp_path / "o.jsonl"), {"kind": "x"})
    assert len(fsync_calls) == 1


def test_shadow_journal_append_fsyncs(tmp_path, fsync_calls):
    rec = ShadowRecord(
        kind="shadow_decision", timestamp=1.0, mint="M", symbol="S", model_version="t",
        feature_schema_version="t", decision_engine_version="t", risk_engine_version="t",
        action="REJECT", confidence=0.0, score=0.0, probability=0.0, risk_status="BLOCK",
        risk_reasons=[], features=[], missing_feature_frac=1.0, detection_ts=1.0,
        detection_latency_seconds=0.0,
    )
    ShadowJournal(str(tmp_path / "s.jsonl")).append(rec)
    assert len(fsync_calls) == 1
    assert len((tmp_path / "s.jsonl").read_text().splitlines()) == 1


def test_save_state_fsyncs_file_and_dir(tmp_path, fsync_calls):
    cfg = Config()
    path = tmp_path / "state.json"
    save_state(str(path), Portfolio(), RiskManager(cfg))
    assert len(fsync_calls) == 2             # temp file, then parent directory
    assert load_state(str(path), Portfolio(), RiskManager(cfg)) is True
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_save_state_removes_tmp_on_failure(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(os, "replace", boom)
    save_state(str(tmp_path / "state.json"), Portfolio(), RiskManager(Config()))
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 5. report does not persist
# ---------------------------------------------------------------------------

def _synthetic_snapshots(now: float) -> list[TokenSnapshot]:
    snap = TokenSnapshot(
        mint="RptMint11111111111111111111111111111111", symbol="RPT", observed_at=now,
        price_usd=0.01, liquidity_usd=50_000.0,
        market={"1h": MarketWindow(volume_usd=5_000, tx_buys=10, tx_sells=5, source="fixture")},
        source="fixture",
    )
    snap.top_holder_pct = 10.0
    snap.top10_holder_pct = 30.0
    return [snap]


@pytest.fixture
def offline_detector(monkeypatch):
    monkeypatch.setattr(MetaDetector, "collect_live",
                        lambda self, now=None: _synthetic_snapshots(now or 1_700_000_000.0))


def test_report_does_not_write_rows(tmp_path, monkeypatch, offline_detector, capsys):
    obs = tmp_path / "obs.jsonl"
    obs.write_text(_row(1) + "\n", encoding="utf-8")
    before = obs.read_bytes()
    monkeypatch.setenv("MEMEBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("META_OBSERVATIONS_FILE", "obs.jsonl")
    assert meta_cli.main(["report"]) == 0
    assert meta_cli.main(["report", "--json"]) == 0
    assert obs.read_bytes() == before
    det = MetaDetector(Config(), session=_NoNetworkSession(), observations_path=str(obs))
    det.scan_live(now=1_700_000_000.0, persist=False)
    assert obs.read_bytes() == before


def test_report_persist_flag_writes_and_respects_lock(tmp_path, monkeypatch, offline_detector, capsys):
    obs = tmp_path / "obs.jsonl"
    monkeypatch.setenv("MEMEBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("META_OBSERVATIONS_FILE", "obs.jsonl")
    holder = ProcessLock(str(tmp_path / "meta.lock"))
    assert holder.acquire()
    try:
        assert meta_cli.main(["report", "--persist"]) == 3   # collector running -> refused
        assert not obs.exists()
    finally:
        holder.release()
    assert meta_cli.main(["report", "--persist"]) == 0
    kinds = [json.loads(l)["kind"] for l in obs.read_text(encoding="utf-8").splitlines()]
    assert kinds == ["signal", "theme_counts"]
