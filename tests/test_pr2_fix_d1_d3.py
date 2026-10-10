"""PR2 follow-up to PR #20 review (D1-D3).

D1: streaming incremental JSONL reads (bot/meta/store.py).
D2: standalone preflight resolves MEMEBOT_DATA_DIR like bot.main (bot/preflight.py).
D3: relative DECISION_MODEL_PATH resolves against the repo root; a missing model
    is logged at ERROR and fails closed (bot/paths.py, bot/decision/bridge.py).

Synthetic fixtures under tmp_path only. No network, no trading process, no real
state / observation files. The shipped model artifact is only read.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

import bot.meta.store as store_module
from bot.config import Config, load_config
from bot.decision.bridge import DecisionShadowBridge
from bot.decision.models.logistic import LogisticModel
from bot.decision.schema import N_FEATURES
from bot.decision.signal import REJECT
from bot.meta.history import ObservationHistory
from bot.meta.model import MarketWindow, TokenSnapshot
from bot.meta.store import append_observation, read_observations_from
from bot.paths import REPO_ROOT, resolve_repo_path, with_resolved_data_paths
from bot.portfolio import Portfolio
from bot.process_lock import ProcessLock
from bot.risk import RiskManager
from bot.safety import TokenSafety

REPO = Path(__file__).resolve().parent.parent
SHIPPED_MODEL_REL = "artifacts/decision/model_logistic.v1.json"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("MEMEBOT_DATA_DIR", "META_OBSERVATIONS_FILE", "META_PROCESS_LOCK_FILE",
                 "DECISION_SHADOW_FILE", "STATE_FILE", "PROCESS_LOCK_FILE", "DECISION_MODEL_PATH"):
        monkeypatch.delenv(name, raising=False)


class _NoNetworkSession:
    def __getattr__(self, name):
        raise AssertionError(f"network access attempted via session.{name}")


def _row(i: int, mint: str = "M") -> dict:
    return {"kind": "market_snapshot", "recorded_at": float(i), "observed_at": float(i),
            "mint": mint, "market": {"1h": {"volume_usd": 1.0}}}


def _line(i: int) -> bytes:
    return (json.dumps(_row(i)) + "\n").encode()


# ---------------------------------------------------------------------------
# D1. Streaming incremental reads
# ---------------------------------------------------------------------------


def test_d1_incremental_rows_correct_without_duplicates(tmp_path):
    path = tmp_path / "obs.jsonl"
    for i in range(3):
        append_observation(str(path), _row(i))
    reader = ObservationHistory(str(path))
    assert [r["recorded_at"] for r in reader.load()] == [0.0, 1.0, 2.0]
    writer = ObservationHistory(str(path))
    writer.load()
    for i in range(3, 8):
        writer.append("market_snapshot", _row(i), recorded_at=float(i))
        if i % 2:
            reader.refresh()
    reader.refresh()
    reader.refresh()  # no-op refresh must not duplicate
    expected = [float(i) for i in range(8)]
    assert [r["recorded_at"] for r in reader.load()] == expected
    assert [r["recorded_at"] for r in writer.load()] == expected
    assert reader._offset == path.stat().st_size == writer._offset


def test_d1_read_from_offset_returns_only_new_rows_and_exact_offsets(tmp_path):
    path = tmp_path / "obs.jsonl"
    data = b"".join(_line(i) for i in range(5))
    path.write_bytes(data)
    first_two = len(_line(0)) + len(_line(1))
    rows, offset = read_observations_from(str(path), first_two)
    assert [r["recorded_at"] for r in rows] == [2.0, 3.0, 4.0]
    assert offset == len(data)
    rows, offset2 = read_observations_from(str(path), offset)
    assert rows == [] and offset2 == offset


def test_d1_partial_trailing_line_not_consumed_until_complete(tmp_path):
    path = tmp_path / "obs.jsonl"
    complete = _line(0) + _line(1)
    torn = _line(2)
    path.write_bytes(complete + torn[:17])
    reader = ObservationHistory(str(path))
    assert [r["recorded_at"] for r in reader.load()] == [0.0, 1.0]
    assert reader._offset == len(complete)
    rows, offset = read_observations_from(str(path), len(complete))
    assert rows == [] and offset == len(complete)
    with open(path, "ab") as fh:  # writer finishes the line
        fh.write(torn[17:])
    reader.refresh()
    assert [r["recorded_at"] for r in reader.load()] == [0.0, 1.0, 2.0]
    assert reader._offset == len(complete) + len(torn) == path.stat().st_size
    reader.refresh()
    assert len(reader.load()) == 3


def test_d1_unterminated_json_object_tail_policy_unchanged(tmp_path):
    # Recovery policy (out of scope to change): a complete JSON object without
    # its newline is consumed, exactly as before.
    path = tmp_path / "obs.jsonl"
    path.write_bytes(_line(0) + json.dumps(_row(1)).encode())
    rows, offset = read_observations_from(str(path), 0)
    assert [r["recorded_at"] for r in rows] == [0.0, 1.0]
    assert offset == path.stat().st_size
    # Undecodable complete lines are skipped but their bytes are consumed.
    path.write_bytes(b"{not json}\n" + _line(5))
    rows, offset = read_observations_from(str(path), 0)
    assert [r["recorded_at"] for r in rows] == [5.0] and offset == path.stat().st_size


def _count_reloads(monkeypatch, hist):
    calls = []
    original = hist._full_reload

    def spy():
        calls.append(1)
        return original()

    monkeypatch.setattr(hist, "_full_reload", spy)
    return calls


def test_d1_truncation_triggers_full_reload(tmp_path, monkeypatch):
    path = tmp_path / "obs.jsonl"
    path.write_bytes(b"".join(_line(i) for i in range(5)))
    hist = ObservationHistory(str(path))
    hist.load()
    calls = _count_reloads(monkeypatch, hist)
    with open(path, "r+b") as fh:  # same inode, shorter
        fh.truncate(len(_line(0)))
    hist.refresh()
    assert calls == [1]
    assert [r["recorded_at"] for r in hist.load()] == [0.0]
    assert hist._offset == len(_line(0))


def test_d1_rotation_triggers_full_reload(tmp_path, monkeypatch):
    path = tmp_path / "obs.jsonl"
    path.write_bytes(b"".join(_line(i) for i in range(5)))
    hist = ObservationHistory(str(path))
    hist.load()
    calls = _count_reloads(monkeypatch, hist)
    rotated = tmp_path / "new.jsonl"
    rotated.write_bytes(b"".join(_line(i) for i in range(100, 107)))
    os.replace(rotated, path)  # new inode, even though longer
    hist.refresh()
    assert calls == [1]
    assert [r["recorded_at"] for r in hist.load()] == [float(i) for i in range(100, 107)]
    assert hist._offset == path.stat().st_size


def test_d1_incremental_read_does_not_load_whole_file(tmp_path, monkeypatch):
    """Reading the tail of a large file must not buffer the file (O(new bytes))."""
    path = tmp_path / "big.jsonl"
    filler = {"kind": "signal", "pad": "x" * 2000}
    with open(path, "wb") as fh:
        for _ in range(2500):  # ~5 MB
            fh.write((json.dumps(filler) + "\n").encode())
        offset = fh.tell()
        fh.write(_line(1) + _line(2))
    assert path.stat().st_size > 4_000_000
    tracemalloc.start()
    try:
        rows, new_offset = read_observations_from(str(path), offset)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert [r["recorded_at"] for r in rows] == [1.0, 2.0]
    assert new_offset == path.stat().st_size
    assert peak < 256 * 1024, peak
    # The full-reload path streams too: never a whole-file read() call.
    real_open = open

    class _Guard:
        def __init__(self, fh):
            self._fh = fh

        def read(self, *a, **k):
            raise AssertionError("whole-file read() used")

        def __getattr__(self, name):
            return getattr(self._fh, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._fh.close()

    monkeypatch.setattr(store_module, "open", lambda *a, **k: _Guard(real_open(*a, **k)), raising=False)
    rows, full_offset = read_observations_from(str(path), 0)
    assert len(rows) == 2502 and full_offset == path.stat().st_size


# ---------------------------------------------------------------------------
# D2. Standalone preflight uses the resolved (bot.main) lock path
# ---------------------------------------------------------------------------


def _preflight_env(data_dir: str) -> dict:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MEMEBOT_DATA_DIR": data_dir,
        "LIVE_TRADING": "false",
        "WALLET_PRIVATE_KEY": "",
        "BURNER_WALLET_PUBKEY": "",
    }


def _run_preflight(cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "bot.preflight"], cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=60)


def _conflict_check(stdout: str) -> dict:
    result = json.loads(stdout)
    return result, next(p for p in result["prerequisites"] if p["name"] == "conflicting_process")


def test_d2_preflight_cli_reports_bot_main_lock_held_from_other_cwd(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    # Resolve the lock exactly as bot.main does: with_resolved_data_paths(load_config()).
    monkeypatch.setenv("MEMEBOT_DATA_DIR", str(data))
    monkeypatch.setenv("LIVE_TRADING", "false")
    monkeypatch.setenv("WALLET_PRIVATE_KEY", "")
    main_cfg = with_resolved_data_paths(load_config())
    assert main_cfg.process_lock_file == str(data / ".memebot.lock")
    env = _preflight_env(str(data))

    # Control: lock free -> no conflict reported.
    free = _run_preflight(other, env)
    assert free.returncode == 1, free.stderr  # not eligible (live disabled), but ran
    _, check = _conflict_check(free.stdout)
    assert check["passed"] is True

    holder = ProcessLock(main_cfg.process_lock_file)  # same mechanism bot.main uses
    assert holder.acquire()
    try:
        held = _run_preflight(other, env)
    finally:
        holder.release()
    assert held.returncode == 1, held.stderr
    result, check = _conflict_check(held.stdout)
    assert check["passed"] is False
    assert "CONFLICTING_PROCESS" in result["reason_codes"]
    assert result["eligible"] is False
    assert list(other.iterdir()) == []  # no stray lock/journal probe in the CWD


def test_d2_preflight_cli_invalid_data_dir_exits_2(tmp_path):
    other = tmp_path / "cwd"
    other.mkdir()
    proc = _run_preflight(other, _preflight_env("relative/data"))
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "invalid path configuration" in proc.stderr
    assert "MEMEBOT_DATA_DIR must be an absolute path" in proc.stderr
    assert list(other.iterdir()) == []


def test_d2_preflight_main_function_resolves_paths(tmp_path, monkeypatch, capsys):
    import bot.preflight as preflight_module

    data = tmp_path / "data"
    monkeypatch.setenv("MEMEBOT_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_run(cfg, **kwargs):
        seen["cfg"] = cfg
        return preflight_module.PreflightResult(
            eligible=False, reason_codes=["X"], wallet_address=None, derived_signer=None,
            observed_sol_balance=None, experiment_ceiling_usd=20.0, prerequisites=[],
            checked_at="t")

    monkeypatch.setattr(preflight_module, "run_preflight", fake_run)
    assert preflight_module.main([]) == 1
    assert seen["cfg"].process_lock_file == str(data / ".memebot.lock")
    assert seen["cfg"].state_file == str(data / "state.json")
    assert seen["cfg"].live_trading is False and seen["cfg"].wallet_private_key == ""
    capsys.readouterr()


# ---------------------------------------------------------------------------
# D3. Decision model path resolution
# ---------------------------------------------------------------------------


def _snap(now: float = 1700000000.0) -> TokenSnapshot:
    return TokenSnapshot(
        mint="MintPr2", symbol="P2", observed_at=now, price_usd=0.01, liquidity_usd=50_000.0,
        market_cap_usd=200_000.0, pair_created_at=now - 7200.0,
        market={"5m": MarketWindow(volume_usd=5000, tx_buys=40, tx_sells=20, price_change_pct=5.0),
                "1h": MarketWindow(volume_usd=40000, tx_buys=200, tx_sells=150, price_change_pct=10.0)},
        top_holder_pct=12.0, top10_holder_pct=35.0, creator_pct=5.0, source="test_meta",
    )


def _bridge(tmp_path, model_path: str) -> DecisionShadowBridge:
    cfg = Config(live_trading=False, bankroll_usd=20.0, max_position_pct=10.0,
                 max_open_positions=3, min_liquidity_usd=1000.0, decision_shadow_enabled=True,
                 decision_model_path=model_path,
                 decision_shadow_file=str(tmp_path / "shadow.jsonl"))
    return DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio(), session=_NoNetworkSession(),
                                history=ObservationHistory(str(tmp_path / "obs.jsonl")))


def test_d3_resolve_repo_path():
    assert resolve_repo_path(SHIPPED_MODEL_REL) == str(REPO_ROOT / SHIPPED_MODEL_REL)
    assert resolve_repo_path("/abs/model.json") == "/abs/model.json"


def test_d3_relative_model_path_loads_from_other_cwd(tmp_path, monkeypatch):
    assert (REPO_ROOT / SHIPPED_MODEL_REL).is_file()
    data = tmp_path / "data"
    monkeypatch.setenv("MEMEBOT_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)  # neither repo root nor data dir has artifacts/
    cfg = with_resolved_data_paths(Config(decision_model_path=SHIPPED_MODEL_REL))
    assert cfg.decision_model_path == str(REPO_ROOT / SHIPPED_MODEL_REL)  # not cwd, not data dir
    bridge = _bridge(tmp_path, SHIPPED_MODEL_REL)  # raw relative value: bridge resolves too
    assert bridge._engine is not None and bridge._engine.model is not None
    assert not (tmp_path / "artifacts").exists() and not data.exists()


def test_d3_missing_model_logs_error_and_rejects(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    missing_rel = "artifacts/decision/does_not_exist.v0.json"
    with caplog.at_level(logging.WARNING, logger="bot.decision.bridge"):
        bridge = _bridge(tmp_path, missing_rel)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, caplog.text
    msg = errors[0].getMessage()
    assert str(REPO_ROOT / missing_rel) in msg       # path actually tried
    assert "DECISION_MODEL_PATH" in msg and "Fix:" in msg  # how to fix
    assert bridge._engine.model is None
    safety = TokenSafety(mint="MintPr2", passed=True, liquidity_usd=50_000, price_usd=1.0, symbol="P2")
    result = bridge.evaluate_candidate(safety, now=1700000000.0, snapshot=_snap(), enrich=False)
    assert result.decision.action == REJECT
    assert "MODEL_UNAVAILABLE" in result.decision.reasons


def test_d3_absolute_model_path_unchanged(tmp_path, monkeypatch):
    model_file = tmp_path / "models" / "m.json"
    model_file.parent.mkdir()
    LogisticModel([0.0] * N_FEATURES, bias=0.0, model_version="pr2.abs").save(model_file)
    monkeypatch.setenv("MEMEBOT_DATA_DIR", str(tmp_path / "data"))
    other = tmp_path / "cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    cfg = with_resolved_data_paths(Config(decision_model_path=str(model_file)))
    assert cfg.decision_model_path == str(model_file)
    bridge = _bridge(tmp_path, str(model_file))
    assert bridge._engine.model is not None
    assert bridge._engine.model.model_version == "pr2.abs"
