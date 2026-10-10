"""PR 22 follow-up: load()/refresh() snapshot contract (Issue 1) and the
consumed-prefix fingerprint that makes disk-fallback reads fail closed on a
same-inode rewrite (Issue 2).

Synthetic rows in tmp_path only. No network, no bot.main / bot.meta process.
The reference is the verbatim origin/main implementation
(tests/_pr3_baseline_history_main.py).
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import shutil
from pathlib import Path

import pytest

from bot.config import Config
from bot.meta import history as history_module
from bot.meta import store as store_module
from bot.meta.detector import MetaDetector
from bot.meta.history import (
    HOUR, MARKET_ROW, SIGNAL_ROW, THEME_ROW, ObservationHistory, ObservationHistoryChanged,
    ObservationRows, StaleHistoryError,
)
from bot.meta.pipeline import MetaPipeline
from bot.decision.schema import N_FEATURES
from bot.safety import TokenSafety

from tests.test_pr3_history_windowing import (  # shared synthetic-stream helpers
    T0, BaselineHistory, _NoNetworkSession, _append, _bridge, _mint, _sha, _snap,
    _strip_latency, compare_histories, make_stream,
)


def _row(i: float, kind: str = SIGNAL_ROW) -> str:
    return json.dumps({"kind": kind, "recorded_at": float(i)}) + "\n"


def _ts(rows):
    return [r["recorded_at"] for r in rows]


def _write(path: Path, *ids) -> None:
    path.write_text("".join(_row(i) for i in ids), encoding="utf-8")


def _rewrite_same_inode(path: Path, pos: int, new: bytes) -> None:
    """Overwrite bytes in place: same inode, same size."""
    before = os.stat(path)
    with open(path, "r+b") as fh:
        fh.seek(pos)
        fh.write(new)
    after = os.stat(path)
    assert (after.st_ino, after.st_dev, after.st_size) == (before.st_ino, before.st_dev, before.st_size)


# ---------------------------------------------------------------------------
# Issue 1: snapshot contract
# ---------------------------------------------------------------------------

def test_snapshot_sequence_api_matches_list(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path, *range(10))
    h = ObservationHistory(str(path))
    base = BaselineHistory(str(path))
    rows, ref = h.load(), base.load()
    assert isinstance(rows, ObservationRows) and isinstance(ref, list)
    assert len(rows) == len(ref) == 10 and bool(rows) and list(rows) == ref
    for i in list(range(-10, 10)):
        assert rows[i] == ref[i]
    for bad in (10, -11, 99):
        with pytest.raises(IndexError):
            rows[bad]
        with pytest.raises(IndexError):
            ref[bad]
    with pytest.raises(TypeError):
        rows["0"]
    for s in (slice(None), slice(2, 5), slice(-3, None), slice(None, None, -1), slice(8, 1, -3),
              slice(1, 9, 2), slice(5, 2), slice(-100, 100), slice(None, None, 4), slice(7, None, -2)):
        assert rows[s] == ref[s], s
        assert isinstance(rows[s], list)
    assert rows == ref and ref == rows and not (rows != ref)
    assert rows != ref[:-1] and rows != ref + [{"kind": "x"}]
    other = list(ref)
    other[4] = {"kind": "changed"}
    assert rows != other
    assert rows != tuple(ref) and ref != tuple(ref)  # like list: never equal to a tuple
    assert rows == h.load() and rows == ObservationHistory(str(path)).load()
    assert ref[3] in rows and {"kind": "nope"} not in rows
    assert rows.index(ref[6]) == 6 and rows.count(ref[2]) == 1
    assert list(reversed(rows)) == list(reversed(ref))
    with pytest.raises(TypeError):
        hash(rows)


@pytest.mark.parametrize("op", [
    lambda r: r.append({}), lambda r: r.extend([{}]), lambda r: r.insert(0, {}), lambda r: r.pop(),
    lambda r: r.remove(r[0]), lambda r: r.clear(), lambda r: r.sort(), lambda r: r.reverse(),
    lambda r: r.__setitem__(0, {}), lambda r: r.__delitem__(0), lambda r: r.__iadd__([{}]),
    lambda r: r.__imul__(2),
])
def test_snapshot_mutation_raises_type_error_and_cache_is_untouched(tmp_path, op):
    path = tmp_path / "o.jsonl"
    _write(path, 1, 2, 3)
    h = ObservationHistory(str(path))
    rows = h.load()
    with pytest.raises(TypeError, match="immutable snapshot"):
        op(rows)
    assert _ts(h.load()) == [1.0, 2.0, 3.0] and len(h.refresh()) == 3


def test_augmented_assignment_rebinds_never_mutates(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path, 1)
    h = ObservationHistory(str(path))
    rows = h.load()
    with pytest.raises(TypeError):
        rows += [{"kind": "x"}]
    assert len(h.load()) == 1


def test_snapshot_retained_across_refresh_is_pinned(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path, 1, 2)
    h = ObservationHistory(str(path))
    first = h.load()
    assert h.refresh() is first and h.load() is first  # nothing new -> same object
    with path.open("a") as fh:
        fh.write(_row(3))
    assert len(first) == 2 and _ts(first) == [1.0, 2.0]  # appended bytes ignored before refresh
    second = h.refresh()
    assert second is not first and _ts(second) == [1.0, 2.0, 3.0]
    assert len(first) == 2 and _ts(first) == [1.0, 2.0] and first[-1]["recorded_at"] == 2.0
    assert first == second[:2] and first != second
    assert second.generation == first.generation and second.consumed_bytes > first.consumed_bytes


def test_snapshot_retained_across_forced_reload(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path, 1, 2)
    h = ObservationHistory(str(path))
    first = h.load()
    reloaded = h.load(force=True)
    assert reloaded is not first and reloaded.generation > first.generation
    assert first == reloaded and _ts(first) == [1.0, 2.0]  # unchanged file: still readable
    with path.open("a") as fh:
        fh.write(_row(3))
    third = h.load(force=True)
    assert _ts(third) == [1.0, 2.0, 3.0] and _ts(first) == [1.0, 2.0]


def test_snapshot_after_rotation_truncation_removal_recreation(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path, 1, 2, 3)
    h = ObservationHistory(str(path))
    base = BaselineHistory(str(path))
    old = h.load()
    base.load()
    # rotation: the bytes now live elsewhere -> old snapshot refuses
    os.replace(path, tmp_path / "o.jsonl.1")
    _write(path, 7, 8)
    with pytest.raises(StaleHistoryError):
        list(old)
    rotated = h.refresh()
    assert _ts(rotated) == _ts(base.refresh()) == [7.0, 8.0]
    with pytest.raises(StaleHistoryError):
        old[0]
    # truncation (in place, same inode)
    _write(path, 9)
    with pytest.raises(StaleHistoryError):
        list(rotated)
    truncated = h.refresh()
    assert _ts(truncated) == _ts(base.refresh()) == [9.0]
    # removal
    path.unlink()
    with pytest.raises(StaleHistoryError, match="no longer exists"):
        list(truncated)
    removed = h.refresh()
    assert removed == [] == base.refresh() and list(removed) == [] and len(removed) == 0
    # recreation
    _write(path, 42)
    assert list(removed) == []  # empty snapshot never touches the file
    recreated = h.refresh()
    assert _ts(recreated) == _ts(base.refresh()) == [42.0]
    # identical content at a new inode is the same history -> still readable
    shutil.copyfile(path, tmp_path / "copy")
    os.replace(tmp_path / "copy", path)
    assert _ts(recreated) == [42.0]


def test_iteration_after_file_changed_never_yields_different_rows(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path, 1, 2, 3)
    h = ObservationHistory(str(path))
    snap = h.load()
    data = path.read_bytes()
    _rewrite_same_inode(path, data.index(b"2.0"), b"5.0")
    for access in (list, lambda r: r[1], lambda r: r[-1], lambda r: r[0:3], lambda r: r == [{}] * 3):
        with pytest.raises(StaleHistoryError):
            access(snap)
    assert len(snap) == 3  # len is pinned, never touches the file
    assert _ts(h.refresh()) == [1.0, 5.0, 3.0]  # stale flag -> full reload from the file


def test_in_repo_callers_usage_patterns(tmp_path):
    """Every in-repo use of the load()/refresh() result (rg in bot/ and tests/)."""
    path = tmp_path / "o.jsonl"
    _write(path, 0, 1, 2)
    h, base = ObservationHistory(str(path)), BaselineHistory(str(path))
    # detector.py:171-172  (collector: force reload, then len of the result)
    h.load(force=True); base.load(force=True)
    assert len(h.load()) == len(base.load()) == 3
    # bridge.py:195  (refresh() result discarded, lookups follow)
    with path.open("a") as fh:
        fh.write(_row(3))
    h.refresh(); base.refresh()
    assert h.theme_first_seen() == base.theme_first_seen() and len(h.load()) == 4
    # tests: list comprehension over load()/refresh(), len(), == [], `is`
    assert [r["recorded_at"] for r in h.refresh()] == [r["recorded_at"] for r in base.refresh()]
    assert h.refresh() is h.load() and base.refresh() is base.load()
    assert len(h.refresh()) == len(ObservationHistory(str(path)).load())
    # writer path: append() reads its own row back through the cursor
    h.append(SIGNAL_ROW, {"token": "A"}, recorded_at=9.0)
    base.refresh()
    assert list(h.load()) == base.load() and len(h.load(force=True)) == 5


def test_detector_report_prior_rows_matches_main(tmp_path):
    path = tmp_path / "obs.jsonl"
    _append(path, make_stream(5, hours=5, mints=2, junk=True))
    det_new = MetaDetector(Config(rpc_url=""), session=_NoNetworkSession(), observations_path=str(path))
    det_base = MetaDetector(Config(rpc_url=""), session=_NoNetworkSession(), observations_path=str(path))
    det_base.history = BaselineHistory(str(path))
    now = T0 + 5 * HOUR
    for _ in range(2):
        r_new = det_new.evaluate([_snap(_mint(i), now) for i in range(2)], persist=False, now=now)
        r_base = det_base.evaluate([_snap(_mint(i), now) for i in range(2)], persist=False, now=now)
        assert r_new.to_dict() == r_base.to_dict()
        assert any(n.startswith("history: ") for n in r_new.notes)


# ---------------------------------------------------------------------------
# Issue 2: consumed-prefix fingerprint on disk fallback
# ---------------------------------------------------------------------------

def _trimmed_history(tmp_path, *, hours=8, retention=900.0, seed=3):
    path = tmp_path / "obs.jsonl"
    _append(path, make_stream(seed, hours=hours, mints=2))
    h = ObservationHistory(str(path), retention_seconds=retention)
    h.load()
    stats = h.stats()
    assert stats["market_horizon"] > T0 and stats["market_rows_in_memory"] < len(h.rows(MARKET_ROW)), stats
    return path, h


def test_same_inode_rewrite_after_head_is_detected_on_fallback(tmp_path):
    """Owner's case: inode, size and first 64 bytes unchanged; bytes after
    them rewritten with same-length different bytes; a lookup that needs the
    file must not return data that differs from what was ingested."""
    path, h = _trimmed_history(tmp_path)
    ingested = BaselineHistory(str(path))           # main's view of the ingested bytes
    ingested.load()
    target = T0 + HOUR                               # far below the trim horizon
    expected = ingested.nearest_market(_mint(0), target)
    assert expected is not None
    snap_before = h.load()
    data = path.read_bytes()
    st0 = os.stat(path)
    # rewrite the 1h volume of the row main would return, beyond byte 64
    line_start = data.index(json.dumps(expected).encode()[:60])
    assert line_start >= 64 or data.find(b'"volume_usd": ', 64) > 64
    pos = data.index(b'"volume_usd": ', max(line_start, 64)) + len(b'"volume_usd": ')
    _rewrite_same_inode(path, pos, b"9" if data[pos:pos + 1] != b"9" else b"8")
    st1 = os.stat(path)
    assert (st1.st_ino, st1.st_size) == (st0.st_ino, st0.st_size)
    assert path.read_bytes()[:64] == data[:64] and path.read_bytes() != data
    # main's stale cache would still answer `expected`; a fresh main read
    # would answer something else. The new code must not answer from the file.
    changed = BaselineHistory(str(path)).nearest_market(_mint(0), target)
    assert changed != expected
    scans = h.disk_scans
    try:
        got = h.nearest_market(_mint(0), target)
    except StaleHistoryError:
        got = "raised"
    assert got == "raised" or got == expected
    assert got != changed and h.disk_scans == scans + 1
    assert issubclass(StaleHistoryError, RuntimeError) and ObservationHistoryChanged is StaleHistoryError
    for fallback in (lambda: h.rows(MARKET_ROW), lambda: h.market_rows(_mint(0)),
                     lambda: h.earliest_observation(_mint(0)), lambda: h.theme_counts_at(target),
                     lambda: list(snap_before), lambda: snap_before[-1],
                     lambda: h.reconstruct_window(_snap(_mint(0), T0 + 3 * HOUR), T0 + 3 * HOUR)):
        with pytest.raises(StaleHistoryError):
            fallback()
    # memory-only lookups still answer from what was ingested
    now = T0 + 8 * HOUR
    last = now - 300  # newest tick; within the 900 s retention -> answered from memory
    scans = h.disk_scans
    assert h.nearest_market(_mint(1), last) == ingested.nearest_market(_mint(1), last) is not None
    assert h.theme_counts_at(last) == ingested.theme_counts_at(last) and h.disk_scans == scans
    # recovery: refresh()/load() see the stale flag and rebuild from the file
    assert h._stale and h.refresh().generation > snap_before.generation and not h._stale
    assert h.nearest_market(_mint(0), target) == changed
    assert compare_histories(h, BaselineHistory(str(path)), now, 2, random.Random(0), full_rows=True) == []


def test_change_in_last_consumed_bytes_is_detected(tmp_path):
    path, h = _trimmed_history(tmp_path)
    size = path.stat().st_size
    data = path.read_bytes()
    pos = data.rindex(b'"recorded_at": ') + len(b'"recorded_at": ')  # last line
    assert pos > size - 400
    _rewrite_same_inode(path, pos, b"1" if data[pos:pos + 1] != b"1" else b"2")
    assert h.refresh() is h.load()  # nothing new: no reload, rewrite unseen by refresh()
    with pytest.raises(StaleHistoryError, match="differ from what was ingested"):
        h.rows(THEME_ROW)


def test_change_only_in_unconsumed_tail_behaves_normally(tmp_path):
    path, h = _trimmed_history(tmp_path)
    consumed = h.fingerprint()["consumed_bytes"]
    assert consumed == path.stat().st_size
    extra = make_stream(4, hours=0.5, start=T0 + 8 * HOUR, mints=2)
    _append(path, extra[:3])
    torn = extra[3][1]
    with path.open("ab") as fh:
        fh.write(torn[:20])                     # unconsumed torn tail
    h.refresh()
    assert h.fingerprint()["consumed_bytes"] < path.stat().st_size
    # rewrite only the unconsumed tail, then complete the line differently
    tail_pos = h.fingerprint()["consumed_bytes"]
    _rewrite_same_inode(path, tail_pos, b"{" * 1 + torn[1:20])
    with path.open("ab") as fh:
        fh.write(torn[20:])
    _append(path, extra[4:])
    h.refresh()
    base = BaselineHistory(str(path))
    assert compare_histories(h, base, T0 + 8.5 * HOUR, 2, random.Random(1), full_rows=True) == []
    assert h.nearest_market(_mint(0), T0 + HOUR) == base.nearest_market(_mint(0), T0 + HOUR)


def test_identical_content_rewrite_passes(tmp_path):
    path, h = _trimmed_history(tmp_path)
    base = BaselineHistory(str(path))
    data = path.read_bytes()
    _rewrite_same_inode(path, 100, data[100:5000])  # same bytes written back
    assert h.nearest_market(_mint(0), T0 + HOUR) == base.nearest_market(_mint(0), T0 + HOUR)
    assert list(h.load()) == base.load()
    assert compare_histories(h, base, T0 + 8 * HOUR, 2, random.Random(2), full_rows=True) == []


def test_mismatch_in_middle_segment_yields_only_verified_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(history_module, "_SEGMENT_BYTES", 257)  # many small segments
    path = tmp_path / "o.jsonl"
    _write(path, *range(200))
    h = ObservationHistory(str(path))
    snap = h.load()
    assert h.fingerprint()["segments"] > 20
    data = path.read_bytes()
    pos = data.index(b"150.0")
    _rewrite_same_inode(path, pos, b"777.0")
    seen = []
    with pytest.raises(StaleHistoryError):
        for row in snap:
            seen.append(row["recorded_at"])
    assert seen == [float(i) for i in range(len(seen))] and len(seen) < 150  # verified prefix only


@pytest.mark.parametrize("seed", range(6))
def test_small_segments_incremental_fingerprint_and_fallback_match_main(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    monkeypatch.setattr(history_module, "_SEGMENT_BYTES", rng.choice([1, 7, 64, 333, 4096]))
    if seed % 2:
        monkeypatch.setattr(history_module, "_REFRESH_LIST_MAX_BYTES", 0)  # streamed refresh path
    path = tmp_path / "obs.jsonl"
    stream = make_stream(seed, hours=6, mints=2, junk=True, dup_prob=0.05, ooo_prob=0.1)
    new = ObservationHistory(str(path), retention_seconds=rng.choice([0.0, 1200.0]))
    base = BaselineHistory(str(path))
    i = 0
    while i < len(stream):
        n = rng.randint(1, 60)
        _append(path, stream[i:i + n])
        i += n
        new.refresh(); base.refresh()
    fresh = ObservationHistory(str(path))
    fresh.load()
    assert new.fingerprint()["digest"] == fresh.fingerprint()["digest"]  # incremental == from-scratch
    assert new.fingerprint()["consumed_bytes"] == path.stat().st_size
    assert compare_histories(new, base, T0 + 6 * HOUR, 2, rng, full_rows=True) == []


def test_rows_from_chunks_matches_iter_observation_rows(tmp_path):
    rng = random.Random(7)
    lines = [b"\n", b"{not json}\n", b"[1]\n", b'{"a": 1}\n', b'{"b": 2}', b'{"c": "\xff"}\n', b"  \n"]
    for trial in range(300):
        data = b"".join(rng.choice(lines) for _ in range(rng.randint(0, 25)))
        end = rng.randint(0, len(data))
        splits = sorted(rng.sample(range(0, end + 2), min(end + 2, rng.randint(0, 4))))
        ref = list(store_module.iter_observation_rows(io.BytesIO(data), end, splits))
        size = rng.randint(1, 40)
        chunks = [data[k:k + size] for k in range(0, end, size)]
        assert list(history_module._rows_from_chunks(iter(chunks), end, splits)) == ref, (data, end, splits)


def test_concurrent_rewrite_between_parse_and_hash_forces_full_reload(tmp_path, monkeypatch):
    """Bytes appended then rewritten between the incremental parse and the
    fingerprint read: memory, snapshot and fingerprint must all describe the
    same (new) bytes, never old rows under a fingerprint of new bytes."""
    path = tmp_path / "o.jsonl"

    def theme(i, name):
        return json.dumps({"kind": THEME_ROW, "recorded_at": float(i), "counts": {name: 1}}) + "\n"

    path.write_text(theme(1, "t1") + theme(2, "t2"), encoding="utf-8")
    h = ObservationHistory(str(path))
    h.load()
    with path.open("a") as fh:
        fh.write(theme(3, "t3"))
    real = history_module.read_observations_from

    def racing(p, off):
        rows, end = real(p, off)
        data = path.read_bytes()
        _rewrite_same_inode(path, data.rindex(b'"t3"'), b'"t9"')  # changes after the parse
        return rows, end

    monkeypatch.setattr(history_module, "read_observations_from", racing)
    snap = h.refresh()
    fresh_main = BaselineHistory(str(path))
    assert list(snap) == fresh_main.load()
    assert h.theme_first_seen() == fresh_main.theme_first_seen() == {"t1": 1.0, "t2": 2.0, "t9": 3.0}
    assert h.theme_counts_at(3.0) == {"t9": 1}
    fresh = ObservationHistory(str(path))
    fresh.load()
    assert h.fingerprint()["digest"] == fresh.fingerprint()["digest"]


def test_fallback_reads_never_modify_the_file(tmp_path):
    path, h = _trimmed_history(tmp_path)
    st = path.stat()
    before = (_sha(path), st.st_size, st.st_mtime_ns)
    h.refresh(); h.load(force=True); h.rows(MARKET_ROW); list(h.load()); h.load()[5]; h.load()[-3:]
    h.nearest_market(_mint(0), T0 + HOUR); h.theme_counts_at(T0 + HOUR); h.theme_first_seen()
    st = path.stat()
    assert (_sha(path), st.st_size, st.st_mtime_ns) == before


def test_production_lookups_do_no_fingerprint_rehash(tmp_path, monkeypatch):
    path = tmp_path / "obs.jsonl"
    stream = make_stream(11, hours=10, mints=3)
    _append(path, stream[: len(stream) // 2])
    h = ObservationHistory(str(path), clock=lambda: T0 + 10 * HOUR)
    h.load()
    calls = []
    real = history_module._verified_chunks
    monkeypatch.setattr(history_module, "_verified_chunks", lambda *a: (calls.append(1), real(*a))[1])
    _append(path, stream[len(stream) // 2:])
    h.refresh()
    now = T0 + 10 * HOUR - 300
    for i in range(3):
        h.attach_reconstructed_windows(_snap(_mint(i), now), now)
    h.theme_acceleration("ai_agents", 2, now); h.theme_first_seen(); len(h.load())
    assert calls == [] and h.disk_scans == 0


# ---------------------------------------------------------------------------
# fail-closed behaviour in the real callers
# ---------------------------------------------------------------------------

def test_bridge_fails_closed_then_recovers(tmp_path):
    path = tmp_path / "obs.jsonl"
    _append(path, make_stream(21, hours=6, mints=1))
    now = T0 + 6 * HOUR - 300
    # Small retention so the bridge's 4h lookups need the disk fallback.
    h = ObservationHistory(str(path), retention_seconds=600.0, clock=lambda: now)
    weights = [0.01] * N_FEATURES
    bridge = _bridge(tmp_path, "new", h, weights)
    safety = TokenSafety(mint=_mint(0), passed=True, liquidity_usd=75_000, price_usd=0.05, symbol="X")
    bridge.evaluate_candidate(safety, now=now, snapshot=_snap(_mint(0), now))
    real_refresh = h.refresh
    data = path.read_bytes()

    def refresh_then_rewrite():
        out = real_refresh()
        if path.read_bytes() == data:
            pos = data.index(b'"volume_usd": ', 200) + len(b'"volume_usd": ')
            _rewrite_same_inode(path, pos, b"9" if data[pos:pos + 1] != b"9" else b"8")
        return out

    h.refresh = refresh_then_rewrite
    journal = tmp_path / "shadow_new.jsonl"
    lines_before = journal.read_text().count("\n")
    with pytest.raises(StaleHistoryError):  # main.py:266-272 catches -> "failed closed", candidate skipped
        bridge.evaluate_candidate(safety, now=now, snapshot=_snap(_mint(0), now))
    assert journal.read_text().count("\n") == lines_before  # nothing journaled from changed data
    assert h._stale
    h.refresh = real_refresh
    # Same history object, fresh bridge (the engine de-duplicates repeat events).
    bridge2 = _bridge(tmp_path, "new2", h, weights)
    r_new = bridge2.evaluate_candidate(safety, now=now, snapshot=_snap(_mint(0), now))
    assert not h._stale
    base_bridge = _bridge(tmp_path, "base", BaselineHistory(str(path)), weights)
    r_base = base_bridge.evaluate_candidate(safety, now=now, snapshot=_snap(_mint(0), now))
    assert _strip_latency(r_new.to_dict()) == _strip_latency(r_base.to_dict())


def test_collector_tick_fails_closed_logs_and_next_tick_reloads(tmp_path, caplog):
    path = tmp_path / "obs.jsonl"
    _append(path, make_stream(22, hours=6, mints=2))
    now = T0 + 6 * HOUR - 300
    pipe = MetaPipeline(Config(rpc_url="", live_trading=False), session=_NoNetworkSession(),
                        observations_path=str(path))
    h = ObservationHistory(str(path), retention_seconds=600.0, clock=lambda: now)
    pipe.detector.history = h
    real_load = h.load
    data = path.read_bytes()
    rewrite_once = [True]

    def load_then_rewrite(force=False):
        out = real_load(force)
        if force and rewrite_once[0]:
            rewrite_once[0] = False
            pos = data.index(b'"volume_usd": ', 200) + len(b'"volume_usd": ')
            _rewrite_same_inode(path, pos, b"9" if data[pos:pos + 1] != b"9" else b"8")
        return out

    h.load = load_then_rewrite
    ticks = []

    def snapshot_universe(now=None, fetch_mints=None):
        ticks.append(1)
        return pipe.detector.evaluate([_snap(_mint(i), T0 + 6 * HOUR - 300) for i in range(2)],
                                      persist=True, persist_raw=True, now=T0 + 6 * HOUR - 300)

    pipe.snapshot_universe = snapshot_universe
    size_after_rewrite = None
    with caplog.at_level(logging.ERROR, logger="bot.meta.pipeline"):
        pipe.run_forever(cycles=1)
        size_after_rewrite = path.stat().st_size
    assert "meta pipeline tick failed" in caplog.text and "StaleHistoryError" in caplog.text
    assert size_after_rewrite == len(data)  # nothing appended by the failed tick
    assert h._stale
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="bot.meta.pipeline"):
        pipe.run_forever(cycles=1)  # next tick: load(force=True) rebuilds from the file
    assert "tick failed" not in caplog.text and ticks == [1, 1]
    assert path.stat().st_size > len(data) and not h._stale
    assert len(h.load()) == len(BaselineHistory(str(path)).load())
