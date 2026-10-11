"""ObservationRows.__eq__ must never answer from stale pinned bytes (follow-up
to #22). PR #22's __eq__ returned True for two snapshots with the same
history/generation/consumed offset (including ``a == a``) without checking the
file, so a same-inode rewrite was invisible to ``==`` although the snapshot
contract says stale contents raise StaleHistoryError.

Synthetic files under tmp_path only. No network, no RPC, no bot.main/bot.meta
process. Every test also checks that reading/comparing/trimming/falling back
never modifies the observation file (sha256 + mtime_ns + size + inode).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tracemalloc
from pathlib import Path

import pytest

from bot.meta.history import (
    MARKET_ROW, SIGNAL_ROW, THEME_ROW, ObservationHistory, ObservationRows, StaleHistoryError,
)

from tests.test_pr3_history_windowing import T0, _append, make_stream


def _row(i: int, **extra) -> bytes:
    # Fixed-width value so a same-length in-place rewrite is easy to place.
    row = {"kind": SIGNAL_ROW, "recorded_at": float(1_000_000 + i), "score": f"{i:06d}"}
    row.update(extra)
    return (json.dumps(row) + "\n").encode()


def _write(path: Path, n: int = 12, **extra) -> None:
    path.write_bytes(b"".join(_row(i, **extra) for i in range(n)))


def _file_state(path: Path) -> tuple:
    st = os.stat(path)
    return (hashlib.sha256(path.read_bytes()).hexdigest(), st.st_mtime_ns, st.st_size, st.st_ino, st.st_dev)


def _parse_file(path: Path) -> list:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _rewrite_consumed_same_inode(path: Path, line: int = 7) -> None:
    """Change one byte of a consumed row past the checked 64-byte head:
    same inode, same size, same head bytes, different content."""
    data = path.read_bytes()
    st0 = os.stat(path)
    start = sum(len(l) for l in data.splitlines(keepends=True)[:line])
    pos = data.index(b'"score": "', start) + len(b'"score": "')
    assert pos > 64
    old = data[pos:pos + 1]
    with open(path, "r+b") as fh:
        fh.seek(pos)
        fh.write(b"9" if old != b"9" else b"8")
    st1 = os.stat(path)
    new = path.read_bytes()
    assert (st1.st_ino, st1.st_dev, st1.st_size) == (st0.st_ino, st0.st_dev, st0.st_size)
    assert new[:64] == data[:64] and new != data


# ---------------------------------------------------------------------------
# 1. normal equality
# ---------------------------------------------------------------------------

def test_1_valid_snapshot_equality(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path)
    before = _file_state(path)
    h = ObservationHistory(str(path))
    snap = h.load()
    expected = _parse_file(path)
    peer = ObservationRows(h)                  # same history + cursor, distinct object
    other = ObservationHistory(str(path)).load()  # equivalent snapshot, other history
    assert peer is not snap and other is not snap

    scans = h.disk_scans
    assert snap == snap
    assert h.disk_scans == scans + 1           # same pin: exactly one verification pass (hash only)
    assert not (snap != snap)
    assert snap == peer and peer == snap
    assert snap == other and other == snap and not (snap != other)
    assert snap == expected and expected == snap and not (snap != expected)
    assert snap == list(snap)
    assert snap != expected[:-1] and snap != expected + [{"kind": "x"}]
    changed = list(expected)
    changed[0] = {"kind": "changed"}
    assert snap != changed and changed != snap
    assert snap != tuple(expected)             # like list: never equal to a tuple
    # empty history
    empty = ObservationHistory(str(tmp_path / "missing.jsonl")).load()
    assert empty == empty and empty == [] and [] == empty
    assert _file_state(path) == before


# ---------------------------------------------------------------------------
# 2. stale self-equality
# ---------------------------------------------------------------------------

def test_2_stale_self_equality_raises(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path)
    h = ObservationHistory(str(path))
    snap = h.load()
    assert snap == snap
    _rewrite_consumed_same_inode(path)
    after = _file_state(path)
    assert h.refresh() is snap                 # inode/size/head unchanged: refresh can't see it
    a = snap
    with pytest.raises(StaleHistoryError, match="differ from what was ingested"):
        snap == snap
    with pytest.raises(StaleHistoryError):
        a == a                                 # identity: `a is a`
    with pytest.raises(StaleHistoryError):
        snap != snap                           # __ne__ delegates to __eq__
    with pytest.raises(StaleHistoryError):
        snap.__eq__(snap)
    assert h._stale                            # next refresh()/load() rebuilds
    assert _file_state(path) == after          # compare never writes


# ---------------------------------------------------------------------------
# 3. stale peer equality
# ---------------------------------------------------------------------------

def test_3_stale_peer_equality_raises(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path)
    h = ObservationHistory(str(path))
    a = h.load()
    b = ObservationRows(h)                     # same history, generation and cursor
    c = ObservationHistory(str(path)).load()   # same bytes via another history
    ref = list(a)
    assert a is not b and a == b == c
    assert (b.generation, b.consumed_bytes) == (a.generation, a.consumed_bytes)
    _rewrite_consumed_same_inode(path)
    after = _file_state(path)
    for x, y in ((a, b), (b, a), (a, c), (c, a)):
        with pytest.raises(StaleHistoryError):
            x == y
        with pytest.raises(StaleHistoryError):
            x != y
    # against lists: equal, differing early (no short-circuit) and length mismatch
    early = [{"kind": "different"}] + ref[1:]
    for other in (ref, early, [], ref[:-1]):
        with pytest.raises(StaleHistoryError):
            a == other
        with pytest.raises(StaleHistoryError):
            other == a                         # reflected: list.__eq__ -> NotImplemented -> a.__eq__
    assert _file_state(path) == after


# ---------------------------------------------------------------------------
# 4. recovery
# ---------------------------------------------------------------------------

def test_4_refresh_recovers_after_mismatch(tmp_path):
    path = tmp_path / "o.jsonl"
    _write(path)
    h = ObservationHistory(str(path))
    old = h.load()
    _rewrite_consumed_same_inode(path)
    after = _file_state(path)
    with pytest.raises(StaleHistoryError):
        old == old
    fresh = h.refresh()
    assert fresh is not old and fresh.generation > old.generation and not h._stale
    current = _parse_file(path)
    assert fresh == fresh and fresh == ObservationRows(h)
    assert fresh == current and current == fresh
    assert fresh == ObservationHistory(str(path)).load()
    assert h.load() is fresh
    with pytest.raises(StaleHistoryError):     # the old snapshot stays stale for good
        old == old
    assert not h._stale                        # an old generation does not re-flag the new one
    assert fresh == current
    # appends after recovery are picked up normally
    with path.open("ab") as fh:
        fh.write(_row(99))
    newer = h.refresh()
    assert newer == _parse_file(path) and fresh == current and newer != fresh
    assert _file_state(path)[2] == after[2] + len(_row(99))


# ---------------------------------------------------------------------------
# 5. NaN rows (documented difference from plain list equality)
# ---------------------------------------------------------------------------

def test_5_nan_rows_documented_semantics(tmp_path):
    import json.decoder
    path = tmp_path / "nan.jsonl"
    _write(path, 6, v=float("nan"))
    assert b"NaN" in path.read_bytes()
    before = _file_state(path)
    h = ObservationHistory(str(path))
    snap = h.load()
    assert all(math.isnan(r["v"]) for r in snap)
    # CPython's json maps every "NaN" token to the module-level json.decoder.NaN
    # object, so all parsed NaNs are one object and dict/list == (identity
    # before __eq__) treats them as equal. Snapshot equality relies on that
    # implementation detail for NaN rows; this asserts it so a change shows up.
    assert snap[0]["v"] is snap[1]["v"] is json.decoder.NaN
    mat = list(snap)
    # Same pin: verified bytes identical -> True without parsing.
    assert snap == snap and snap == ObservationRows(h)
    # Element-wise paths: rows are freshly parsed, NaN singleton -> equal.
    assert snap == mat and mat == snap and list(snap) == list(snap)
    assert snap == ObservationHistory(str(path)).load()
    # Difference from plain list equality: list == checks identity per element
    # (row dict), the snapshot never shares row objects with the other operand.
    # A NaN that is not json's singleton therefore makes rows unequal even
    # when the same list object compares equal to itself.
    foreign = list(snap)
    foreign[0] = dict(foreign[0], v=float("nan"))
    assert foreign == foreign                  # plain list: same object -> True
    assert snap != foreign and foreign != snap # snapshot: element-wise -> False
    same_rows = list(mat)                      # shares the row objects of `mat`
    assert same_rows == mat                    # list vs list sharing row objects -> True
    # Validation still wins over NaN: a stale NaN snapshot raises, never True/False.
    _rewrite_consumed_same_inode(path, line=4)
    with pytest.raises(StaleHistoryError):
        snap == snap
    with pytest.raises(StaleHistoryError):
        snap == mat
    assert _file_state(path)[0] != before[0]   # only our own rewrite changed it


# ---------------------------------------------------------------------------
# Python-semantics limits (documented, not fixable in __eq__)
# ---------------------------------------------------------------------------

def test_limit_container_identity_shortcut_skips_eq(tmp_path):
    """list containment/equality/index/count use PyObject_RichCompareBool, which
    returns True for the identical object without calling __eq__, so they can
    not detect a stale snapshot. Direct comparison does."""
    path = tmp_path / "o.jsonl"
    _write(path)
    h = ObservationHistory(str(path))
    snap = h.load()
    _rewrite_consumed_same_inode(path)
    scans = h.disk_scans
    assert snap in [snap]
    assert [snap] == [snap]
    assert [snap].index(snap) == 0 and [snap].count(snap) == 1
    assert (snap,) == (snap,)
    assert h.disk_scans == scans               # __eq__ was never called
    # non-list operands: NotImplemented, nothing read, so never raises
    assert snap != (1, 2) and not (snap == "x") and h.disk_scans == scans
    with pytest.raises(StaleHistoryError):
        snap == snap
    with pytest.raises(StaleHistoryError):
        snap in [ObservationRows(h)]           # distinct object -> __eq__ runs


# ---------------------------------------------------------------------------
# Read/compare/trim/fallback never modify the file; bounded memory holds
# ---------------------------------------------------------------------------

def test_compare_trim_fallback_do_not_modify_file_and_memory_is_bounded(tmp_path):
    path = tmp_path / "obs.jsonl"
    _append(path, make_stream(5, hours=8, mints=2))
    while path.stat().st_size < 9 * (1 << 20):  # 9+ fingerprint segments
        _append(path, make_stream(6, hours=8, start=T0 + 9 * 3600, mints=2))
    before = _file_state(path)
    h = ObservationHistory(str(path), retention_seconds=900.0)
    snap = h.load()                            # read + trim
    assert h.stats()["market_horizon"] > T0    # trimmed
    assert h.fingerprint()["segments"] >= 9
    assert len(h.rows(MARKET_ROW)) > h.stats()["market_rows_in_memory"]  # disk fallback
    h.theme_counts_at(T0 + 3600.0)            # fallback below horizon
    assert ObservationRows.__slots__ == ("_history", "_path", "_generation", "_len", "_end", "_splits", "_pin")
    tracemalloc.start()
    try:
        assert snap == snap and snap == ObservationRows(h)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # same-pin equality is hash-only: about one 1 MiB segment buffer, no rows
    assert peak < 2.5 * (1 << 20), peak
    other = ObservationHistory(str(path), retention_seconds=900.0).load()
    tracemalloc.start()
    try:
        assert snap == other
        _, peak_cmp = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # two verified segment streams, rows one at a time: independent of file size
    assert peak_cmp < 6.5 * (1 << 20) < path.stat().st_size, peak_cmp
    assert h.refresh() is snap
    assert _file_state(path) == before
    print(f"file={path.stat().st_size} same_pin_peak={peak} cross_peak={peak_cmp}")
