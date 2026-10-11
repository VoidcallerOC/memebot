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


# ===========================================================================
# Follow-up: distinct snapshots must be consumed to the END on both sides.
# zip(self, other) stops as soon as `self` is exhausted, leaving `other`
# suspended after its last parsed row, so consumed bytes after that row that
# live in a later fingerprint chunk (segment or tail) were never verified.
# The fixtures below put a snapshot's last valid row in segment 0 and its
# trailing consumed (unparseable/blank) lines past the 1 MiB boundary.
# ===========================================================================

from bot.meta import history as _history_module  # noqa: E402
from bot.meta.store import _parse_line  # noqa: E402

SEG = _history_module._SEGMENT_BYTES
assert SEG == 1 << 20


def _junk(nbytes: int) -> bytes:
    """Complete lines that are consumed but parse to no row: invalid JSON,
    blank lines and a JSON non-object."""
    unit = b"#not json " + b"x" * 52 + b"\n" + b"\n" + b"[1, 2, 3]\n"
    return unit * (nbytes // len(unit) + 1)


def _rows_bytes(n: int, first: str | None = None, pad: int = 0) -> bytes:
    out = []
    for i in range(n):
        extra = {"pad": "p" * pad} if pad else {}
        line = _row(i, **extra)
        if i == 0 and first is not None:
            line = line.replace(b'"score": "000000"', f'"score": "{first}"'.encode())
        out.append(line)
    return b"".join(out)


def _rows_of(path: Path) -> list:
    return [r for r in (_parse_line(l) for l in path.read_bytes().splitlines(keepends=True)) if r is not None]


def _flip_byte_same_inode(path: Path, pos: int) -> None:
    data = path.read_bytes()
    st0 = os.stat(path)
    assert pos >= 64
    with open(path, "r+b") as fh:
        fh.seek(pos)
        fh.write(b"y" if data[pos:pos + 1] != b"y" else b"z")
    st1 = os.stat(path)
    new = path.read_bytes()
    assert (st1.st_ino, st1.st_dev, st1.st_size) == (st0.st_ino, st0.st_dev, st0.st_size)
    assert new[:64] == data[:64] and new != data


def _junk_tail_pos(snap: ObservationRows) -> int:
    """A byte in the snapshot's last fingerprint chunk, past its last row."""
    pos = snap.consumed_bytes - 20
    assert pos >= SEG and snap._pin.count >= 1
    return pos


def test_eq_follow_1_same_rows_different_consumed_length_trailing_stale(tmp_path):
    """Same history, same generation, same parsed rows; `b` additionally
    consumed >1 MiB of unparseable lines. A stale byte in that trailing
    region must raise, not compare equal."""
    path = tmp_path / "o.jsonl"
    path.write_bytes(_rows_bytes(12))
    h = ObservationHistory(str(path))
    a = h.load()
    with path.open("ab") as fh:
        fh.write(_junk(SEG + 4096))
    b = h.refresh()
    assert b is not a and b.generation == a.generation and len(b) == len(a) == 12
    assert b.consumed_bytes > SEG > a.consumed_bytes
    assert a == b and b == a                    # valid: equal
    _flip_byte_same_inode(path, _junk_tail_pos(b))
    after = _file_state(path)
    with pytest.raises(StaleHistoryError):
        a == b                                  # plain zip: True (b's tail never read)
    with pytest.raises(StaleHistoryError):
        b == a
    assert _file_state(path) == after


def test_eq_follow_2_both_operand_orders_raise(tmp_path):
    """Distinct histories/files; the stale bytes belong to one operand only.
    Both orders and both operators raise."""
    x_path, y_path = tmp_path / "x.jsonl", tmp_path / "y.jsonl"
    rows = _rows_bytes(12)
    x_path.write_bytes(rows + _junk(100)[:100 // 64 * 64])
    y_path.write_bytes(rows + _junk(SEG + 4096))
    x = ObservationHistory(str(x_path)).load()
    y = ObservationHistory(str(y_path)).load()
    assert len(x) == len(y) and x == y and y == x
    _flip_byte_same_inode(y_path, _junk_tail_pos(y))
    for op in (lambda: x == y, lambda: y == x, lambda: x != y, lambda: y != x):
        with pytest.raises(StaleHistoryError):
            op()
    assert x == x and x == _rows_of(x_path)    # the untouched operand is still valid


@pytest.mark.parametrize("stale_in", ["self", "other"])
def test_eq_follow_3_early_row_mismatch_then_stale_later_segment(tmp_path, stale_in):
    """The first parsed row differs (result would be False) and a later
    segment of one operand is stale: equality must raise, in both orders."""
    x_path, y_path = tmp_path / "x.jsonl", tmp_path / "y.jsonl"
    # >1 MiB of valid rows so rows themselves span segments, then a trailing
    # consumed region in a later chunk.
    # The junk is > 1 MiB so the last fingerprint chunk holds no row at all
    # (validation is per chunk, before any row of that chunk is yielded).
    x_path.write_bytes(_rows_bytes(4500, pad=200) + _junk(SEG + 4096))
    y_path.write_bytes(_rows_bytes(4500, first="ZZZZZZ", pad=200) + _junk(SEG + 4096))
    assert x_path.stat().st_size > 2 * SEG and x_path.stat().st_size == y_path.stat().st_size
    x = ObservationHistory(str(x_path)).load()
    y = ObservationHistory(str(y_path)).load()
    assert len(x) == len(y) == 4500
    assert x[0] != y[0] and x != y and y != x  # valid: unequal, no exception
    stale_path, stale_snap = (x_path, x) if stale_in == "self" else (y_path, y)
    _flip_byte_same_inode(stale_path, _junk_tail_pos(stale_snap))
    for op in (lambda: x == y, lambda: y == x, lambda: x != y, lambda: y != x):
        with pytest.raises(StaleHistoryError):
            op()
    # length mismatch does not bypass validation either (either side stale)
    with pytest.raises(StaleHistoryError):
        stale_snap == [{"kind": "only one"}]


def test_eq_follow_4_valid_distinct_snapshots_with_different_trailing_junk_are_equal(tmp_path):
    x_path, y_path, z_path = tmp_path / "x.jsonl", tmp_path / "y.jsonl", tmp_path / "z.jsonl"
    rows = _rows_bytes(12)
    x_path.write_bytes(rows)
    y_path.write_bytes(rows + b"\n\n#garbage\n")
    z_path.write_bytes(rows + _junk(SEG + 4096))
    before = [_file_state(p) for p in (x_path, y_path, z_path)]
    snaps = [ObservationHistory(str(p)).load() for p in (x_path, y_path, z_path)]
    assert len({s.consumed_bytes for s in snaps}) == 3
    expected = _rows_of(x_path)
    for s in snaps:
        assert s == expected and expected == s
        for t in snaps:
            assert s == t and not (s != t)
    # and unequal when a row really differs
    w_path = tmp_path / "w.jsonl"
    w_path.write_bytes(_rows_bytes(12, first="ZZZZZZ") + _junk(SEG + 4096))
    w = ObservationHistory(str(w_path)).load()
    for s in snaps:
        assert s != w and w != s
    assert [_file_state(p) for p in (x_path, y_path, z_path)] == before


def test_eq_follow_5_refresh_recovery_after_trailing_stale(tmp_path):
    path = tmp_path / "o.jsonl"
    path.write_bytes(_rows_bytes(12))
    h = ObservationHistory(str(path))
    a = h.load()
    with path.open("ab") as fh:
        fh.write(_junk(SEG + 4096))
    b = h.refresh()
    _flip_byte_same_inode(path, _junk_tail_pos(b))
    with pytest.raises(StaleHistoryError):
        a == b
    assert h._stale
    fresh = h.refresh()
    assert fresh.generation > b.generation and not h._stale
    assert fresh.consumed_bytes == b.consumed_bytes
    assert fresh == fresh and fresh == ObservationRows(h) and fresh == _rows_of(path)
    assert fresh == ObservationHistory(str(path)).load()
    assert a == a and a == fresh               # a's bytes [0, a.end) are unchanged: still valid
    with pytest.raises(StaleHistoryError):
        b == b                                  # the old stale snapshot keeps raising
    with pytest.raises(StaleHistoryError):
        b == fresh
    with pytest.raises(StaleHistoryError):
        fresh == b
    assert not h._stale                         # old generation does not re-flag the new one


def test_eq_follow_6_distinct_comparison_memory_bounded_and_file_untouched(tmp_path):
    x_path, y_path = tmp_path / "x.jsonl", tmp_path / "y.jsonl"
    body = _rows_bytes(36000, pad=200)          # ~9.6 MB of valid rows
    x_path.write_bytes(body)
    y_path.write_bytes(body + _junk(2 * SEG))   # same rows, ~2 MiB more consumed bytes
    states = [_file_state(p) for p in (x_path, y_path)]
    x = ObservationHistory(str(x_path)).load()
    y = ObservationHistory(str(y_path)).load()
    assert x._pin.count >= 9 and y._pin.count >= x._pin.count + 2
    peaks = []
    for op in (lambda: x == y, lambda: y == x):
        tracemalloc.start()
        try:
            assert op() is True
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    # two verified segment streams, one row at a time: independent of file size
    assert max(peaks) < 6.5 * (1 << 20) < x_path.stat().st_size, peaks
    assert ObservationRows.__slots__ == ("_history", "_path", "_generation", "_len", "_end", "_splits", "_pin")
    assert [_file_state(p) for p in (x_path, y_path)] == states
    print(f"x={x_path.stat().st_size} y={y_path.stat().st_size} peaks={peaks}")
