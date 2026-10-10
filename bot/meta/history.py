"""Historical observation support on top of the JSONL store.

Reconstructed implementation. Rows are appended with a ``kind``:
  * ``market_snapshot`` — raw TokenSnapshot components as observed
  * ``signal``          — scored per-token summary
  * ``theme_counts``    — tokens per narrative theme at that evaluation

A longer window (4h) is reconstructed only when every hourly trailing 1h
observation it needs exists within tolerance. Nothing is interpolated: a gap
means the window stays UNVERIFIED.

The file is the complete record and is never trimmed. In memory only a
bounded window of market/theme rows plus an all-time theme first-seen index is
kept (see ObservationHistory); results are identical to holding every row.
Anything that has to re-read already-consumed rows from the file (lookups
below the in-memory horizon, iterating a load()/refresh() result) first checks
those bytes against a fingerprint taken while they were ingested and raises
StaleHistoryError instead of returning different data.
"""
from __future__ import annotations

import hashlib
import io
import math
import operator
import os
import time
from collections.abc import Sequence
from itertools import islice
from typing import Any, Callable, Iterable, Iterator, NoReturn, Optional

from .model import MarketWindow, TokenSnapshot, Score, ok, unverified
from .store import (_parse_line, append_observation, iter_observation_rows,
                    read_observations_from, stream_observations_from)
from ..paths import resolve_data_path

MARKET_ROW = "market_snapshot"
SIGNAL_ROW = "signal"
THEME_ROW = "theme_counts"

DEFAULT_OBSERVATIONS_FILE = "meta_observations.jsonl"
HOUR = 3600.0
WINDOW_TOLERANCE_SECONDS = 600.0
RECONSTRUCTED_WINDOWS = {"4h": 4}
_HEAD_BYTES = 64
# Incremental refreshes larger than this are streamed row by row instead of
# being parsed into one transient list first (e.g. a reader idle for hours).
_REFRESH_LIST_MAX_BYTES = 4 * 1024 * 1024
# Consumed-prefix fingerprint: one BLAKE2b-256 digest per fixed-size segment
# of consumed bytes (32 bytes of digest per MiB of file) plus a running hash of
# the partial last segment. Updated with exactly the bytes the cursor consumes,
# so maintaining it costs O(new bytes); the file is only re-hashed on the
# disk-fallback paths, one verified segment at a time.
_SEGMENT_BYTES = 1 << 20
_DIGEST_SIZE = 32

# In-memory retention (the JSONL file on disk is never trimmed or rewritten).
# The oldest stored observation any production lookup needs is the 4h window's
# now-3h hourly point, +-tolerance (theme acceleration only needs now-1h
# +-tolerance). The extra margin absorbs: the collector's evaluate(now) using
# the tick-start time while its reload happens up to a whole tick (~600 s worst
# case) later, late/out-of-order appends, bridge refresh lag and small
# backwards wall-clock steps. Exactness never depends on the margin: a lookup
# reaching below the trimmed horizon is answered from the file (see
# ObservationHistory), so the margin only decides whether that slow path runs.
MEMORY_SAFETY_MARGIN_SECONDS = HOUR
_TS_KEYS = {MARKET_ROW: "observed_at", THEME_ROW: "recorded_at"}
_NEG_INF = float("-inf")


def default_observations_path() -> str:
    raw = os.getenv("META_OBSERVATIONS_FILE", DEFAULT_OBSERVATIONS_FILE).strip() or DEFAULT_OBSERVATIONS_FILE
    return resolve_data_path(raw)


def default_retention_seconds(tolerance_seconds: float = WINDOW_TOLERANCE_SECONDS) -> float:
    """Max lookback any in-memory lookup needs, plus MEMORY_SAFETY_MARGIN_SECONDS.

    Default: (4 - 1) h + 600 s + 1 h = 15,000 s (4 h 10 min).
    """
    longest = (max(RECONSTRUCTED_WINDOWS.values()) - 1) * HOUR
    return max(longest, HOUR) + float(tolerance_seconds) + MEMORY_SAFETY_MARGIN_SECONDS


class StaleHistoryError(RuntimeError):
    """Already-ingested rows had to be re-read from the file, but the file no
    longer holds the bytes they were parsed from (rewritten in place, rotated,
    truncated or removed). Raised instead of returning different data.

    Bridge: caught by the caller's ``except Exception`` -> candidate skipped
    ("decision shadow failed closed"). Collector: the tick fails and is logged
    by run_forever before anything is appended. Either way the history marks
    itself stale, so the next ``refresh()``/``load()`` does a full reload and
    takes a new fingerprint from the file.
    """


# Name used by the first revision of this change; kept as an alias.
ObservationHistoryChanged = StaleHistoryError


def _new_hasher():
    return hashlib.blake2b(digest_size=_DIGEST_SIZE)


class _Fingerprint:
    """Rolling fingerprint of the consumed byte prefix [0, offset)."""

    __slots__ = ("segment_size", "segments", "_hasher", "partial")

    def __init__(self, segment_size: Optional[int] = None):
        self.segment_size = int(segment_size or _SEGMENT_BYTES)
        self.segments: list[bytes] = []
        self._hasher = _new_hasher()
        self.partial = 0  # bytes hashed into the current (incomplete) segment

    def update(self, data: bytes) -> None:
        size = self.segment_size
        n = len(data)
        if self.partial + n < size:  # fast path: one line inside the current segment
            self._hasher.update(data)
            self.partial += n
            return
        view = memoryview(data)
        while view:
            take = min(len(view), size - self.partial)
            self._hasher.update(view[:take])
            self.partial += take
            view = view[take:]
            if self.partial == size:
                self.segments.append(self._hasher.digest())
                self._hasher = _new_hasher()
                self.partial = 0

    @property
    def length(self) -> int:
        return len(self.segments) * self.segment_size + self.partial

    def copy(self) -> "_Fingerprint":
        other = _Fingerprint(self.segment_size)
        other.segments = list(self.segments)
        other._hasher = self._hasher.copy()
        other.partial = self.partial
        return other

    def pin(self) -> "_Pin":
        # The segment list is only ever appended to (a reset creates a new
        # one), so referencing it with a pinned count is an immutable view.
        return _Pin(self.segment_size, self.segments, len(self.segments),
                    self._hasher.digest(), self.partial)

    def hexdigest(self) -> str:
        h = _new_hasher()
        for seg in self.segments:
            h.update(seg)
        h.update(self._hasher.digest())
        h.update(self.length.to_bytes(8, "big"))
        return h.hexdigest()


class _Pin:
    __slots__ = ("segment_size", "segments", "count", "tail_digest", "tail_len")

    def __init__(self, segment_size: int, segments: list[bytes], count: int,
                 tail_digest: bytes, tail_len: int):
        self.segment_size = segment_size
        self.segments = segments
        self.count = count
        self.tail_digest = tail_digest
        self.tail_len = tail_len

    @property
    def length(self) -> int:
        return self.count * self.segment_size + self.tail_len


def _read_exact(fh, n: int) -> bytes:
    parts = []
    while n > 0:
        chunk = fh.read(n)
        if not chunk:
            break
        parts.append(chunk)
        n -= len(chunk)
    return b"".join(parts)


def _verified_chunks(path: str, pin: _Pin, on_mismatch: Callable[[str], NoReturn]) -> Iterator[bytes]:
    """Yield bytes [0, pin.length) of ``path`` one segment at a time, each only
    after its hash matched the pinned fingerprint. The yielded buffer is the
    one that was hashed, so there is no window between check and use."""
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        on_mismatch(f"{path} no longer exists")
    with fh:
        if os.fstat(fh.fileno()).st_size < pin.length:
            on_mismatch(f"{path} is shorter than the {pin.length} bytes already read")
        for i in range(pin.count):
            buf = _read_exact(fh, pin.segment_size)
            if len(buf) != pin.segment_size or hashlib.blake2b(buf, digest_size=_DIGEST_SIZE).digest() != pin.segments[i]:
                on_mismatch(f"{path}: bytes [{i * pin.segment_size}, {(i + 1) * pin.segment_size}) "
                            "differ from what was ingested")
            yield buf
        if pin.tail_len:
            buf = _read_exact(fh, pin.tail_len)
            if len(buf) != pin.tail_len or hashlib.blake2b(buf, digest_size=_DIGEST_SIZE).digest() != pin.tail_digest:
                start = pin.count * pin.segment_size
                on_mismatch(f"{path}: bytes [{start}, {start + pin.tail_len}) differ from what was ingested")
            yield buf


def _rows_from_chunks(chunks: Iterable[bytes], end: int, split_points: Iterable[int]) -> Iterator[dict[str, Any]]:
    """Same splitting as store.iter_observation_rows (at every newline and at
    each split point, up to ``end``), over an iterator of byte chunks."""
    stops = sorted(p for p in split_points if 0 < p < end)
    stops.append(end)
    stop_iter = iter(stops)
    stop = next(stop_iter)
    carry = b""
    base = 0  # absolute offset of carry[0]
    for chunk in chunks:
        data = carry + chunk if carry else chunk
        i = 0
        while stop is not None:
            limit = stop - base
            if limit > len(data):
                nl = data.find(b"\n", i)
                if nl == -1:
                    break
                cut = nl + 1
            else:
                nl = data.find(b"\n", i, limit)
                cut = nl + 1 if nl != -1 else limit
            row = _parse_line(data[i:cut])
            if row is not None:
                yield row
            i = cut
            if cut == limit:
                stop = next(stop_iter, None)
        carry = data[i:]
        base += i
        if stop is None:
            return


def _row_ts(row: dict[str, Any], key: str) -> Optional[float]:
    # Same expression the lookups use; None when it would raise there.
    try:
        return float(row.get(key) or 0.0)
    except Exception:
        return None


def _immutable(name: str):
    def method(self, *args, **kwargs):
        raise TypeError(f"ObservationRows is an immutable snapshot ({name}() is not supported); "
                        "copy it with list(rows) if you need a mutable list")
    method.__name__ = name
    return method


class ObservationRows(Sequence):
    """Immutable snapshot of every row ingested up to one load()/refresh().

    Returned by ``ObservationHistory.load()``/``refresh()`` instead of the
    list main returned. It holds no rows: it is pinned to the cursor state at
    snapshot time (generation, consumed byte length, row count, split points
    and the consumed-prefix fingerprint).

    Contract:
      * ``len()`` is O(1) and fixed at snapshot time.
      * Iteration re-reads bytes [0, consumed length) of the file, verifying
        each segment against the fingerprint before parsing it, and yields
        exactly the rows ingested at snapshot time. If the file no longer
        holds those bytes (in-place rewrite, rotation, truncation, removal)
        it raises StaleHistoryError; rows yielded before that point come only
        from verified bytes. Bytes appended after the snapshot are ignored.
      * Indexing (negative too), slicing (returns a list), ``in``,
        ``index``/``count``, ``reversed`` and ``==`` against a list or another
        snapshot (element-wise, like list ==) follow from that iteration.
        Like list, it is never equal to a tuple.
      * Mutation (append, extend, insert, pop, remove, clear, sort, reverse,
        item assignment/deletion, +=, *=) raises TypeError.

    Differences from main's list (documented API change):
      * Main returned its internal cache list. That alias grew in place on
        incremental refresh, went silently stale after a full reload, and
        mutating it corrupted the cache. A snapshot never changes after it
        is returned; call ``refresh()``/``load()`` again for newer rows.
      * Element access re-reads the file (O(consumed bytes)); a snapshot is
        not meant for repeated random access. ``list(rows)`` materialises it.
      * ``refresh()``/``load()`` return the same object while nothing new was
        consumed (so ``refresh() is load()`` still holds then).
    """

    __hash__ = None  # type: ignore[assignment]
    __slots__ = ("_history", "_path", "_generation", "_len", "_end", "_splits", "_pin")

    def __init__(self, history: "ObservationHistory"):
        self._history = history
        self._path = history.path
        self._generation = history._generation
        self._len = history._row_count
        self._end = history._offset if history._file_id is not None else 0
        self._splits = tuple(p for p in history._split_points if p < self._end)
        self._pin = history._fp.pin()
        if self._pin.length != self._end:  # invariant: fingerprint covers exactly [0, offset)
            raise AssertionError(f"fingerprint covers {self._pin.length} bytes, cursor {self._end}")

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def consumed_bytes(self) -> int:
        return self._end

    def __len__(self) -> int:
        return self._len

    def _stale(self, reason: str) -> NoReturn:
        self._history._mark_stale(self._generation)
        raise StaleHistoryError(f"{reason}; refresh() or load(force=True) for the current history")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self._end:
            return
        self._history.disk_scans += 1
        count = 0
        for row in _rows_from_chunks(_verified_chunks(self._path, self._pin, self._stale),
                                     self._end, self._splits):
            count += 1
            yield row
        if count != self._len:  # cannot happen when the bytes verified
            self._stale(f"{self._path}: re-read {count} rows, snapshot has {self._len}")

    def __reversed__(self) -> Iterator[dict[str, Any]]:
        return reversed(list(self))

    def __getitem__(self, index):
        if isinstance(index, slice):
            wanted = range(self._len)[index]
            if not wanted:
                return []
            lo, hi = min(wanted[0], wanted[-1]), max(wanted[0], wanted[-1]) + 1
            part = list(islice(self, lo, hi))
            return [part[i - lo] for i in wanted]
        i = operator.index(index)
        if i < 0:
            i += self._len
        if not 0 <= i < self._len:
            raise IndexError("ObservationRows index out of range")
        for row in islice(self, i, i + 1):
            return row
        raise IndexError("ObservationRows index out of range")  # pragma: no cover

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ObservationRows):
            if (other._path == self._path and other._generation == self._generation
                    and other._end == self._end and other._history is self._history):
                return True
        elif not isinstance(other, list):
            return NotImplemented
        if len(other) != self._len:
            return False
        return all(a == b for a, b in zip(self, other))

    def __repr__(self) -> str:
        return (f"ObservationRows(len={self._len}, consumed_bytes={self._end}, "
                f"generation={self._generation}, path={self._path!r})")

    append = _immutable("append")
    extend = _immutable("extend")
    insert = _immutable("insert")
    pop = _immutable("pop")
    remove = _immutable("remove")
    clear = _immutable("clear")
    sort = _immutable("sort")
    reverse = _immutable("reverse")
    __setitem__ = _immutable("__setitem__")
    __delitem__ = _immutable("__delitem__")
    __iadd__ = _immutable("__iadd__")
    __imul__ = _immutable("__imul__")


class ObservationHistory:
    """Observation history with a bounded in-memory footprint.

    Memory holds only: market_snapshot rows and theme_counts rows whose
    timestamp is >= the trim horizon (default ~4h10m before
    min(wall clock, newest row)), rows whose timestamp cannot be parsed, an
    all-time theme -> first-seen index, the total row count, the read cursor
    and the consumed-prefix fingerprint (32 bytes per MiB consumed). signal
    rows are never held (nothing reads them from memory).

    Every public lookup returns exactly what the former keep-every-row
    implementation returned: a lookup whose candidate range lies entirely at
    or above the horizon is answered from memory (rows below the horizon can
    never be within tolerance), anything else re-reads the already-consumed
    bytes of the file, verified against the fingerprint (StaleHistoryError on
    mismatch, never different data). The theme index is the same left fold
    over theme rows in file order, rebuilt on every full reload and extended
    on refresh.
    """

    def __init__(self, path: Optional[str] = None, tolerance_seconds: float = WINDOW_TOLERANCE_SECONDS,
                 *, retention_seconds: Optional[float] = None,
                 clock: Callable[[], float] = time.time):
        self.path = path or default_observations_path()
        self.tolerance = float(tolerance_seconds)
        self.retention = (float(retention_seconds) if retention_seconds is not None
                          else default_retention_seconds(self.tolerance))
        self._clock = clock
        self._loaded = False
        self._generation = 0
        self._view: Optional[ObservationRows] = None
        self.disk_scans = 0
        # Incremental-read cursor: (st_dev, st_ino) of the file we read, the
        # byte offset just past the last consumed complete line, and the first
        # bytes of the file (detects copy-truncate followed by regrowth).
        self._file_id: Optional[tuple[int, int]] = None
        self._offset = 0
        self._head = b""
        self._reset_memory()

    def _reset_memory(self) -> None:
        self._generation += 1
        self._stale = False
        self._fp = _Fingerprint()
        self._row_count = 0
        self._split_points: list[int] = []
        self._kept: dict[str, list[dict[str, Any]]] = {MARKET_ROW: [], THEME_ROW: []}
        self._horizon = {MARKET_ROW: _NEG_INF, THEME_ROW: _NEG_INF}
        self._watermark = {MARKET_ROW: _NEG_INF, THEME_ROW: _NEG_INF}
        self._first_seen: dict[str, float] = {}
        self._first_seen_valid = True

    def _memory_snapshot(self) -> dict[str, Any]:
        # Small: the bounded window lists are copied by reference only.
        return {
            "_generation": self._generation,
            "_stale": self._stale,
            "_fp": self._fp.copy(),
            "_row_count": self._row_count,
            "_split_points": list(self._split_points),
            "_kept": {k: list(v) for k, v in self._kept.items()},
            "_horizon": dict(self._horizon),
            "_watermark": dict(self._watermark),
            "_first_seen": dict(self._first_seen),
            "_first_seen_valid": self._first_seen_valid,
            "_offset": self._offset,
        }

    def _mark_stale(self, generation: int) -> None:
        if generation == self._generation:
            self._stale = True

    def _snapshot(self) -> ObservationRows:
        view = self._view
        if (view is None or view._generation != self._generation or view._len != self._row_count
                or view._end != (self._offset if self._file_id is not None else 0)):
            view = self._view = ObservationRows(self)
        return view

    def fingerprint(self) -> dict[str, Any]:
        """Cursor generation, consumed byte length and prefix digest (hex)."""
        return {"generation": self._generation, "consumed_bytes": self._fp.length,
                "digest": self._fp.hexdigest(), "segments": len(self._fp.segments)}

    # -- storage ------------------------------------------------------------

    def load(self, force: bool = False) -> ObservationRows:
        """Return an immutable snapshot of all rows (see ObservationRows).

        Reads the whole file on first use, with ``force=True``, or after a
        fingerprint mismatch was detected (StaleHistoryError); otherwise
        returns the snapshot of what has been read so far.
        """
        if not self._loaded or force or self._stale:
            self._full_reload()
        return self._snapshot()

    def refresh(self) -> ObservationRows:
        """Pick up rows appended by another process since the last read.

        Reads only new complete lines from the remembered byte offset. A torn
        last line is left until it is completed. If the file was replaced
        (inode change), truncated (size < offset), rewritten (head bytes
        differ) or a fingerprint mismatch was detected earlier, the whole file
        is reloaded and the in-memory window, theme index and fingerprint are
        rebuilt from it. Never interpolates or invents rows. Returns an
        immutable snapshot (see ObservationRows).
        """
        if not self._loaded:
            return self.load()
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            if self._file_id is not None or self._row_count:
                self._reset_memory()
                self._file_id, self._offset, self._head = None, 0, b""
            return self._snapshot()
        if (self._stale or (st.st_dev, st.st_ino) != self._file_id or st.st_size < self._offset
                or not self._head_matches()):
            return self._full_reload()
        if st.st_size > self._offset:
            start = self._offset
            if st.st_size - start <= _REFRESH_LIST_MAX_BYTES:
                rows, offset = read_observations_from(self.path, start)
                if offset > start:
                    consumed = self._read_consumed(start, offset)
                    if consumed is None or not _same_rows(consumed[1], rows):
                        # The bytes changed between the two reads: rebuild
                        # rows and fingerprint from one consistent stream.
                        return self._full_reload()
                    self._fp.update(consumed[0])
                self._offset = offset
                self._note_split(start)
                self._ingest(rows)
            else:
                saved = self._memory_snapshot()
                try:
                    self._offset = stream_observations_from(self.path, start, self._ingest_row,
                                                            on_raw=self._fp.update)
                except BaseException:
                    self.__dict__.update(saved)  # nothing half-applied
                    raise
                self._note_split(start)
                self._trim()
            if len(self._head) < _HEAD_BYTES:
                self._head = self._read_head()
        return self._snapshot()

    def _read_consumed(self, start: int, end: int) -> Optional[tuple[bytes, list[dict[str, Any]]]]:
        """Bytes [start, end) of the file we are reading, and the rows they
        parse to, so the fingerprint gets exactly the bytes behind ``rows``."""
        try:
            with open(self.path, "rb") as fh:
                st = os.fstat(fh.fileno())
                if (st.st_dev, st.st_ino) != self._file_id:
                    return None
                fh.seek(start)
                data = _read_exact(fh, end - start)
        except OSError:
            return None
        if len(data) != end - start:
            return None
        return data, list(iter_observation_rows(io.BytesIO(data), len(data)))

    def _full_reload(self) -> ObservationRows:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            self._loaded = True
            self._reset_memory()
            self._file_id, self._offset, self._head = None, 0, b""
            return self._snapshot()
        saved = self._memory_snapshot()
        self._reset_memory()
        try:
            # Streamed: one parsed row at a time; only window rows are kept.
            # The fingerprint is fed the very bytes the rows came from.
            offset = stream_observations_from(self.path, 0, self._ingest_row, on_raw=self._fp.update)
        except BaseException:
            # Leave the previous (consistent) state untouched on a read error.
            self.__dict__.update(saved)
            raise
        self._loaded = True
        self._file_id = (st.st_dev, st.st_ino)
        self._offset = offset
        self._split_points = []
        self._note_split(0)
        self._trim()
        self._head = self._read_head()
        return self._snapshot()

    def _note_split(self, start: int) -> None:
        # Remember where a read ended on a consumed unterminated JSON tail, so
        # streaming the file later splits there exactly like the reads did.
        if self._offset <= start:
            return
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self._offset - 1)
                last = fh.read(1)
        except OSError:
            return
        if last != b"\n":
            self._split_points.append(self._offset)

    def _ingest(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._ingest_row(row)
        self._trim()

    def _ingest_row(self, row: dict[str, Any]) -> None:
        self._row_count += 1
        kind = row.get("kind")
        if kind == THEME_ROW:
            self._index_theme(row)
            kind = THEME_ROW
        elif kind == MARKET_ROW:
            kind = MARKET_ROW
        else:
            return  # signal and unknown rows: counted, never held
        ts = _row_ts(row, _TS_KEYS[kind])
        if ts is not None and ts < self._horizon[kind]:
            return
        self._kept[kind].append(row)
        if ts is not None and math.isfinite(ts) and ts > self._watermark[kind]:
            self._watermark[kind] = ts
            # Trim as we go so a full reload of a long file never holds more
            # than about one retention window (plus one hour of slack).
            if ts - self._horizon[kind] > self.retention + HOUR:
                self._trim()

    def _index_theme(self, row: dict[str, Any]) -> None:
        # Identical fold to the original theme_first_seen loop body.
        if not self._first_seen_valid:
            return
        first = self._first_seen
        try:
            ts = float(row.get("recorded_at") or 0.0)
            for theme in (row.get("counts") or {}):
                if theme not in first or ts < first[theme]:
                    first[theme] = ts
        except Exception:
            # The original raises on every call; let theme_first_seen re-run
            # it from the file so the same exception surfaces.
            self._first_seen_valid = False
            self._first_seen = {}

    def _trim(self) -> None:
        if not math.isfinite(self.retention):
            return
        now = self._clock()
        for kind, key in _TS_KEYS.items():
            mark = self._watermark[kind]
            if mark == _NEG_INF:
                continue
            cutoff = min(float(now), mark) - self.retention
            if not cutoff > self._horizon[kind]:
                continue
            self._horizon[kind] = cutoff
            kept = []
            for row in self._kept[kind]:
                ts = _row_ts(row, key)
                if ts is None or not ts < cutoff:
                    kept.append(row)
            self._kept[kind] = kept

    def _covered(self, kind: str, target_ts: float, tolerance: float) -> bool:
        """True when no row dropped from memory can be within tolerance.

        Dropped rows have ts < horizon <= target, and IEEE subtraction is
        monotonic, so fl(target - ts) >= fl(target - horizon) > tolerance.
        """
        horizon = self._horizon[kind]
        if horizon == _NEG_INF:
            return True
        try:
            return (float(target_ts) - horizon) > tolerance
        except Exception:
            return False

    def _ensure_loaded(self) -> None:
        # Lookups only load on first use. They never reload on their own after
        # a fingerprint mismatch: that would mix rows read before and after
        # the rewrite within one evaluation. refresh()/load() recover.
        if not self._loaded:
            self._full_reload()

    def _iter_disk(self) -> Iterator[dict[str, Any]]:
        """Rows consumed so far (bytes [0, offset)), verified against the
        consumed-prefix fingerprint; StaleHistoryError on any mismatch."""
        self._ensure_loaded()
        return iter(self._snapshot())

    def _read_head(self) -> bytes:
        try:
            with open(self.path, "rb") as fh:
                return fh.read(min(_HEAD_BYTES, self._offset))
        except OSError:
            return b""

    def _head_matches(self) -> bool:
        if not self._head:
            return True
        try:
            with open(self.path, "rb") as fh:
                return fh.read(len(self._head)) == self._head
        except OSError:
            return False

    def append(self, kind: str, payload: dict[str, Any], recorded_at: Optional[float] = None) -> dict[str, Any]:
        row = {"kind": kind, "recorded_at": recorded_at if recorded_at is not None else time.time()}
        row.update(payload)
        append_observation(self.path, row)
        if self._loaded:
            # Read our own row back through the cursor so the cache and the
            # byte offset stay consistent (no double counting on refresh).
            self.refresh()
        return row

    def append_market_snapshot(self, snap: TokenSnapshot, recorded_at: Optional[float] = None) -> dict[str, Any]:
        return self.append(MARKET_ROW, snap.to_dict(), recorded_at)

    def rows(self, kind: str) -> list[dict[str, Any]]:
        self._ensure_loaded()
        if kind in self._kept and self._horizon[kind] == _NEG_INF:
            return list(self._kept[kind])  # nothing trimmed yet: memory is complete
        return [r for r in self._iter_disk() if r.get("kind") == kind]

    def stats(self) -> dict[str, Any]:
        return {
            "rows_total": self._row_count,
            "market_rows_in_memory": len(self._kept[MARKET_ROW]),
            "theme_rows_in_memory": len(self._kept[THEME_ROW]),
            "themes_indexed": len(self._first_seen),
            "market_horizon": self._horizon[MARKET_ROW],
            "theme_horizon": self._horizon[THEME_ROW],
            "retention_seconds": self.retention,
            "disk_scans": self.disk_scans,
        }

    # -- lookup -------------------------------------------------------------

    @staticmethod
    def _mint_rows(market: list[dict[str, Any]], mint: str) -> list[dict[str, Any]]:
        rows = [r for r in market if r.get("mint") == mint]
        rows.sort(key=lambda r: float(r.get("observed_at") or 0.0))
        return rows

    def market_rows(self, mint: str) -> list[dict[str, Any]]:
        return self._mint_rows(self.rows(MARKET_ROW), mint)

    def nearest_market(self, mint: str, target_ts: float,
                       tolerance: Optional[float] = None) -> Optional[dict[str, Any]]:
        tolerance = self.tolerance if tolerance is None else tolerance
        self._ensure_loaded()
        if self._covered(MARKET_ROW, target_ts, tolerance):
            candidates = self._mint_rows(self._kept[MARKET_ROW], mint)
        else:
            candidates = self.market_rows(mint)
        best = None
        best_gap = None
        for row in candidates:
            gap = abs(float(row.get("observed_at") or 0.0) - target_ts)
            if gap <= tolerance and (best_gap is None or gap < best_gap):
                best, best_gap = row, gap
        return best

    def earliest_observation(self, mint: str) -> Optional[float]:
        rows = self.market_rows(mint)
        return float(rows[0]["observed_at"]) if rows else None

    # -- reconstruction -----------------------------------------------------

    def reconstruct_window(self, snap: TokenSnapshot, now: float, window: str = "4h") -> Optional[MarketWindow]:
        """Sum non-overlapping trailing 1h windows at now, now-1h, ... (hours-1).

        Hour 0 comes from the snapshot being evaluated; every older hour must
        have a stored observation within tolerance and a 1h volume, otherwise
        the window cannot be established and None is returned.
        """
        hours = RECONSTRUCTED_WINDOWS.get(window)
        if not hours:
            return None
        current = snap.market.get("1h")
        if current is None or current.volume_usd is None:
            return None
        volume = float(current.volume_usd)
        buys = current.tx_buys
        sells = current.tx_sells
        sources = [f"live@{snap.observed_at or now:.0f}"]
        for step in range(1, hours):
            row = self.nearest_market(snap.mint, now - step * HOUR)
            if row is None:
                return None
            hour = ((row.get("market") or {}).get("1h") or {})
            hour_volume = hour.get("volume_usd")
            if hour_volume is None:
                return None
            volume += float(hour_volume)
            buys = _sum_opt(buys, hour.get("tx_buys"))
            sells = _sum_opt(sells, hour.get("tx_sells"))
            sources.append(f"history@{float(row.get('observed_at') or 0):.0f}")
        return MarketWindow(volume_usd=volume, tx_buys=buys, tx_sells=sells,
                            source="history_reconstructed:" + ",".join(sources))

    def attach_reconstructed_windows(self, snap: TokenSnapshot, now: float) -> dict[str, bool]:
        coverage: dict[str, bool] = {}
        for window in RECONSTRUCTED_WINDOWS:
            rebuilt = self.reconstruct_window(snap, now, window)
            coverage[window] = rebuilt is not None
            if rebuilt is not None:
                snap.market[window] = rebuilt
        return coverage

    # -- themes -------------------------------------------------------------

    def theme_counts_at(self, target_ts: float, tolerance: Optional[float] = None) -> Optional[dict[str, int]]:
        tolerance = self.tolerance if tolerance is None else tolerance
        self._ensure_loaded()
        if self._covered(THEME_ROW, target_ts, tolerance):
            candidates = self._kept[THEME_ROW]
        else:
            candidates = self.rows(THEME_ROW)
        best = None
        best_gap = None
        for row in candidates:
            gap = abs(float(row.get("recorded_at") or 0.0) - target_ts)
            if gap <= tolerance and (best_gap is None or gap < best_gap):
                best, best_gap = row, gap
        if best is None:
            return None
        counts = best.get("counts") or {}
        return {str(k): int(v) for k, v in counts.items()}

    def theme_first_seen(self) -> dict[str, float]:
        self._ensure_loaded()
        if self._first_seen_valid:
            return dict(self._first_seen)  # all-time index, same fold, file order
        first: dict[str, float] = {}
        for row in self.rows(THEME_ROW):
            ts = float(row.get("recorded_at") or 0.0)
            for theme in (row.get("counts") or {}):
                if theme not in first or ts < first[theme]:
                    first[theme] = ts
        return first

    def theme_acceleration(self, theme: str, count_now: int, now: float, lookback: float = HOUR) -> Score:
        earlier = self.theme_counts_at(now - lookback)
        if earlier is None:
            return unverified("no theme observation within tolerance of the lookback", lookback_seconds=lookback)
        before = earlier.get(theme, 0)
        if before <= 0:
            if count_now <= 0:
                return unverified("theme absent now and at lookback", lookback_seconds=lookback)
            return ok(100.0, "narrative_acceleration = tokens_now / tokens_at_lookback (new theme -> 100)",
                      ["NARRATIVE_NEW_SINCE_LOOKBACK"], tokens_now=count_now, tokens_before=0, lookback_seconds=lookback)
        ratio = count_now / before
        codes = []
        if ratio >= 1.5:
            codes.append("NARRATIVE_ACCELERATING")
        elif ratio < 0.7:
            codes.append("NARRATIVE_DECELERATING")
        value = max(0.0, min(100.0, 50.0 + 25.0 * _log2(ratio)))
        return ok(value, "narrative_acceleration = clip(50 + 25*log2(tokens_now / tokens_at_lookback), 0, 100)",
                  codes, ratio=ratio, tokens_now=count_now, tokens_before=before, lookback_seconds=lookback)


def _same_rows(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    # repr() as the tie-breaker so NaN values (json.loads accepts NaN) compare.
    return len(a) == len(b) and (a == b or repr(a) == repr(b))


def _sum_opt(a: Optional[int], b: Any) -> Optional[int]:
    if a is None or b is None:
        return None
    try:
        return int(a) + int(b)
    except (TypeError, ValueError):
        return None


def _log2(x: float) -> float:
    return math.log(x, 2) if x > 0 else float("-inf")
