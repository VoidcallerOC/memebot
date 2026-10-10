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
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from typing import Any, Callable, Iterator, Optional

from .model import MarketWindow, TokenSnapshot, Score, ok, unverified
from .store import (append_observation, iter_observation_rows, read_observations_from,
                    stream_observations_from)
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


class ObservationHistoryChanged(RuntimeError):
    """The file changed under the cursor and a lookup needed rows that are no
    longer held in memory. Raised instead of answering from different data."""


def _row_ts(row: dict[str, Any], key: str) -> Optional[float]:
    # Same expression the lookups use; None when it would raise there.
    try:
        return float(row.get(key) or 0.0)
    except Exception:
        return None


class ObservationRows(Sequence):
    """All rows read so far (file order), without holding them in memory.

    ``len()`` is O(1). Iteration streams the bytes the cursor has consumed
    from the file, so it yields exactly the rows the reads returned. Returned
    by ``load()``/``refresh()`` in place of the former full list.
    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, history: "ObservationHistory"):
        self._history = history

    def __len__(self) -> int:
        return self._history._row_count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self._history._iter_disk()

    def __getitem__(self, index):
        return list(self)[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple, ObservationRows)):
            return list(self) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"ObservationRows(len={len(self)}, path={self._history.path!r})"


class ObservationHistory:
    """Observation history with a bounded in-memory footprint.

    Memory holds only: market_snapshot rows and theme_counts rows whose
    timestamp is >= the trim horizon (default ~4h10m before
    min(wall clock, newest row)), rows whose timestamp cannot be parsed, an
    all-time theme -> first-seen index, the total row count and the read
    cursor. signal rows are never held (nothing reads them from memory).

    Every public lookup returns exactly what the former keep-every-row
    implementation returned: a lookup whose candidate range lies entirely at
    or above the horizon is answered from memory (rows below the horizon can
    never be within tolerance), anything else streams the already-consumed
    bytes of the file. The theme index is the same left fold over theme rows
    in file order, rebuilt on every full reload and extended on refresh.
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
        self._view = ObservationRows(self)
        self.disk_scans = 0
        # Incremental-read cursor: (st_dev, st_ino) of the file we read, the
        # byte offset just past the last consumed complete line, and the first
        # bytes of the file (detects copy-truncate followed by regrowth).
        self._file_id: Optional[tuple[int, int]] = None
        self._offset = 0
        self._head = b""
        self._reset_memory()

    def _reset_memory(self) -> None:
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
            "_row_count": self._row_count,
            "_split_points": list(self._split_points),
            "_kept": {k: list(v) for k, v in self._kept.items()},
            "_horizon": dict(self._horizon),
            "_watermark": dict(self._watermark),
            "_first_seen": dict(self._first_seen),
            "_first_seen_valid": self._first_seen_valid,
            "_offset": self._offset,
        }

    # -- storage ------------------------------------------------------------

    def load(self, force: bool = False) -> ObservationRows:
        if not self._loaded or force:
            self._full_reload()
        return self._view

    def refresh(self) -> ObservationRows:
        """Pick up rows appended by another process since the last read.

        Reads only new complete lines from the remembered byte offset. A torn
        last line is left until it is completed. If the file was replaced
        (inode change), truncated (size < offset) or rewritten (head bytes
        differ), the whole file is reloaded and the in-memory window and theme
        index are rebuilt from it. Never interpolates or invents rows.
        """
        if not self._loaded:
            return self.load()
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            if self._file_id is not None or self._row_count:
                self._reset_memory()
                self._file_id, self._offset, self._head = None, 0, b""
            return self._view
        if ((st.st_dev, st.st_ino) != self._file_id or st.st_size < self._offset
                or not self._head_matches()):
            return self._full_reload()
        if st.st_size > self._offset:
            start = self._offset
            if st.st_size - start <= _REFRESH_LIST_MAX_BYTES:
                rows, self._offset = read_observations_from(self.path, self._offset)
                self._note_split(start)
                self._ingest(rows)
            else:
                saved = self._memory_snapshot()
                try:
                    self._offset = stream_observations_from(self.path, start, self._ingest_row)
                except BaseException:
                    self.__dict__.update(saved)  # nothing half-applied
                    raise
                self._note_split(start)
                self._trim()
            if len(self._head) < _HEAD_BYTES:
                self._head = self._read_head()
        return self._view

    def _full_reload(self) -> ObservationRows:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            self._loaded = True
            self._reset_memory()
            self._file_id, self._offset, self._head = None, 0, b""
            return self._view
        saved = self._memory_snapshot()
        self._reset_memory()
        try:
            # Streamed: one parsed row at a time; only window rows are kept.
            offset = stream_observations_from(self.path, 0, self._ingest_row)
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
        return self._view

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

    def _iter_disk(self) -> Iterator[dict[str, Any]]:
        """Stream exactly the rows consumed so far (bytes [0, offset))."""
        if not self._loaded:
            self.load()
        if self._file_id is None:
            return
        self.disk_scans += 1
        try:
            fh = open(self.path, "rb")
        except FileNotFoundError as exc:
            raise ObservationHistoryChanged(f"{self.path} disappeared; refresh() first") from exc
        with fh:
            st = os.fstat(fh.fileno())
            if (st.st_dev, st.st_ino) != self._file_id or st.st_size < self._offset \
                    or fh.read(len(self._head)) != self._head:
                raise ObservationHistoryChanged(f"{self.path} changed since it was read; refresh() first")
            fh.seek(0)
            yield from iter_observation_rows(fh, self._offset, self._split_points)

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
        self.load()
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
        self.load()
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
        self.load()
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
        self.load()
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


def _sum_opt(a: Optional[int], b: Any) -> Optional[int]:
    if a is None or b is None:
        return None
    try:
        return int(a) + int(b)
    except (TypeError, ValueError):
        return None


def _log2(x: float) -> float:
    return math.log(x, 2) if x > 0 else float("-inf")
