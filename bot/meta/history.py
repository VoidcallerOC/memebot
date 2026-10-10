"""Historical observation support on top of the JSONL store.

Reconstructed implementation. Rows are appended with a ``kind``:
  * ``market_snapshot`` — raw TokenSnapshot components as observed
  * ``signal``          — scored per-token summary
  * ``theme_counts``    — tokens per narrative theme at that evaluation

A longer window (4h) is reconstructed only when every hourly trailing 1h
observation it needs exists within tolerance. Nothing is interpolated: a gap
means the window stays UNVERIFIED.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

from .model import MarketWindow, TokenSnapshot, Score, ok, unverified
from .store import append_observation, read_observations_from
from ..paths import resolve_data_path

MARKET_ROW = "market_snapshot"
SIGNAL_ROW = "signal"
THEME_ROW = "theme_counts"

DEFAULT_OBSERVATIONS_FILE = "meta_observations.jsonl"
HOUR = 3600.0
WINDOW_TOLERANCE_SECONDS = 600.0
RECONSTRUCTED_WINDOWS = {"4h": 4}
_HEAD_BYTES = 64


def default_observations_path() -> str:
    raw = os.getenv("META_OBSERVATIONS_FILE", DEFAULT_OBSERVATIONS_FILE).strip() or DEFAULT_OBSERVATIONS_FILE
    return resolve_data_path(raw)


class ObservationHistory:
    def __init__(self, path: Optional[str] = None, tolerance_seconds: float = WINDOW_TOLERANCE_SECONDS):
        self.path = path or default_observations_path()
        self.tolerance = float(tolerance_seconds)
        self._rows: Optional[list[dict[str, Any]]] = None
        # Incremental-read cursor: (st_dev, st_ino) of the file we read, the
        # byte offset just past the last consumed complete line, and the first
        # bytes of the file (detects copy-truncate followed by regrowth).
        self._file_id: Optional[tuple[int, int]] = None
        self._offset = 0
        self._head = b""

    # -- storage ------------------------------------------------------------

    def load(self, force: bool = False) -> list[dict[str, Any]]:
        if self._rows is None or force:
            self._full_reload()
        return self._rows  # type: ignore[return-value]

    def refresh(self) -> list[dict[str, Any]]:
        """Pick up rows appended by another process since the last read.

        Reads only new complete lines from the remembered byte offset. A torn
        last line is left until it is completed. If the file was replaced
        (inode change), truncated (size < offset) or rewritten (head bytes
        differ), the whole file is reloaded. Never interpolates or invents rows.
        """
        if self._rows is None:
            return self.load()
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            if self._file_id is not None or self._rows:
                self._rows = []
                self._file_id, self._offset, self._head = None, 0, b""
            return self._rows
        if ((st.st_dev, st.st_ino) != self._file_id or st.st_size < self._offset
                or not self._head_matches()):
            return self._full_reload()
        if st.st_size > self._offset:
            rows, self._offset = read_observations_from(self.path, self._offset)
            self._rows.extend(rows)
            if len(self._head) < _HEAD_BYTES:
                self._head = self._read_head()
        return self._rows

    def _full_reload(self) -> list[dict[str, Any]]:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            self._rows = []
            self._file_id, self._offset, self._head = None, 0, b""
            return self._rows
        rows, offset = read_observations_from(self.path, 0)
        self._rows = rows
        self._file_id = (st.st_dev, st.st_ino)
        self._offset = offset
        self._head = self._read_head()
        return self._rows

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
        if self._rows is not None:
            # Read our own row back through the cursor so the cache and the
            # byte offset stay consistent (no double counting on refresh).
            self.refresh()
        return row

    def append_market_snapshot(self, snap: TokenSnapshot, recorded_at: Optional[float] = None) -> dict[str, Any]:
        return self.append(MARKET_ROW, snap.to_dict(), recorded_at)

    def rows(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.load() if r.get("kind") == kind]

    # -- lookup -------------------------------------------------------------

    def market_rows(self, mint: str) -> list[dict[str, Any]]:
        rows = [r for r in self.rows(MARKET_ROW) if r.get("mint") == mint]
        rows.sort(key=lambda r: float(r.get("observed_at") or 0.0))
        return rows

    def nearest_market(self, mint: str, target_ts: float,
                       tolerance: Optional[float] = None) -> Optional[dict[str, Any]]:
        tolerance = self.tolerance if tolerance is None else tolerance
        best = None
        best_gap = None
        for row in self.market_rows(mint):
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
        best = None
        best_gap = None
        for row in self.rows(THEME_ROW):
            gap = abs(float(row.get("recorded_at") or 0.0) - target_ts)
            if gap <= tolerance and (best_gap is None or gap < best_gap):
                best, best_gap = row, gap
        if best is None:
            return None
        counts = best.get("counts") or {}
        return {str(k): int(v) for k, v in counts.items()}

    def theme_first_seen(self) -> dict[str, float]:
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
