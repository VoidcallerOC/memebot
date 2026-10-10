"""Append-only timestamped observation log. JSONL, not a database."""
from __future__ import annotations

import json
import os
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Optional


def append_observation(path: str, payload: dict[str, Any]) -> None:
    """Append one JSON line and make it durable (flush + fsync) before returning."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_observations(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _parse_line(raw: bytes) -> Optional[dict[str, Any]]:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def read_observations_from(path: str, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read complete rows starting at byte ``offset``.

    Returns ``(rows, new_offset)``. The file is streamed line by line from
    ``offset`` (binary ``seek`` + ``readline``), so only the newly appended
    bytes are read and at most one line is buffered at a time; the remaining
    file is never loaded into memory as a whole. This is also the full-reload
    path (``offset == 0``).

    Only newline-terminated lines are consumed, so a torn last line (a writer
    mid-append) is left for the next call. The one exception is an
    unterminated tail that already parses as a JSON object: a proper prefix of
    a JSON object line can never itself be a JSON object, so such a tail is
    complete and only its newline is missing (e.g. hand-written fixtures); it
    is consumed so behaviour matches ``load_observations``. Undecodable
    complete lines are skipped, exactly like ``load_observations``.
    """
    rows: list[dict[str, Any]] = []
    consumed = stream_observations_from(path, offset, rows.append)
    return rows, consumed


def stream_observations_from(path: str, offset: int, on_row: Callable[[dict[str, Any]], None],
                             on_raw: Optional[Callable[[bytes], None]] = None) -> int:
    """Same consumption rules as ``read_observations_from``, but hands each row
    to ``on_row`` instead of building a list (bounded memory on full reloads).
    Returns the new offset.

    ``on_raw``, when given, receives exactly the bytes that are consumed (every
    complete line, parsed or skipped, and a consumed unterminated JSON tail),
    in file order, from the same read that produced the rows: the
    concatenation of everything passed to it is bytes [offset, return value).
    """
    consumed = offset
    with open(path, "rb") as fh:
        fh.seek(offset)
        for raw in iter(fh.readline, b""):
            if raw.endswith(b"\n"):
                consumed += len(raw)
                if on_raw is not None:
                    on_raw(raw)
                row = _parse_line(raw)
                if row is not None:
                    on_row(row)
                continue
            # Unterminated tail: only ever the last chunk before EOF.
            row = _parse_line(raw)
            if row is not None:
                if on_raw is not None:
                    on_raw(raw)
                on_row(row)
                consumed += len(raw)
            break
    return consumed


def iter_observation_rows(fh: BinaryIO, end: int, split_points: Iterable[int] = ()) -> Iterator[dict[str, Any]]:
    """Yield the rows a sequence of ``read_observations_from`` calls produced.

    ``fh`` is an open binary handle positioned at byte 0; only bytes before
    ``end`` are read (bytes appended after the caller's cursor are ignored).
    Lines split at every newline and additionally at each offset in
    ``split_points`` (positions where an unterminated, JSON-object tail was
    consumed by an earlier read), so the result is row-for-row identical to
    what the incremental reads returned. Streams one line at a time.
    """
    stops = sorted(p for p in split_points if 0 < p < end)
    stops.append(end)
    consumed = 0
    for stop in stops:
        while consumed < stop:
            raw = fh.readline(stop - consumed)
            if not raw:
                return
            consumed += len(raw)
            row = _parse_line(raw)
            if row is not None:
                yield row

