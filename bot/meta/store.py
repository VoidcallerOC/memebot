"""Append-only timestamped observation log. JSONL, not a database."""
from __future__ import annotations

import json
import os
from typing import Any, Optional


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

    Returns ``(rows, new_offset)``. Only newline-terminated lines are consumed,
    so a torn last line (a writer mid-append) is left for the next call. The
    one exception is an unterminated tail that already parses as a JSON object:
    a proper prefix of a JSON object line can never itself be a JSON object, so
    such a tail is complete and only its newline is missing (e.g. hand-written
    fixtures); it is consumed so behaviour matches ``load_observations``.
    Undecodable complete lines are skipped, exactly like ``load_observations``.
    """
    rows: list[dict[str, Any]] = []
    with open(path, "rb") as fh:
        fh.seek(offset)
        data = fh.read()
    if not data:
        return rows, offset
    end = data.rfind(b"\n")
    complete = data[: end + 1] if end >= 0 else b""
    tail = data[end + 1:] if end >= 0 else data
    for raw in complete.split(b"\n"):
        row = _parse_line(raw)
        if row is not None:
            rows.append(row)
    consumed = len(complete)
    if tail:
        row = _parse_line(tail)
        if row is not None:
            rows.append(row)
            consumed += len(tail)
    return rows, offset + consumed
