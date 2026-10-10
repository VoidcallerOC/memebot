"""PR3: bounded in-memory observation history (windowed market/theme rows +
all-time theme first-seen index) must be behaviour-identical to main.

Every test uses synthetic rows in tmp_path only. No network (sessions raise),
no bot.main / bot.meta process, no real market data. The reference
implementation is a verbatim copy of origin/main's ObservationHistory
(tests/_pr3_baseline_history_main.py); differential tests run it and the new
code on identical files and compare every lookup, feature and decision.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import pytest

from bot.config import Config
from bot.decision.bridge import DecisionShadowBridge
from bot.decision.models.logistic import LogisticModel
from bot.decision.schema import N_FEATURES
from bot.meta import history as history_module
from bot.meta.detector import MetaDetector
from bot.meta.history import (
    HOUR, MARKET_ROW, SIGNAL_ROW, THEME_ROW, ObservationHistory, ObservationHistoryChanged,
    default_retention_seconds,
)
from bot.meta.model import MarketWindow, TokenSnapshot
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.safety import TokenSafety

_spec = importlib.util.spec_from_file_location(
    "_pr3_baseline_history_main", Path(__file__).with_name("_pr3_baseline_history_main.py"))
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
BaselineHistory = _baseline.BaselineObservationHistory

T0 = 1_700_300_000.0
TICK = 300.0
THEME_POOL = ["ai_agents", "animals", "politics", "celebrity", "gaming", "food", "space",
              "sports", "music", "defi", "rwa"]


class _NoNetworkSession:
    def __getattr__(self, name):
        raise AssertionError(f"network access attempted via session.{name}")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("MEMEBOT_DATA_DIR", "META_OBSERVATIONS_FILE", "META_PROCESS_LOCK_FILE",
                 "DECISION_SHADOW_FILE", "STATE_FILE", "PROCESS_LOCK_FILE"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# synthetic stream
# ---------------------------------------------------------------------------

def _mint(i: int) -> str:
    return f"Pr3Mint{i:02d}" + "1" * 35


def _market(ts: float, mint: str, rng: random.Random, observed: object = None) -> dict:
    return {
        "kind": MARKET_ROW, "recorded_at": ts,
        "observed_at": ts if observed is None else observed,
        "mint": mint, "symbol": mint[:6], "price_usd": round(rng.uniform(0.001, 1.0), 6),
        "liquidity_usd": round(rng.uniform(5e3, 2e5), 2),
        "market": {
            "5m": {"volume_usd": round(rng.uniform(10, 5e3), 2), "tx_buys": rng.randint(0, 40),
                   "tx_sells": rng.randint(0, 40), "price_change_pct": rng.uniform(-5, 5), "source": "dexscreener"},
            "1h": {"volume_usd": (None if rng.random() < 0.03 else round(rng.uniform(100, 5e4), 2)),
                   "tx_buys": (None if rng.random() < 0.05 else rng.randint(0, 400)),
                   "tx_sells": rng.randint(0, 400), "price_change_pct": rng.uniform(-9, 9), "source": "dexscreener"},
        },
    }


def make_stream(seed: int, *, hours: float, start: float = T0, mints: int = 4, gap_prob: float = 0.0,
                dup_prob: float = 0.0, ooo_prob: float = 0.0, jitter: float = 0.0, junk: bool = False,
                themes_per_day: int = 0, theme_pool=THEME_POOL) -> list[tuple[float, bytes]]:
    """Collector-like rows: per tick N market rows, N signal rows, 1 theme row.

    Returns [(tick_ts, line_bytes)] in file order. Options inject gaps (whole
    ticks missing), duplicated rows, out-of-order (late, older-timestamped)
    rows, observed_at jitter, junk lines and per-day new themes.
    """
    rng = random.Random(seed)
    out: list[tuple[float, bytes]] = []
    late: list[bytes] = []
    ticks = int(hours * 3600 / TICK)
    pool = list(theme_pool)
    for k in range(ticks):
        t = start + k * TICK
        if themes_per_day and k % int(86400 / TICK) == 0:
            day = int(k * TICK // 86400)
            pool = pool + [f"day{day}_theme{j}" for j in range(themes_per_day)]
        if rng.random() < gap_prob:
            continue
        lines = []
        for i in range(mints):
            j = jitter * (rng.random() * 2 - 1) if jitter else 0.0
            lines.append(json.dumps(_market(t, _mint(i), rng, observed=t + j if j else None)))
            lines.append(json.dumps({"kind": SIGNAL_ROW, "recorded_at": t, "token": _mint(i),
                                     "narratives": rng.sample(pool, 2)}))
        chosen = rng.sample(pool, rng.randint(0, min(4, len(pool))))
        theme_ts = t - (rng.uniform(0, 7200) if rng.random() < ooo_prob else 0.0)
        lines.append(json.dumps({"kind": THEME_ROW, "recorded_at": theme_ts,
                                 "counts": {th: rng.randint(1, 6) for th in chosen}}))
        if rng.random() < ooo_prob:  # a late market row with an older timestamp
            late.append(json.dumps(_market(t - rng.uniform(600, 4 * 3600), _mint(rng.randrange(mints)), rng)).encode())
        for line in lines:
            out.append((t, line.encode() + b"\n"))
            if rng.random() < dup_prob:
                out.append((t, line.encode() + b"\n"))
        if late and rng.random() < 0.5:
            out.append((t, late.pop() + b"\n"))
        if junk and rng.random() < 0.02:
            out.append((t, rng.choice([b"\n", b"{not json}\n", b"[1,2]\n", b"\"str\"\n",
                                       json.dumps({"kind": MARKET_ROW, "mint": _mint(0)}).encode() + b"\n",
                                       json.dumps({"kind": THEME_ROW, "counts": {"zero_ts": 1}}).encode() + b"\n",
                                       json.dumps({"kind": "other", "recorded_at": t}).encode() + b"\n"])))
    return out


def _append(path: Path, chunk: list[tuple[float, bytes]]) -> None:
    with path.open("ab") as fh:
        for _, line in chunk:
            fh.write(line)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
    except Exception as exc:  # compare failure modes too
        return ("raised", type(exc).__name__)
    if isinstance(value, MarketWindow):
        return ("window", value.to_dict())
    if hasattr(value, "to_dict"):
        return ("obj", value.to_dict())
    if isinstance(value, dict):
        return ("dict", list(value.items()))
    return ("value", value)


def _snap(mint: str, now: float, seed: int = 0) -> TokenSnapshot:
    rng = random.Random(f"{mint}{now}{seed}")
    snap = TokenSnapshot(
        mint=mint, symbol=mint[:6], name=f"{mint[:6]} ai agent dog", observed_at=now,
        price_usd=round(rng.uniform(0.001, 1), 6), liquidity_usd=round(rng.uniform(2e4, 2e5), 2),
        market_cap_usd=round(rng.uniform(1e5, 5e6), 2), pair_created_at=now - 86_400.0,
        market={
            "5m": MarketWindow(volume_usd=round(rng.uniform(50, 6e3), 2), tx_buys=rng.randint(1, 40),
                               tx_sells=rng.randint(1, 40), price_change_pct=rng.uniform(-5, 5), source="dexscreener"),
            "1h": MarketWindow(volume_usd=round(rng.uniform(500, 5e4), 2), tx_buys=rng.randint(1, 300),
                               tx_sells=rng.randint(1, 300), price_change_pct=rng.uniform(-9, 9), source="dexscreener"),
        },
        source="dexscreener",
    )
    snap.top_holder_pct = round(rng.uniform(3, 30), 2)   # present -> no holder RPC
    snap.top10_holder_pct = round(rng.uniform(15, 60), 2)
    return snap


def compare_histories(new: ObservationHistory, base, now: float, mints: int, rng: random.Random,
                      full_rows: bool = False) -> list:
    """Every lookup at production-like and arbitrary (old) query times."""
    diffs = []

    def check(label, a, b):
        if a != b:
            diffs.append((label, a, b))

    queries = [now, now - rng.uniform(0, 2 * HOUR), now - rng.uniform(0, 3 * 86400), now + rng.uniform(0, 600)]
    for q in queries:
        for i in range(mints):
            mint = _mint(i)
            check(("reconstruct", q, mint),
                  _outcome(new.reconstruct_window, _snap(mint, q), q),
                  _outcome(base.reconstruct_window, _snap(mint, q), q))
            for step in (1, 2, 3):
                check(("nearest", q, mint, step), _outcome(new.nearest_market, mint, q - step * HOUR),
                      _outcome(base.nearest_market, mint, q - step * HOUR))
        check(("theme_at", q), _outcome(new.theme_counts_at, q - HOUR), _outcome(base.theme_counts_at, q - HOUR))
        for theme in rng.sample(THEME_POOL, 3):
            count_now = rng.randint(0, 5)
            check(("accel", q, theme, count_now), _outcome(new.theme_acceleration, theme, count_now, q),
                  _outcome(base.theme_acceleration, theme, count_now, q))
    # Probe right at the trim horizon, where memory vs file is decided.
    for kind, lookup in ((MARKET_ROW, None), (THEME_ROW, None)):
        horizon = new.stats()["market_horizon" if kind == MARKET_ROW else "theme_horizon"]
        if not math.isfinite(horizon):
            continue
        for _ in range(12):
            target = horizon + rng.uniform(-700.0, 1400.0)
            if kind == MARKET_ROW:
                for i in range(mints):
                    check(("edge_nearest", target, i), _outcome(new.nearest_market, _mint(i), target),
                          _outcome(base.nearest_market, _mint(i), target))
            else:
                check(("edge_theme", target), _outcome(new.theme_counts_at, target),
                      _outcome(base.theme_counts_at, target))
    check("first_seen", _outcome(new.theme_first_seen), _outcome(base.theme_first_seen))
    check("len", len(new.load()), len(base.load()))
    if full_rows:
        check("rows", list(new.load()), list(base.load()))
        for kind in (MARKET_ROW, THEME_ROW, SIGNAL_ROW):
            check(("rows", kind), new.rows(kind), base.rows(kind))
        check("earliest", _outcome(new.earliest_observation, _mint(0)),
              _outcome(base.earliest_observation, _mint(0)))
    return diffs


# ---------------------------------------------------------------------------
# 1. differential: randomized streams, fixed seeds, incremental refresh
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("retention", [None, 1800.0, 0.0])
def test_differential_randomized_streams_match_main(tmp_path, seed, retention):
    """None = production default (4h10m); 1800 s and 0 s are deliberately below
    the needed lookback so the exact from-file path is exercised as well."""
    rng = random.Random(1000 + seed)
    stream = make_stream(seed, hours=20, mints=3, gap_prob=0.05, dup_prob=0.03, ooo_prob=0.1,
                         jitter=120.0 if seed % 2 else 0.0, junk=seed % 3 == 0, themes_per_day=2)
    path = tmp_path / "obs.jsonl"
    path.write_bytes(b"")
    clock_now = [T0]
    new = ObservationHistory(str(path), retention_seconds=retention, clock=lambda: clock_now[0])
    base = BaselineHistory(str(path))
    new.load(), base.load()
    cursor = 0
    checkpoints = 0
    chunks = 0
    while cursor < len(stream):
        chunks += 1
        step = rng.randint(1, 400)
        chunk = stream[cursor:cursor + step]
        cursor += step
        _append(path, chunk)
        clock_now[0] = chunk[-1][0]
        new.refresh(), base.refresh()
        if chunks % 4 == 1 or cursor >= len(stream):
            checkpoints += 1
            diffs = compare_histories(new, base, chunk[-1][0], 3, rng, full_rows=cursor >= len(stream))
            assert diffs == [], diffs[:5]
    assert checkpoints >= 3
    if retention is None:
        # bounded: never more than ~(retention + 1h slack) of market rows held
        assert new.stats()["market_rows_in_memory"] <= 3 * 2 * ((default_retention_seconds() + HOUR) / TICK + 60)


def test_late_rows_just_above_horizon_are_kept(tmp_path):
    path = tmp_path / "obs.jsonl"
    rows = [_market(T0 + k * TICK, _mint(0), random.Random(k)) for k in range(60)]
    _append(path, [(0, (json.dumps(r) + "\n").encode()) for r in rows])
    now = T0 + 59 * TICK
    h = ObservationHistory(str(path), retention_seconds=3600.0, clock=lambda: now)
    h.load()
    horizon = h.stats()["market_horizon"]
    assert horizon == now - 3600.0
    late = _market(horizon + 100.0, _mint(0), random.Random(99))
    late["market"]["1h"]["volume_usd"] = 123.0
    _append(path, [(0, (json.dumps(late) + "\n").encode())])
    h.refresh()
    base = BaselineHistory(str(path))
    for target in (horizon + 650.0, horizon + 101.0, horizon - 100.0):
        assert _outcome(h.nearest_market, _mint(0), target) == _outcome(base.nearest_market, _mint(0), target)
    assert h.nearest_market(_mint(0), horizon + 650.0)["market"]["1h"]["volume_usd"] in (
        123.0, base.nearest_market(_mint(0), horizon + 650.0)["market"]["1h"]["volume_usd"])


def test_production_queries_never_touch_the_file(tmp_path):
    """At query time = collector clock, every lookup is served from memory."""
    stream = make_stream(3, hours=36, mints=3, themes_per_day=1)
    path = tmp_path / "obs.jsonl"
    _append(path, stream)
    clock = [stream[-1][0]]
    new = ObservationHistory(str(path), clock=lambda: clock[0])
    new.load()
    now = stream[-1][0] + TICK
    assert new.stats()["market_horizon"] > T0  # really trimmed
    for i in range(3):
        new.reconstruct_window(_snap(_mint(i), now), now)
    new.theme_acceleration("animals", 2, now)
    new.theme_first_seen()
    assert new.disk_scans == 0


# ---------------------------------------------------------------------------
# 2. theme first-seen: all-time equality old/new observations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_theme_first_seen_equals_full_history(tmp_path, seed):
    rng = random.Random(seed)
    path = tmp_path / "obs.jsonl"
    path.write_bytes(b"")
    clock = [T0]
    new = ObservationHistory(str(path), retention_seconds=rng.choice([None, 600.0, 0.0]), clock=lambda: clock[0])
    new.load()
    seen_order: dict[str, float] = {}
    t = T0
    for _ in range(rng.randint(200, 900)):
        t += rng.uniform(0, 900)
        ts = t - (rng.uniform(0, 5 * 86400) if rng.random() < 0.08 else 0.0)  # very late rows
        counts = {rng.choice(THEME_POOL + [f"x{rng.randint(0, 60)}"]): 1 for _ in range(rng.randint(0, 3))}
        row = {"kind": THEME_ROW, "recorded_at": ts, "counts": counts}
        _append(path, [(t, (json.dumps(row) + "\n").encode())])
        for th in counts:
            if th not in seen_order or ts < seen_order[th]:
                seen_order[th] = ts
        clock[0] = t
        if rng.random() < 0.3:
            new.refresh()
    new.refresh()
    fresh_base = BaselineHistory(str(path))
    got = new.theme_first_seen()
    assert list(got.items()) == list(fresh_base.theme_first_seen().items()) == list(seen_order.items())
    assert new.disk_scans == 0  # served by the index, not by re-reading the file


def test_theme_first_seen_malformed_rows_fail_like_main(tmp_path):
    path = tmp_path / "obs.jsonl"
    good = {"kind": THEME_ROW, "recorded_at": T0, "counts": {"a": 1}}
    for bad in ({"kind": THEME_ROW, "recorded_at": "not-a-number", "counts": {"b": 1}},
                {"kind": THEME_ROW, "recorded_at": T0 + 1, "counts": [["unhashable"]]},
                {"kind": THEME_ROW, "recorded_at": T0 + 2, "counts": ["list", "themes"]},
                {"kind": THEME_ROW, "recorded_at": None, "counts": {"zero": 1}},
                {"kind": THEME_ROW, "recorded_at": float("nan"), "counts": {"nan": 1}}):
        path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
        new, base = ObservationHistory(str(path), retention_seconds=0.0), BaselineHistory(str(path))
        assert _outcome(new.theme_first_seen) == _outcome(base.theme_first_seen), bad
        assert _outcome(new.theme_counts_at, T0, 10.0) == _outcome(base.theme_counts_at, T0, 10.0), bad


# ---------------------------------------------------------------------------
# 3. 4h availability over time; startup vs refresh; missing/stale/gapped
# ---------------------------------------------------------------------------

def _availability(tmp_path, *, pre_history_h: float, checkpoints: dict, gap: tuple = ()):
    path = tmp_path / "obs.jsonl"
    stream = make_stream(11, hours=pre_history_h + max(checkpoints.values()) / 3600 + 1,
                         start=T0 - pre_history_h * 3600, mints=2)
    if gap:
        stream = [(t, line) for t, line in stream if not (gap[0] <= t < gap[1])]
    clock = [T0]
    reader = ObservationHistory(str(path), clock=lambda: clock[0])
    _append(path, [x for x in stream if x[0] <= T0])
    reader.load()  # bridge starts at T0
    written = T0
    out = {}
    for label, offset in checkpoints.items():
        now = T0 + offset
        _append(path, [x for x in stream if written < x[0] <= now])
        written = now
        clock[0] = now
        reader.refresh()
        base = BaselineHistory(str(path))  # what main's (fresh, full) cache says
        got = reader.reconstruct_window(_snap(_mint(0), now), now)
        want = base.reconstruct_window(_snap(_mint(0), now), now)
        assert _outcome(lambda: got) == _outcome(lambda: want), label
        startup = ObservationHistory(str(path), clock=lambda: now).reconstruct_window(_snap(_mint(0), now), now)
        assert _outcome(lambda: startup) == _outcome(lambda: want), label  # startup == refresh
        out[label] = got is not None
    return out


def test_4h_window_availability_with_pre_history(tmp_path):
    marks = {"+75min": 75 * 60, "+3h": 3 * 3600, "+6h": 6 * 3600, "+1d": 86400,
             "+3d": 3 * 86400, "+7d": 7 * 86400}
    assert _availability(tmp_path, pre_history_h=3.25, checkpoints=marks) == {k: True for k in marks}


def test_4h_window_needs_three_hours_of_history(tmp_path):
    marks = {"+75min": 75 * 60, "+2h": 2 * 3600, "+3h": 3 * 3600, "+6h": 6 * 3600}
    got = _availability(tmp_path, pre_history_h=0.0, checkpoints=marks)
    assert got == {"+75min": False, "+2h": False, "+3h": True, "+6h": True}


def test_4h_window_absent_across_gap_and_when_stale(tmp_path):
    # 80-minute collector outage: windows needing a point inside it are absent
    marks = {"+5h": 5 * 3600, "+6h": 6 * 3600, "+8h": 8 * 3600}
    got = _availability(tmp_path, pre_history_h=4, checkpoints=marks,
                        gap=(T0 + 3 * 3600 - 2400, T0 + 3 * 3600 + 2400))
    assert got == {"+5h": False, "+6h": False, "+8h": True}
    # stale: collector stopped; nothing newer than 5h -> no window, same as main
    path = tmp_path / "stale.jsonl"
    _append(path, make_stream(5, hours=6, start=T0 - 11 * 3600, mints=1))
    now = T0
    new = ObservationHistory(str(path), clock=lambda: now)
    assert new.reconstruct_window(_snap(_mint(0), now), now) is None
    assert BaselineHistory(str(path)).reconstruct_window(_snap(_mint(0), now), now) is None
    # missing file
    assert ObservationHistory(str(tmp_path / "nope.jsonl")).reconstruct_window(_snap(_mint(0), now), now) is None


def test_duplicates_and_out_of_order_rows_match_main(tmp_path):
    path = tmp_path / "obs.jsonl"
    stream = make_stream(21, hours=12, mints=2, dup_prob=0.3, ooo_prob=0.5, jitter=300.0)
    rng = random.Random(5)
    rng.shuffle(stream)  # fully out of order on disk
    _append(path, stream)
    now = T0 + 12 * 3600
    new = ObservationHistory(str(path), clock=lambda: now)
    base = BaselineHistory(str(path))
    assert compare_histories(new, base, now, 2, random.Random(1), full_rows=True) == []


# ---------------------------------------------------------------------------
# 4. disk bytes never change; rotation / truncation / torn tails
# ---------------------------------------------------------------------------

def test_trimming_never_changes_file_bytes(tmp_path):
    path = tmp_path / "obs.jsonl"
    _append(path, make_stream(2, hours=48, mints=3, junk=True, themes_per_day=3))
    before, size, mtime = _sha(path), path.stat().st_size, path.stat().st_mtime_ns
    h = ObservationHistory(str(path), retention_seconds=600.0)
    h.load(); h.refresh(); h.load(force=True)
    now = T0 + 47 * 3600
    h.reconstruct_window(_snap(_mint(0), now), now)
    h.reconstruct_window(_snap(_mint(0), T0 + 4 * 3600), T0 + 4 * 3600)  # from-file path
    h.theme_first_seen(); list(h.load()); h.rows(SIGNAL_ROW)
    assert h.stats()["market_horizon"] > T0
    assert (_sha(path), path.stat().st_size, path.stat().st_mtime_ns) == (before, size, mtime)


def test_rotation_truncation_rebuild_summary(tmp_path):
    path = tmp_path / "obs.jsonl"
    clock = [T0]
    first = make_stream(1, hours=10, mints=2, theme_pool=["old_a", "old_b"])
    _append(path, first)
    clock[0] = first[-1][0]
    h = ObservationHistory(str(path), clock=lambda: clock[0])
    h.load()
    assert set(h.theme_first_seen()) <= {"old_a", "old_b"} and h.stats()["market_horizon"] > T0

    # rotation (new inode): summary rebuilt from the new file only, like main
    second = make_stream(2, hours=6, start=T0 + 20 * 3600, mints=2, theme_pool=["new_x", "new_y"])
    tmp = tmp_path / "next.jsonl"
    _append(tmp, second)
    os.replace(tmp, path)
    clock[0] = second[-1][0]
    h.refresh()
    assert _outcome(h.theme_first_seen) == _outcome(BaselineHistory(str(path)).theme_first_seen)
    assert set(h.theme_first_seen()) <= {"new_x", "new_y"}
    assert compare_histories(h, BaselineHistory(str(path)), clock[0], 2, random.Random(3), full_rows=True) == []

    # truncation in place (size < offset) -> full reload + rebuild
    with open(path, "r+b") as fh:
        fh.truncate(len(second[0][1]) + len(second[1][1]))
    h.refresh()
    assert len(h.load()) == 2
    assert compare_histories(h, BaselineHistory(str(path)), clock[0], 2, random.Random(4), full_rows=True) == []

    # copy-truncate + regrowth past the old offset (head differs) -> reload
    third = make_stream(3, hours=5, start=T0 + 40 * 3600, mints=2, theme_pool=["third_a", "third_b"])
    with open(path, "r+b") as fh:
        fh.truncate(0)
        fh.write(b"".join(line for _, line in third))
    clock[0] = third[-1][0]
    h.refresh()
    assert compare_histories(h, BaselineHistory(str(path)), clock[0], 2, random.Random(5), full_rows=True) == []

    # file removed -> empty; recreated -> picked up
    path.unlink()
    assert h.refresh() == [] and h.theme_first_seen() == {} and h.stats()["market_rows_in_memory"] == 0
    _append(path, first[:5])
    h.refresh()
    assert len(h.load()) == 5


def test_unterminated_json_tail_then_append_matches_main(tmp_path):
    """D5 shape: a complete JSON tail without newline is consumed, then a
    writer appends right after it. Streaming the file must split where the
    incremental reads did (exactly main's cached rows)."""
    path = tmp_path / "obs.jsonl"
    a = {"kind": THEME_ROW, "recorded_at": T0, "counts": {"a": 1}}
    b = {"kind": THEME_ROW, "recorded_at": T0 + 1, "counts": {"b": 1}}
    path.write_bytes((json.dumps(a) + "\n").encode() + json.dumps(a).encode())
    new, base = ObservationHistory(str(path), retention_seconds=0.0), BaselineHistory(str(path))
    new.load(), base.load()
    _append(path, [(0, (json.dumps(b) + "\n").encode())])
    new.refresh(), base.refresh()
    assert list(new.load()) == base.load() and len(base.load()) == 3
    assert new.rows(THEME_ROW) == base.rows(THEME_ROW)
    assert new.theme_first_seen() == base.theme_first_seen()
    new.load(force=True), base.load(force=True)
    assert list(new.load()) == base.load()


def test_lookup_below_horizon_after_unseen_rewrite_fails_closed(tmp_path):
    """The only case where memory cannot answer and the file no longer holds
    what was read: raise instead of silently answering from other data."""
    path = tmp_path / "obs.jsonl"
    _append(path, make_stream(4, hours=10, mints=1))
    h = ObservationHistory(str(path), retention_seconds=600.0)
    h.load()
    path.write_bytes(b"")  # rewritten under the reader, refresh() not called
    with pytest.raises(ObservationHistoryChanged):
        h.nearest_market(_mint(0), T0 + HOUR)
    assert h.refresh() == []  # refresh recovers


def test_partial_refresh_error_leaves_state_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "obs.jsonl"
    _append(path, make_stream(6, hours=2, mints=1))
    h = ObservationHistory(str(path))
    h.load()
    before = (len(h.load()), h._offset, h.stats())
    monkeypatch.setattr(history_module, "_REFRESH_LIST_MAX_BYTES", 0)  # force streaming path
    _append(path, make_stream(7, hours=1, start=T0 + 3 * 3600, mints=1))

    def boom(path_, start, on_row):
        on_row({"kind": MARKET_ROW, "observed_at": T0 + 9e5, "mint": _mint(0)})
        raise OSError("disk read error")

    monkeypatch.setattr(history_module, "stream_observations_from", boom)
    with pytest.raises(OSError):
        h.refresh()
    assert (len(h.load()), h._offset, h.stats()) == before


# ---------------------------------------------------------------------------
# 5. collector vs decision views; detector + bridge decisions vs main
# ---------------------------------------------------------------------------

def _bridge(tmp_path, name, history, weights):
    model_path = tmp_path / f"model_{name}.json"
    LogisticModel(weights, bias=-0.2, model_version="pr3.test").save(model_path)
    cfg = Config(live_trading=False, bankroll_usd=20.0, max_position_pct=10.0,
                 decision_shadow_enabled=True, decision_model_path=str(model_path),
                 decision_shadow_file=str(tmp_path / f"shadow_{name}.jsonl"), rpc_url="")
    return DecisionShadowBridge(cfg, RiskManager(cfg), Portfolio(), session=_NoNetworkSession(),
                                history=history)


def _strip_latency(obj):
    if isinstance(obj, dict):
        return {k: _strip_latency(v) for k, v in obj.items() if "latency" not in k}
    if isinstance(obj, list):
        return [_strip_latency(v) for v in obj]
    return obj


@pytest.mark.parametrize("seed", range(3))
def test_decisions_and_detector_reports_match_main(tmp_path, seed):
    rng = random.Random(seed)
    path = tmp_path / "obs.jsonl"
    stream = make_stream(50 + seed, hours=23, start=T0 - 1 * 3600, mints=3, gap_prob=0.04, ooo_prob=0.1,
                         themes_per_day=1)
    clock = [T0]
    new_h = ObservationHistory(str(path), clock=lambda: clock[0])
    base_h = BaselineHistory(str(path))
    weights = [rng.uniform(-0.05, 0.05) for _ in range(N_FEATURES)]
    new_b, base_b = _bridge(tmp_path, "new", new_h, weights), _bridge(tmp_path, "base", base_h, weights)
    det_new = MetaDetector(Config(rpc_url=""), session=_NoNetworkSession(), observations_path=str(path))
    det_base = MetaDetector(Config(rpc_url=""), session=_NoNetworkSession(), observations_path=str(path))
    det_new.history = ObservationHistory(str(path), clock=lambda: clock[0])
    det_base.history = BaselineHistory(str(path))
    written = -1
    decisions = 0
    for offset in [0, 75 * 60, 3 * 3600, 6 * 3600, 12 * 3600, 21 * 3600]:
        now = T0 + offset
        _append(path, [x for x in stream if written < x[0] <= now])
        written = now
        clock[0] = now
        for i in range(3):
            mint = _mint(i)
            safety = TokenSafety(mint=mint, passed=True, liquidity_usd=75_000, price_usd=0.05, symbol=mint[:6])
            snap_new, snap_base = _snap(mint, now, seed), _snap(mint, now, seed)
            r_new = new_b.evaluate_candidate(safety, now=now, snapshot=snap_new)
            r_base = base_b.evaluate_candidate(safety, now=now, snapshot=snap_base)
            assert r_new.decision.to_dict() == r_base.decision.to_dict()
            assert r_new.features.to_dict() == r_base.features.to_dict()
            assert _strip_latency(r_new.to_dict()) == _strip_latency(r_base.to_dict())
            assert {k: v.to_dict() for k, v in snap_new.market.items()} == \
                   {k: v.to_dict() for k, v in snap_base.market.items()}
            decisions += 1
        # collector (MetaDetector.evaluate, persist=False: writes nothing)
        snaps_new = [_snap(_mint(i), now, seed) for i in range(3)]
        snaps_base = [_snap(_mint(i), now, seed) for i in range(3)]
        rep_new = det_new.evaluate(snaps_new, persist=False, now=now)
        rep_base = det_base.evaluate(snaps_base, persist=False, now=now)
        assert rep_new.to_dict() == rep_base.to_dict()
        # collector view == decision view for the same token/time
        for s_col, s_dec in zip(snaps_new, [_snap(_mint(i), now, seed) for i in range(3)]):
            new_h.attach_reconstructed_windows(s_dec, now)
            assert {k: v.to_dict() for k, v in s_col.market.items()} == \
                   {k: v.to_dict() for k, v in s_dec.market.items()}
    journal_new = [_strip_latency(json.loads(x)) for x in (tmp_path / "shadow_new.jsonl").read_text().splitlines()]
    journal_base = [_strip_latency(json.loads(x)) for x in (tmp_path / "shadow_base.jsonl").read_text().splitlines()]
    assert journal_new == journal_base and len(journal_new) == decisions
    notes = [n for rec in journal_new for n in rec.get("notes", []) if n.startswith("meta_window_4h")]
    assert "meta_window_4h=present" in notes and "meta_window_4h=absent" in notes


# ---------------------------------------------------------------------------
# 6. multi-week: bounded memory, summary preserved, file untouched
# ---------------------------------------------------------------------------

def test_21_days_bounded_rows_and_theme_summary(tmp_path):
    path = tmp_path / "obs.jsonl"
    mints = 2
    stream = make_stream(77, hours=21 * 24, mints=mints, themes_per_day=2)
    clock = [T0]
    h = ObservationHistory(str(path), clock=lambda: clock[0])
    h.load()
    bound = mints * math.ceil((default_retention_seconds() + HOUR) / TICK + 2)
    peak_rows = 0
    per_day = int(86400 / TICK)
    by_day: dict[int, list] = {}
    for item in stream:
        by_day.setdefault(int((item[0] - T0) // 86400), []).append(item)
    for day in sorted(by_day):
        for k in range(0, len(by_day[day]), 500):  # refresh in tick-sized batches
            chunk = by_day[day][k:k + 500]
            _append(path, chunk)
            clock[0] = chunk[-1][0]
            h.refresh()
            peak_rows = max(peak_rows, h.stats()["market_rows_in_memory"])
    digest = _sha(path)
    base = BaselineHistory(str(path))
    stats = h.stats()
    assert stats["rows_total"] == len(base.load()) == len(stream)
    assert peak_rows <= bound, (peak_rows, bound)
    assert stats["market_rows_in_memory"] < len(base.rows(MARKET_ROW)) / 20
    assert list(h.theme_first_seen().items()) == list(base.theme_first_seen().items())
    assert len(h.theme_first_seen()) >= 2 * 21  # day-0 themes still indexed on day 21
    now = clock[0] + TICK
    assert compare_histories(h, base, now, mints, random.Random(9)) == []
    assert _sha(path) == digest and per_day == 288
