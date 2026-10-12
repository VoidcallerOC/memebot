"""Position tracking and PnL.

Deliberately simple and in-memory. For real long-running use you'd persist
this to disk/db, but the accounting logic is what matters and it lives here.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Raw (smallest-unit) SPL amounts are u64: the same bound as
# ``jupiter.parse_raw_amount``. Raw holdings and raw sell amounts outside
# 1..2**64-1 (or not a real int) are rejected, never clamped or repaired.
MAX_RAW_UNITS = 2**64 - 1


def is_raw_units(value: object) -> bool:
    """True for an int (not bool) in ``1 .. MAX_RAW_UNITS``."""
    return (
        isinstance(value, int) and not isinstance(value, bool)
        and 0 < value <= MAX_RAW_UNITS
    )


# Buy-intent statuses the live entry lifecycle writes (bot.main
# OPEN_BUY_INTENT_STATUSES plus the terminal ``failed`` / ``settled``).
BUY_INTENT_STATUSES = frozenset({"pending", "unresolved", "confirmed_unrecorded", "failed", "settled"})


def _real_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


# Every sell-intent status any writer has persisted (git history of
# bot/main.py: c733c9e pending/blocked/unresolved/settled; 1975efd +failed;
# 4587d65 +confirmed_unbooked). da613e4 introduced the field but never
# appended to it.
SELL_INTENT_STATUSES = frozenset({
    "pending", "blocked", "unresolved", "confirmed_unbooked", "failed", "settled",
})


def sell_intent_problem(entry: object) -> str | None:
    """Why a persisted live SELL intent cannot be classified, else None.

    Accepts every historical shape explicitly: c733c9e (mint, fraction,
    reason, requested_price, status, tx_signature), 1975efd (+created_at,
    raw_amount possibly None, resolution fields) and 4587d65+ (+raw_amount at
    creation, optional rung, booking fields). Required to classify: a
    non-empty ``mint`` (what the open intent freezes) and a known string
    ``status``; ``tx_signature`` is None or a non-empty string. A
    ``confirmed_unbooked`` intent is booked from its stored evidence, so it
    also needs its signature, ``fraction``, ``requested_price`` and
    ``reason`` and must not already be booked.
    """
    if not isinstance(entry, dict):
        return "entry is not an object"
    mint = entry.get("mint")
    if not isinstance(mint, str) or not mint:
        return "missing or invalid mint"
    status = entry.get("status")
    if not isinstance(status, str):
        return "missing or non-string status"
    if status not in SELL_INTENT_STATUSES:
        return f"unknown status {status!r}"
    sig = entry.get("tx_signature")
    if sig is not None and (not isinstance(sig, str) or not sig):
        return "invalid tx_signature"
    if status == "confirmed_unbooked":
        if sig is None:
            return "confirmed_unbooked without tx_signature"
        if not _real_number(entry.get("fraction")) or not _real_number(entry.get("requested_price")):
            return "confirmed_unbooked without numeric fraction/requested_price"
        if not isinstance(entry.get("reason"), str):
            return "confirmed_unbooked without reason"
        if entry.get("booked_at") is not None:
            return "confirmed_unbooked already carries booked_at"
    return None


def buy_intent_problem(entry: object) -> str | None:
    """Why a persisted buy intent is malformed, or None. Mirrors what
    ``TradingBot._live_entry`` writes; anything else is corruption, never
    guessed or repaired."""
    if not isinstance(entry, dict):
        return "entry is not an object"
    if entry.get("side") != "buy":
        return "side is not 'buy'"
    mint = entry.get("mint")
    if not isinstance(mint, str) or not mint:
        return "mint missing or not a string"
    status = entry.get("status")
    if status not in BUY_INTENT_STATUSES:
        return f"unknown status {status!r}"
    if not is_raw_units(entry.get("lamports")):
        return "intended lamports missing or not an integer in 1..2**64-1"
    if not _real_number(entry.get("created_at")):
        return "created_at missing or not a finite number"
    if not _real_number(entry.get("allocation_usd")):
        return "allocation_usd missing or not a finite number"
    if "tx_signature" not in entry:
        return "tx_signature field missing"
    sig = entry["tx_signature"]
    if sig is not None and (not isinstance(sig, str) or not sig):
        return "tx_signature is not null or a non-empty string"
    if status in {"confirmed_unrecorded", "settled"} and sig is None:
        return f"{status} intent has no tx_signature"
    return None


@dataclass
class Position:
    mint: str
    symbol: str
    entry_price: float
    size_usd: float                 # USD cost basis remaining
    tokens: float                   # token units currently held
    ladder_filled: set[float] = field(default_factory=set)
    realized_pnl: float = 0.0       # banked PnL from partial sells
    token_decimals: int | None = None
    entry_signature: str | None = None
    actual_in_amount: int | None = None
    actual_out_amount: int | None = None
    fee_lamports: int | None = None
    # Live positions only: exact integer raw holding (smallest token units)
    # and the raw amount bought, seeded from the verified entry fill and
    # decremented only by confirmed, settled sells. None for paper and for
    # legacy positions persisted before these fields existed.
    raw_tokens: int | None = None
    raw_tokens_initial: int | None = None

    @property
    def original_tokens(self) -> float:
        return self._original_tokens

    original_tokens_override: float | None = None

    def __post_init__(self):
        if self.original_tokens_override is not None:
            self._original_tokens = self.original_tokens_override
        else:
            self._original_tokens = self.tokens

    def to_dict(self) -> dict:
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "size_usd": self.size_usd,
            "tokens": self.tokens,
            "ladder_filled": sorted(self.ladder_filled),
            "realized_pnl": self.realized_pnl,
            "original_tokens": self._original_tokens,
            "token_decimals": self.token_decimals,
            "entry_signature": self.entry_signature,
            "actual_in_amount": self.actual_in_amount,
            "actual_out_amount": self.actual_out_amount,
            "fee_lamports": self.fee_lamports,
            "raw_tokens": self.raw_tokens,
            "raw_tokens_initial": self.raw_tokens_initial,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(
            mint=d["mint"],
            symbol=d["symbol"],
            entry_price=d["entry_price"],
            size_usd=d["size_usd"],
            tokens=d["tokens"],
            ladder_filled=set(d.get("ladder_filled", [])),
            realized_pnl=d.get("realized_pnl", 0.0),
            original_tokens_override=d.get("original_tokens"),
            token_decimals=d.get("token_decimals"),
            entry_signature=d.get("entry_signature"),
            actual_in_amount=d.get("actual_in_amount"),
            actual_out_amount=d.get("actual_out_amount"),
            fee_lamports=d.get("fee_lamports"),
            raw_tokens=d.get("raw_tokens"),
            raw_tokens_initial=d.get("raw_tokens_initial"),
        )


class Portfolio:
    def __init__(self):
        self.positions: dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.pending_intents: list[dict] = []   # live SELL intents
        # Live BUY intents (armed entries), kept apart from sell intents so no
        # sell path (freeze, resolver, pruning) ever sees them. Same atomic
        # state file; absent in legacy state (loads as an empty list).
        self.buy_intents: list[dict] = []
        # Persisted intent fields that failed validation: field -> reason.
        # Their raw data is kept verbatim in ``_raw_corrupt`` and written back
        # unchanged by ``to_dict`` (never dropped or rewritten); the bot
        # refuses armed trading and all new entries while any is set.
        self.corrupt: dict[str, str] = {}
        self._raw_corrupt: dict[str, object] = {}

    def open(self, mint: str, symbol: str, entry_price: float,
             size_usd: float, tokens: float, **fill) -> Position:
        pos = Position(
            mint=mint, symbol=symbol, entry_price=entry_price,
            size_usd=size_usd, tokens=tokens, **fill,
        )
        self.positions[mint] = pos
        log.info("OPEN  %s: $%.2f @ %.8f (%.2f tokens)", symbol, size_usd, entry_price, tokens)
        return pos

    def has(self, mint: str) -> bool:
        return mint in self.positions

    @property
    def open_count(self) -> int:
        return len(self.positions)

    def sell_fraction(self, mint: str, fraction: float, current_price: float,
                      reason: str, *, raw_sold: int | None = None) -> float:
        """Book a sell. ``raw_sold`` (live) is the exact confirmed raw amount
        sold: the quantity booked is ``raw_sold / 10**decimals`` and the raw
        holding is decremented exactly; ``tokens`` is re-derived from it.
        Without ``raw_sold`` (paper / legacy) the float path is unchanged."""
        pos = self.positions[mint]
        if raw_sold is not None:
            return self._sell_raw(pos, raw_sold, current_price, reason)
        tokens_to_sell = min(pos.original_tokens * fraction, pos.tokens)
        if tokens_to_sell <= 0:
            return 0.0

        proceeds = tokens_to_sell * current_price
        cost_basis = tokens_to_sell * pos.entry_price
        pnl = proceeds - cost_basis

        pos.tokens -= tokens_to_sell
        pos.size_usd -= cost_basis
        pos.realized_pnl += pnl
        self.realized_pnl += pnl

        log.info("SELL  %s %.0f%% (%s): proceeds $%.2f, pnl $%.2f",
                 pos.symbol, fraction * 100, reason, proceeds, pnl)

        if pos.tokens <= 1e-9 or reason == "stop_loss":
            self._close(mint, current_price)
        return pnl

    def _sell_raw(self, pos: Position, raw_sold: int, current_price: float,
                  reason: str) -> float:
        dec = pos.token_decimals
        raw = pos.raw_tokens
        initial = pos.raw_tokens_initial
        if (
            not is_raw_units(raw_sold) or not is_raw_units(raw) or not is_raw_units(initial)
            or isinstance(dec, bool) or not isinstance(dec, int) or not 0 <= dec <= 18
            or not raw_sold <= raw <= initial
        ):
            raise ValueError(
                f"invalid exact raw sell for {pos.symbol}: sold={raw_sold!r} held={raw!r} "
                f"bought={initial!r} decimals={dec!r}"
            )
        tokens_to_sell = raw_sold / (10 ** dec)
        proceeds = tokens_to_sell * current_price
        cost_basis = tokens_to_sell * pos.entry_price
        pnl = proceeds - cost_basis

        pos.raw_tokens = raw - raw_sold
        pos.tokens = pos.raw_tokens / (10 ** dec)
        pos.size_usd -= cost_basis
        pos.realized_pnl += pnl
        self.realized_pnl += pnl

        log.info("SELL  %s %d raw (%s): proceeds $%.2f, pnl $%.2f",
                 pos.symbol, raw_sold, reason, proceeds, pnl)
        if pos.raw_tokens == 0 or reason == "stop_loss":
            self._close(pos.mint, current_price)
        return pnl

    def _close(self, mint: str, current_price: float) -> None:
        pos = self.positions.pop(mint)
        log.info("CLOSE %s: total realized pnl $%.2f", pos.symbol, pos.realized_pnl)

    def to_dict(self) -> dict:
        return {
            "realized_pnl": self.realized_pnl,
            "positions": [p.to_dict() for p in self.positions.values()],
            "pending_intents": (
                copy.deepcopy(self._raw_corrupt["pending_intents"])
                if "pending_intents" in self.corrupt else list(self.pending_intents)
            ),
            "buy_intents": (
                copy.deepcopy(self._raw_corrupt["buy_intents"])
                if "buy_intents" in self.corrupt else list(self.buy_intents)
            ),
        }

    def load_dict(self, d: dict) -> None:
        self.realized_pnl = d.get("realized_pnl", 0.0)
        self.positions = {}
        for pd in d.get("positions", []):
            pos = Position.from_dict(pd)
            self.positions[pos.mint] = pos
        self.corrupt, self._raw_corrupt = {}, {}
        # Sell intents: absent / null / [] (legacy) -> []. A present non-list
        # value or a non-object entry would be silently ignored by the
        # sell-freeze checks, so it is corruption (raw kept, trading refused).
        # Field-level validation of sell entries is a follow-up (legacy sell
        # intents have several historical shapes).
        if "pending_intents" not in d:  # legacy state: field never written
            raw_sells: object = []
        else:
            raw_sells = d["pending_intents"]
        sell_problems = (
            [] if not isinstance(raw_sells, list)
            else [(n, sell_intent_problem(i)) for n, i in enumerate(raw_sells)]
        )
        sell_problems = [(n, why) for n, why in sell_problems if why is not None]
        if not isinstance(raw_sells, list):
            # Present but null / non-list is corrupt, never silently [].
            self._mark_corrupt("pending_intents", raw_sells, "field is not a list")
            self.pending_intents = []
        elif sell_problems:
            n, why = sell_problems[0]
            self._mark_corrupt("pending_intents", raw_sells, f"entry {n}: {why}")
            self.pending_intents = []
        else:
            self.pending_intents = list(raw_sells)
        # Buy intents: key ABSENT (legacy state) -> []. Present but not a list,
        # or any malformed entry -> corrupt: raw kept verbatim, not loaded.
        if "buy_intents" not in d:
            self.buy_intents = []
        else:
            raw_buys = d["buy_intents"]
            if not isinstance(raw_buys, list):
                problem = "field is not a list"
            else:
                problem = next((
                    f"entry {n}: {why}" for n, why in
                    ((n, buy_intent_problem(e)) for n, e in enumerate(raw_buys)) if why
                ), None)
            if problem is None:
                self.buy_intents = list(raw_buys)
            else:
                self._mark_corrupt("buy_intents", raw_buys, problem)
                self.buy_intents = []

    def _mark_corrupt(self, name: str, raw: object, reason: str) -> None:
        self.corrupt[name] = reason
        self._raw_corrupt[name] = copy.deepcopy(raw)
        log.error("state %s corrupt (%s): raw data preserved; state recovery required",
                  name, reason)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for mint, pos in self.positions.items():
            price = prices.get(mint)
            if price is None:
                continue
            total += pos.tokens * (price - pos.entry_price)
        return total
