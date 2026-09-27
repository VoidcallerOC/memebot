"""Position tracking and PnL.

Deliberately simple and in-memory. For real long-running use you'd persist
this to disk/db, but the accounting logic is what matters and it lives here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


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
        )


class Portfolio:
    def __init__(self):
        self.positions: dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.pending_intents: list[dict] = []

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
                      reason: str) -> float:
        pos = self.positions[mint]
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

    def _close(self, mint: str, current_price: float) -> None:
        pos = self.positions.pop(mint)
        log.info("CLOSE %s: total realized pnl $%.2f", pos.symbol, pos.realized_pnl)

    def to_dict(self) -> dict:
        return {
            "realized_pnl": self.realized_pnl,
            "positions": [p.to_dict() for p in self.positions.values()],
            "pending_intents": list(self.pending_intents),
        }

    def load_dict(self, d: dict) -> None:
        self.realized_pnl = d.get("realized_pnl", 0.0)
        self.positions = {}
        for pd in d.get("positions", []):
            pos = Position.from_dict(pd)
            self.positions[pos.mint] = pos
        self.pending_intents = list(d.get("pending_intents") or [])

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for mint, pos in self.positions.items():
            price = prices.get(mint)
            if price is None:
                continue
            total += pos.tokens * (price - pos.entry_price)
        return total
