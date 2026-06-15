"""Risk management — the rules that keep losses small and survivable.

None of this predicts the market. Its only job is to make sure that when a
trade goes against us (and many will), the damage is bounded, and that when a
trade works, we actually bank some of the gain instead of round-tripping it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class SellAction:
    """A risk-driven instruction to sell part or all of a position."""
    fraction: float          # fraction of the ORIGINAL position to sell
    reason: str              # "stop_loss" | "take_profit:<gain>" | "trailing"


class RiskManager:
    """Stateless-per-decision risk rules plus daily-loss circuit breaker."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._day: date = date.today()
        self._realized_pnl_today: float = 0.0
        self._halted_today: bool = False

    # -- daily circuit breaker ---------------------------------------------

    def _roll_day_if_needed(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self._realized_pnl_today = 0.0
            self._halted_today = False
            log.info("New trading day — daily PnL and halt reset.")

    def record_realized_pnl(self, pnl_usd: float) -> None:
        self._roll_day_if_needed()
        self._realized_pnl_today += pnl_usd
        if self._realized_pnl_today <= -self.cfg.daily_loss_limit_usd():
            if not self._halted_today:
                log.warning(
                    "DAILY LOSS LIMIT HIT (%.2f USD lost). Halting new entries "
                    "for the rest of the day.", -self._realized_pnl_today,
                )
            self._halted_today = True

    @property
    def realized_pnl_today(self) -> float:
        self._roll_day_if_needed()
        return self._realized_pnl_today

    def trading_halted(self) -> bool:
        self._roll_day_if_needed()
        return self._halted_today

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "day": self._day.isoformat(),
            "realized_pnl_today": self._realized_pnl_today,
            "halted_today": self._halted_today,
        }

    def load_dict(self, d: dict) -> None:
        try:
            self._day = date.fromisoformat(d["day"])
        except (KeyError, ValueError):
            self._day = date.today()
        self._realized_pnl_today = d.get("realized_pnl_today", 0.0)
        self._halted_today = d.get("halted_today", False)
        # If the saved state is from a previous day, reset on next access.
        self._roll_day_if_needed()

    # -- entry sizing -------------------------------------------------------

    def can_open_new_position(self, open_count: int) -> bool:
        if self.trading_halted():
            return False
        return open_count < self.cfg.max_open_positions

    def position_size_usd(self) -> float:
        """Fixed-fractional sizing: a constant small % of bankroll per trade.
        Simple, robust, and the single biggest protection against ruin."""
        return self.cfg.max_position_usd()

    # -- exit rules ---------------------------------------------------------

    def stop_loss_price(self, entry_price: float) -> float:
        return entry_price * (1.0 - self.cfg.stop_loss_pct / 100.0)

    def evaluate_exit(
        self,
        entry_price: float,
        current_price: float,
        ladder_filled: set[float],
    ) -> list[SellAction]:
        """Given a position, return any sells to execute right now.

        ``ladder_filled`` is the set of take-profit gain-targets already sold,
        so we don't sell the same rung twice. The caller persists it.
        """
        actions: list[SellAction] = []
        if entry_price <= 0:
            return actions

        # 1) Stop-loss takes absolute priority — dump the whole position.
        if current_price <= self.stop_loss_price(entry_price):
            actions.append(SellAction(fraction=1.0, reason="stop_loss"))
            return actions

        # 2) Take-profit ladder — bank gains as targets are hit.
        gain_pct = (current_price / entry_price - 1.0) * 100.0
        for target_gain, sell_fraction in self.cfg.take_profit_ladder:
            if gain_pct >= target_gain and target_gain not in ladder_filled:
                actions.append(
                    SellAction(fraction=sell_fraction, reason=f"take_profit:{target_gain:g}")
                )
        return actions
