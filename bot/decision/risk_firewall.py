"""Deterministic risk firewall.

The local model can say BUY 0.94 and this layer can still say BLOCK.
Wraps existing SafetyScreener + RiskManager; does not replace them.
The model cannot override these checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Config
from ..portfolio import Portfolio
from ..risk import RiskManager
from ..safety import SafetyScreener, TokenSafety
from .signal import BUY, TypedDecision
from .versions import RISK_ENGINE_VERSION


ALLOW = "ALLOW"
BLOCK = "BLOCK"


@dataclass
class RiskVerdict:
    status: str  # ALLOW | BLOCK
    reasons: list[str] = field(default_factory=list)
    risk_engine_version: str = RISK_ENGINE_VERSION
    safety: Optional[TokenSafety] = None
    position_size_usd: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.status == ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "risk_engine_version": self.risk_engine_version,
            "position_size_usd": self.position_size_usd,
            "safety_passed": None if self.safety is None else self.safety.passed,
            "safety_reasons": [] if self.safety is None else list(self.safety.reasons),
            "liquidity_usd": None if self.safety is None else self.safety.liquidity_usd,
            "price_usd": None if self.safety is None else self.safety.price_usd,
        }


class RiskFirewall:
    """Independent enforcement of hard risk rules before any shadow/live entry."""

    def __init__(
        self,
        cfg: Config,
        risk: Optional[RiskManager] = None,
        screener: Optional[SafetyScreener] = None,
        portfolio: Optional[Portfolio] = None,
        *,
        kill_switch: bool = False,
    ):
        self.cfg = cfg
        self.risk = risk or RiskManager(cfg)
        self.screener = screener
        self.portfolio = portfolio or Portfolio()
        self.kill_switch = kill_switch
        self.version = RISK_ENGINE_VERSION

    def evaluate(
        self,
        mint: str,
        decision: TypedDecision,
        *,
        safety: Optional[TokenSafety] = None,
        skip_network_screen: bool = False,
    ) -> RiskVerdict:
        reasons: list[str] = []

        if self.kill_switch:
            return RiskVerdict(BLOCK, ["KILL_SWITCH"], safety=safety)

        if decision.action != BUY:
            # Non-BUY never proceeds to execution; still report as blocked for journal.
            return RiskVerdict(
                BLOCK,
                [f"MODEL_ACTION_{decision.action}"],
                safety=safety,
            )

        if self.risk.trading_halted():
            reasons.append("DAILY_LOSS_HALT")

        open_count = self.portfolio.open_count
        if not self.risk.can_open_new_position(open_count):
            reasons.append("MAX_OPEN_POSITIONS_OR_HALTED")

        if self.portfolio.has(mint):
            reasons.append("ALREADY_HOLDING")

        # Bankroll / experiment ceiling (authoritative $20 live ceiling stays in executor;
        # firewall also rejects oversized paper allocations).
        size = self.risk.position_size_usd()
        if size <= 0:
            reasons.append("ZERO_POSITION_SIZE")
        if size > self.cfg.bankroll_usd:
            reasons.append("SIZE_EXCEEDS_BANKROLL")

        if safety is None and self.screener is not None and not skip_network_screen:
            safety = self.screener.screen(mint)

        if safety is not None:
            if not safety.passed:
                reasons.append("SAFETY_REJECT:" + ";".join(safety.reasons[:3]))
            if safety.liquidity_usd < self.cfg.min_liquidity_usd:
                reasons.append("LIQUIDITY_FLOOR")
            if safety.price_usd <= 0:
                reasons.append("NO_PRICE")
        elif not skip_network_screen:
            # Live/shadow with network: cannot proceed without a safety result.
            reasons.append("SAFETY_UNAVAILABLE")
        # skip_network_screen=True and safety=None: caller opts into offline
        # evaluation (unit tests / labeled replay). Safety must be applied
        # separately before any capital decision.

        # Slippage ceiling is enforced at quote time by JupiterClient; record intent.
        if self.cfg.max_slippage_bps <= 0:
            reasons.append("SLIPPAGE_CEILING_INVALID")

        if reasons:
            return RiskVerdict(BLOCK, reasons, safety=safety, position_size_usd=size)
        return RiskVerdict(ALLOW, ["RISK_OK"], safety=safety, position_size_usd=size)
