"""Path-based shadow PnL simulation for arm comparison.

Uses latency-adjusted entry from labels.label_path. Never uses wallet entry price.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .dataset import LabeledExample
from .labels import label_path
from .models.base import LocalModel
from .schema import DEFAULT_DETECTION_LATENCY_SECONDS


@dataclass
class SimTrade:
    mint: str
    entered: bool
    label: int
    net_pnl_pct: float
    gross_pnl_pct: float
    mfe_pct: float
    mae_pct: float
    probability: float
    action_buy: bool


@dataclass
class SimReport:
    n_candidates: int = 0
    n_entered: int = 0
    n_skipped: int = 0
    net_pnl_pct_sum: float = 0.0
    gross_pnl_pct_sum: float = 0.0
    max_drawdown_pct: float = 0.0
    opportunity_cost_pct: float = 0.0  # net PnL left on table by skips that were label=1
    false_positive_entries: int = 0  # entered & label=0
    false_negative_skips: int = 0  # skipped & label=1
    true_positive_entries: int = 0
    true_positive_rejections: int = 0  # skipped & label=0
    expectancy_per_entry: float = 0.0
    trades: list[SimTrade] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": self.n_candidates,
            "n_entered": self.n_entered,
            "n_skipped": self.n_skipped,
            "net_pnl_pct_sum": self.net_pnl_pct_sum,
            "gross_pnl_pct_sum": self.gross_pnl_pct_sum,
            "max_drawdown_pct": self.max_drawdown_pct,
            "opportunity_cost_pct": self.opportunity_cost_pct,
            "false_positive_entries": self.false_positive_entries,
            "false_negative_skips": self.false_negative_skips,
            "true_positive_entries": self.true_positive_entries,
            "true_positive_rejections": self.true_positive_rejections,
            "expectancy_per_entry": self.expectancy_per_entry,
            "profitability": "NO VERIFIED PROFITABILITY" if self.n_candidates == 0 else (
                "SYNTHETIC_OR_UNVERIFIED — not live evidence"
            ),
        }


def _net_from_outcome(example: LabeledExample, fee_pct: float, slippage_pct: float) -> tuple[float, float, float, float]:
    outcome = label_path(
        example.path, example.detection_ts,
        latency_seconds=DEFAULT_DETECTION_LATENCY_SECONDS,
    )
    if outcome.entry_price <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if outcome.exit_price and outcome.entry_price > 0:
        fill_in = outcome.entry_price * (1.0 + slippage_pct / 100.0)
        fill_out = outcome.exit_price * (1.0 - slippage_pct / 100.0)
        gross = (fill_out / fill_in - 1.0) * 100.0
        net = gross - 2.0 * fee_pct
    else:
        gross = 0.0
        net = -2.0 * (fee_pct + slippage_pct)  # timeout still pays round-trip friction if flat
        # Prefer milder timeout cost: fees only on a flat mark
        net = -2.0 * fee_pct
    return net, gross, outcome.max_favorable_excursion_pct, outcome.max_adverse_excursion_pct


def simulate_arm(
    model: LocalModel,
    examples: Sequence[LabeledExample],
    *,
    buy_threshold: float = 0.65,
    fee_pct: float = 0.3,
    slippage_pct: float = 1.0,
) -> SimReport:
    """Replay examples chronologically; enter when model p >= threshold."""
    ordered = sorted(examples, key=lambda e: e.detection_ts)
    report = SimReport(n_candidates=len(ordered))
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    entry_pnls: list[float] = []

    for ex in ordered:
        p = model.predict_proba(ex.features.values).probability
        buy = p >= buy_threshold
        net, gross, mfe, mae = _net_from_outcome(ex, fee_pct, slippage_pct)
        # Counterfactual net if we had entered (for opportunity cost / TPR)
        if buy:
            report.n_entered += 1
            report.net_pnl_pct_sum += net
            report.gross_pnl_pct_sum += gross
            entry_pnls.append(net)
            equity += net
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity))
            if ex.label == 1:
                report.true_positive_entries += 1
            else:
                report.false_positive_entries += 1
        else:
            report.n_skipped += 1
            if ex.label == 1:
                report.false_negative_skips += 1
                report.opportunity_cost_pct += net  # what we missed (could be + or -)
            else:
                report.true_positive_rejections += 1
        report.trades.append(
            SimTrade(
                mint=ex.features.mint, entered=buy, label=ex.label,
                net_pnl_pct=net if buy else 0.0, gross_pnl_pct=gross if buy else 0.0,
                mfe_pct=mfe, mae_pct=mae, probability=p, action_buy=buy,
            )
        )
    report.max_drawdown_pct = max_dd
    report.expectancy_per_entry = (
        sum(entry_pnls) / len(entry_pnls) if entry_pnls else 0.0
    )
    return report
