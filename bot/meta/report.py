"""Human-readable read-only META DETECTOR output. No trade verbs."""
from __future__ import annotations

from .model import MetaCluster, MetaReport, MetaSignal, OK


def _level(score) -> str:
    if score.status != OK or score.value is None:
        return "UNVERIFIED"
    if score.value >= 75:
        return "HIGH"
    if score.value >= 45:
        return "MEDIUM"
    return "LOW"


def format_signal(signal: MetaSignal) -> str:
    lines = [
        f"TOKEN {signal.symbol or signal.token[:8]} ({signal.token})",
        f"  Attention: {_level(signal.attention_velocity)}  Trading flow: {_level(signal.trading_velocity)}  "
        f"Wallet flow: {_level(signal.wallet_velocity)}",
        f"  Liquidity: {_level(signal.liquidity_quality)}  Manipulation risk: {_level(signal.manipulation_risk)}",
        f"  Narratives: {', '.join(signal.narratives) or 'UNVERIFIED'}",
        f"  Flow alignment: {signal.flow_alignment}",
        f"  Reason codes: {', '.join(signal.reason_codes) or 'NONE'}",
        f"  Provider freshness: {signal.provider_freshness}",
    ]
    return "\n".join(lines)


def format_cluster(cluster: MetaCluster) -> str:
    conf = cluster.narrative_confidence.value if cluster.narrative_confidence.status == OK else None
    conf_txt = f"{conf:.0f}" if conf is not None else "UNVERIFIED"
    lines = [
        "META DETECTED",
        f"Theme: {cluster.theme}",
        "Tokens:",
        *[f"  {mint}" for mint in cluster.tokens],
        f"Attention: {_level(cluster.attention_velocity)}",
        f"Trading flow: {_level(cluster.volume_velocity)}",
        f"Wallet flow: {_level(cluster.wallet_velocity)}",
        f"Narrative confidence: {conf_txt}",
        f"Status: {cluster.status}",
        f"First detected: {cluster.first_detected}",
        f"Last updated: {cluster.last_updated}",
        f"Reason codes: {', '.join(cluster.reason_codes)}",
        "This output is not a buy, sell, or appear-long instruction.",
    ]
    return "\n".join(lines)


def format_report(report: MetaReport) -> str:
    parts = ["META DETECTOR REPORT (read-only)", *report.notes]
    if report.clusters:
        parts.append("")
        parts.extend(format_cluster(c) for c in report.clusters)
    parts.append("")
    parts.append("TOKEN BREAKDOWN")
    parts.extend(format_signal(s) for s in report.signals)
    return "\n\n".join(parts)
