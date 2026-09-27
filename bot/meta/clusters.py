"""Meta cluster detection.

A single accelerating token is not a market-wide meta. At least two
independent tokens must share a theme.
"""
from __future__ import annotations

from collections import defaultdict

from .formulas import weighted_mean
from .model import CLUSTER_STATES, MetaCluster, MetaSignal, OK, Score, ok, unverified


def _avg(scores: list[Score]) -> Score:
    values = [s.value for s in scores if s.status == OK and s.value is not None]
    codes: list[str] = []
    for s in scores:
        codes.extend(s.reason_codes)
    if not values:
        return unverified("cluster members lack this component")
    return ok(sum(values) / len(values), "cluster_mean = mean(member scores)", sorted(set(codes)))


def cluster_status(attention: Score, trading: Score, member_count: int) -> str:
    att = attention.value if attention.status == OK else None
    trd = trading.value if trading.status == OK else None
    if member_count < 2:
        return "NEW"
    peakish = (att or 0) >= 80 or (trd or 0) >= 80
    rising = (att or 0) >= 60 or (trd or 0) >= 60
    decaying = (att is not None and att < 35) and (trd is None or trd < 35)
    if decaying:
        return "DECAYING"
    if peakish and member_count >= 3:
        return "PEAKING"
    if rising and member_count >= 3:
        return "ACCELERATING"
    if rising:
        return "EMERGING"
    return "DORMANT"


def build_clusters(signals: list[MetaSignal], first_seen: dict[str, float], now: float) -> list[MetaCluster]:
    by_theme: dict[str, list[MetaSignal]] = defaultdict(list)
    for signal in signals:
        for theme in signal.narratives:
            by_theme[theme].append(signal)

    clusters: list[MetaCluster] = []
    for theme, members in sorted(by_theme.items()):
        unique = {}
        for member in members:
            unique[member.token] = member
        members = list(unique.values())
        tokens = [m.token for m in members]
        attention = _avg([m.attention_velocity for m in members])
        volume = _avg([m.trading_velocity for m in members])
        wallets = _avg([m.wallet_velocity for m in members])
        status = cluster_status(attention, volume, len(tokens))
        if status not in CLUSTER_STATES:
            status = "NEW"
        confidence_parts = []
        if attention.status == OK:
            confidence_parts.append((attention.value, 0.4))
        if volume.status == OK:
            confidence_parts.append((volume.value, 0.3))
        if wallets.status == OK:
            confidence_parts.append((wallets.value, 0.3))
        conf_val = weighted_mean(confidence_parts)
        if conf_val is None:
            confidence = unverified("not enough verified components for cluster confidence")
        else:
            if len(tokens) < 2:
                conf_val *= 0.4
            confidence = ok(conf_val, "narrative_confidence = weighted mean of member velocities",
                            ["NARRATIVE_CLUSTER_FORMED"] if len(tokens) >= 2 else ["SINGLE_TOKEN_THEME"],
                            members=len(tokens))
        codes = ["NARRATIVE_CLUSTER_FORMED"] if len(tokens) >= 2 else ["SINGLE_TOKEN_NOT_A_META"]
        if status == "DECAYING":
            codes.append("NARRATIVE_DECAY")
        clusters.append(MetaCluster(
            theme=theme,
            tokens=tokens,
            attention_velocity=attention,
            volume_velocity=volume,
            wallet_velocity=wallets,
            narrative_confidence=confidence,
            first_detected=first_seen.get(theme, now),
            last_updated=now,
            status=status,
            reason_codes=codes,
        ))
    return clusters
