"""Narrative classification from token metadata + supplied social texts.

Does not assert an external catalyst unless the snapshot marks it verified.
"""
from __future__ import annotations

import re

from .model import Score, TokenSnapshot, ok, unverified

NARRATIVE_LEXICON = {
    "ai_agents": (
        "ai", "agent", "gpt", "llm", "autonomous", "bot army", "machine learning",
    ),
    "political": ("trump", "biden", "election", "maga", "president", "congress"),
    "celebrity": ("celebrity", "influencer", "elon", "kanye", "idol"),
    "gaming": ("game", "gaming", "steam", "xbox", "playstation", "npc"),
    "sports": ("nfl", "nba", "fifa", "soccer", "football", "ufc"),
    "finance": ("defi", "etf", "fed", "rates", "wallstreet", "sec"),
    "animals": ("dog", "cat", "pepe", "frog", "inu", "shib", "monkey", "ape"),
    "internet_culture": ("meme", "wojak", "copium", "based", "sigma", "tiktok"),
    "platform": ("solana", "pump.fun", "pumpfun", "jupiter", "phantom", "telegram"),
    "current_events": ("breaking", "war", "launch", "listing", "hack"),
    "parody_copycat": ("inu", "2.0", "3.0", "fork", "copy", "parody"),
}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9.]+")


def _tokens(*parts: str) -> set[str]:
    blob = " ".join(p.lower() for p in parts if p)
    return {t for t in _TOKEN_SPLIT.split(blob) if t}


def classify_narratives(snap: TokenSnapshot) -> list[str]:
    blob = " ".join(
        part.lower()
        for part in (snap.symbol, snap.name, snap.description, *snap.texts)
        if part
    )
    hits = []
    for theme, lexicon in NARRATIVE_LEXICON.items():
        if any(term in blob for term in lexicon):
            hits.append(theme)
    return hits


def score_narrative(snap: TokenSnapshot, themes: list[str]) -> tuple[Score, Score, str]:
    if not (snap.symbol or snap.name or snap.description or snap.texts):
        return (
            unverified("no token metadata or social texts"),
            unverified("no token metadata or social texts"),
            "UNVERIFIED",
        )
    velocity = min(100.0, len(themes) * 25.0)
    freshness = 100.0 if themes else 10.0
    codes = ["NARRATIVE_PRESENT"] if themes else ["NARRATIVE_WEAK"]
    if snap.external_catalyst_verified:
        catalyst = "VERIFIED"
        codes.append("EXTERNAL_CATALYST_VERIFIED")
    else:
        catalyst = "UNVERIFIED"
    vel = ok(velocity, "narrative_velocity = min(100, theme_count*25)", codes, themes=themes)
    fresh = ok(freshness, "narrative_freshness = 100 if themes else 10", codes, themes=themes)
    return vel, fresh, catalyst
