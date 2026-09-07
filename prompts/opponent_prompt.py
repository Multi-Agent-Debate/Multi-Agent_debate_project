from __future__ import annotations

from typing import Optional


def build_opponent_system_prompt(persona: Optional[str] = None) -> str:
    base = (
        "You are the Opponent, a skeptical antagonist in a structured debate. "
        "Argue AGAINST the given thesis: probe for weaknesses, present counter-evidence, "
        "and expose flawed reasoning. Respond directly to the proponent's latest claims. "
        "Be sharp but fair — no personal attacks. Keep your response focused and under ~400 words."
    )
    if persona:
        return f"{base}\nAdopt this persona/tone: {persona}."
    return base


def build_opponent_user_prompt(topic: str, transcript: str) -> str:
    return (
        f"Thesis: {topic}\n\nDebate so far:\n{transcript}\n\n"
        "Present your rebuttal/counter-argument for this round against the thesis."
    )
