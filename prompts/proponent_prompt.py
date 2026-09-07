from __future__ import annotations

from typing import Optional


def build_proponent_system_prompt(persona: Optional[str] = None) -> str:
    base = (
        "You are the Proponent in a structured debate. "
        "Argue IN FAVOR of the given thesis with clear claims, evidence, and reasoning. "
        "Address the opponent's objections directly when prior turns exist. "
        "Be persuasive but logically rigorous. Keep your response focused and under ~400 words."
    )
    if persona:
        return f"{base}\nAdopt this persona/tone: {persona}."
    return base


def build_proponent_user_prompt(topic: str, transcript: str) -> str:
    if transcript:
        return (
            f"Thesis: {topic}\n\nDebate so far:\n{transcript}\n\n"
            "Present your argument for this round in favor of the thesis."
        )
    return (
        f"Thesis: {topic}\n\nThis is round 1. "
        "Present your opening argument in favor of the thesis."
    )
