from __future__ import annotations


def build_moderator_system_prompt() -> str:
    return (
        "You are the Moderator, an impartial judge. Score both sides on logical rigor (0-10), "
        "detect informal fallacies (straw_man, ad_hominem, false_dichotomy, slippery_slope, "
        "appeal_to_authority, hasty_generalization, begging_the_question, etc.). "
        "Return ONLY valid JSON with exactly these keys: "
        '{"rigor_score_proponent": float, "rigor_score_opponent": float, '
        '"fallacies_detected": [{"type": str, "excerpt": str, "explanation": str}], '
        "'commentary': str}. No markdown, no extra text."
    )


def build_moderator_user_prompt(topic: str, transcript: str) -> str:
    return (
        f"Thesis: {topic}\n\nRound transcript:\n{transcript}\n\n"
        "Judge this round and return the JSON verdict."
    )


RETRY_SUFFIX = " Your previous output was not valid JSON. Return valid JSON only, no other text."
