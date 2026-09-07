import json

from pydantic import ValidationError

from engine.agents.base_agent import BaseAgent
from engine.llm_client import LLMClient
from engine.prompts.moderator_prompt import (
    RETRY_SUFFIX,
    build_moderator_system_prompt,
    build_moderator_user_prompt,
)
from engine.schemas import Fallacy, ModeratorVerdict


class ModeratorParseError(ValueError):
    pass


def parse_moderator_output(raw: str, round_number: int) -> ModeratorVerdict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ModeratorParseError(f"Moderator output is not valid JSON: {e}") from e
    try:
        return ModeratorVerdict(
            round_number=round_number,
            rigor_score_proponent=float(data["rigor_score_proponent"]),
            rigor_score_opponent=float(data["rigor_score_opponent"]),
            fallacies_detected=[Fallacy(**f) for f in data.get("fallacies_detected", [])],
            commentary=data["commentary"],
        )
    except (KeyError, TypeError, ValidationError) as e:
        raise ModeratorParseError(f"Moderator JSON does not match schema: {e}") from e


class Moderator(BaseAgent):
    """Uses a lower temperature by default since it scores rather than argues."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        super().__init__(llm or LLMClient(temperature=0.2))

    def build_system_prompt(self, persona: str | None = None) -> str:
        return build_moderator_system_prompt()

    def run(self, topic: str, transcript: str, round_number: int) -> ModeratorVerdict:
        system = self.build_system_prompt()
        user = build_moderator_user_prompt(topic, transcript)
        raw = self.llm.generate(system, user, temperature=0.2, json_mode=True)
        try:
            return parse_moderator_output(raw, round_number)
        except ModeratorParseError:
            retry = self.llm.generate(
                system, user + RETRY_SUFFIX, temperature=0.2, json_mode=True
            )
            try:
                return parse_moderator_output(retry, round_number)
            except ModeratorParseError as e:
                raise ModeratorParseError(
                    f"Moderator failed to return valid JSON after retry (round {round_number}): {e}"
                ) from e
