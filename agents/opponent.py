from engine.agents.base_agent import BaseAgent
from engine.llm_client import LLMClient
from engine.prompts.opponent_prompt import (
    build_opponent_system_prompt,
    build_opponent_user_prompt,
)
from engine.schemas import Role, Turn


class Opponent(BaseAgent):
    def __init__(self, llm: LLMClient | None = None) -> None:
        super().__init__(llm or LLMClient())

    def build_system_prompt(self, persona: str | None = None) -> str:
        return build_opponent_system_prompt(persona)

    def run(self, topic: str, transcript: str, persona: str | None, temperature: float) -> Turn:
        system = self.build_system_prompt(persona)
        user = build_opponent_user_prompt(topic, transcript)
        content = self.llm.generate(system, user, temperature=temperature)
        return Turn(round_number=0, role=Role.OPPONENT, content=content)
