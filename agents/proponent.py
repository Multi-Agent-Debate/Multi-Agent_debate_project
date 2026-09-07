from engine.agents.base_agent import BaseAgent
from engine.llm_client import LLMClient
from engine.prompts.proponent_prompt import (
    build_proponent_system_prompt,
    build_proponent_user_prompt,
)
from engine.schemas import Role, Turn


class Proponent(BaseAgent):
    def __init__(self, llm: LLMClient | None = None) -> None:
        super().__init__(llm or LLMClient())

    def build_system_prompt(self, persona: str | None = None) -> str:
        return build_proponent_system_prompt(persona)

    def run(self, topic: str, transcript: str, persona: str | None, temperature: float) -> Turn:
        system = self.build_system_prompt(persona)
        user = build_proponent_user_prompt(topic, transcript)
        content = self.llm.generate(system, user, temperature=temperature)
        return Turn(round_number=0, role=Role.PROPONENT, content=content)
