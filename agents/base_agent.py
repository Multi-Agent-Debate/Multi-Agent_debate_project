from abc import ABC, abstractmethod

from engine.llm_client import LLMClient


class BaseAgent(ABC):
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    @abstractmethod
    def build_system_prompt(self, persona: str | None = None) -> str:
        raise NotImplementedError
