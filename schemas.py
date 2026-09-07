from pydantic import BaseModel
from enum import Enum
from typing import Optional


class Role(str, Enum):
    PROPONENT = "proponent"
    OPPONENT = "opponent"
    MODERATOR = "moderator"


class Fallacy(BaseModel):
    type: str            # e.g. "straw_man", "ad_hominem"
    excerpt: str
    explanation: str


class Turn(BaseModel):
    round_number: int
    role: Role
    content: str
    citations: list[str] = []


class ModeratorVerdict(BaseModel):
    round_number: int
    rigor_score_proponent: float   # 0-10
    rigor_score_opponent: float
    fallacies_detected: list[Fallacy] = []
    commentary: str


class ConsensusReport(BaseModel):
    topic: str
    total_rounds: int
    points_of_agreement: list[str]
    points_of_conflict: list[str]
    recommended_actions: list[str]
    turns: list[Turn]
    verdicts: list[ModeratorVerdict]


class DebateConfig(BaseModel):
    topic: str
    rounds: int = 3
    proponent_persona: Optional[str] = None
    opponent_persona: Optional[str] = None
    temperature: float = 0.7
    use_rag: bool = False
