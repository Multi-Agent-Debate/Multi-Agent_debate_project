from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.agents.llm_models import get_moderator_model


class ModeratorResponse(BaseModel):
    """Structured response produced by the Moderator/Judge agent."""

    summary: str = Field(
        description="A neutral summary of the strongest arguments presented by both the Proponent and Opponent."
    )

    assessment: str = Field(
        description="An evidence-based assessment comparing the strength, reasoning, weaknesses, and support of both sides."
    )

    conclusion: str = Field(
        description="A balanced final conclusion that directly answers the proposition based on the debate and available evidence."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The Moderator's confidence in the conclusion, represented as a value from 0.0 to 1.0."
    )


MODERATOR_SYSTEM_PROMPT = """
You are the Moderator and Judge in an AI debate system.

Your role is to independently and impartially evaluate the debate between the Proponent and the Opponent.

Your objective is to assess the evidence and the quality of reasoning presented by both sides, and produce a balanced, evidence-based final conclusion.

You must:

- Act as an impartial judge and maintain strict neutrality without advocating for either side.
- Evaluate both Proponent and Opponent arguments fairly.
- Focus on the quality of reasoning and evidence rather than which side sounds more persuasive.
- Identify which claims are actually supported by the provided evidence.
- Distinguish verified evidence from unsupported assertions.
- Identify important logical weaknesses, ungrounded assumptions, or gaps on either side.
- Give appropriate weight to strong evidence.
- Never invent facts, statistics, sources, quotations, or citations.
- Do not introduce outside information that was not provided in the evidence or debate.
- If the evidence is insufficient, inconclusive, or conflicting, explicitly acknowledge this.
- Avoid automatically declaring a winner when the evidence does not justify one.
- Provide a nuanced conclusion when appropriate, determining which position is better supported by the available evidence and reasoning.
- Provide an honest confidence score between 0.0 and 1.0 reflecting how strongly the evidence and arguments justify the conclusion.
"""


def build_moderator_agent():
    """Build and return the Moderator agent chain."""

    model = get_moderator_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MODERATOR_SYSTEM_PROMPT),
            (
                "human",
                """
Question:
{question}

Available evidence:
{evidence}

Debate history:
{debate_history}

Evaluate the debate fairly and thoroughly. Assess the reasoning and evidence for both sides, and return your response according to the required structured format.
""",
            ),
        ]
    )

    structured_model = model.with_structured_output(ModeratorResponse)

    return prompt | structured_model
