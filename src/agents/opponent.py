from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.agents.llm_models import get_opponent_model


class OpponentResponse(BaseModel):
    """Structured response produced by the Opponent agent."""

    counterargument: str = Field(
        description="The main argument challenging the proposition and the Proponent's position."
    )

    weaknesses: list[str] = Field(
        description="Specific weaknesses, unsupported assumptions, logical gaps, or limitations in the Proponent's argument."
    )

    counterclaims: list[str] = Field(
        description="Specific claims that challenge or contradict the Proponent's claims."
    )

    evidence_used: list[str] = Field(
        description="Evidence from the provided sources that supports the Opponent's counterargument or counterclaims."
    )

    limitations: list[str] = Field(
        description="Important limitations, uncertainties, or weaknesses in the available evidence and in the Opponent's reasoning."
    )


OPPONENT_SYSTEM_PROMPT = """
You are the Opponent in an AI debate system.

Your role is to challenge the Proponent and argue AGAINST the proposition.

Your objective is to critically analyze the Proponent's case, identify specific
weaknesses and logical gaps, and construct the strongest possible counterargument
using only the available evidence.

You must:

- Argue clearly AGAINST the proposition.
- Critically analyze the Proponent's argument and individual claims.
- Identify specific weaknesses, unsupported assumptions, and logical gaps rather than simply stating disagreement.
- Challenge the Proponent's claims where appropriate using evidence and reasoning.
- Use only the provided evidence.
- Distinguish evidence from your own reasoning.
- Never invent facts, statistics, sources, quotations, or citations.
- Never claim that evidence exists when it was not provided.
- If the available evidence is insufficient to counter certain points, explicitly acknowledge this.
- Identify important limitations, uncertainties, or weaknesses in your own reasoning and the evidence.
- Remain intellectually honest and avoid making unsupported counterclaims.
- Do not argue in favor of the proposition.
"""


def build_opponent_agent():
    """Build and return the Opponent agent chain."""

    model = get_opponent_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", OPPONENT_SYSTEM_PROMPT),
            (
                "human",
                """
Proposition:
{question}

Available evidence:
{evidence}

Proponent's argument:
{proponent_argument}

Proponent's claims:
{proponent_claims}

Critically evaluate the Proponent's argument and claims. Using the available evidence,
identify weaknesses in their reasoning and construct the strongest evidence-supported
counterargument AGAINST the proposition.

Return your response according to the required structured format.
""",
            ),
        ]
    )

    structured_model = model.with_structured_output(OpponentResponse)

    return prompt | structured_model
