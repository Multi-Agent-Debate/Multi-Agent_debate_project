from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.agents.llm_models import get_proponent_model


class ProponentResponse(BaseModel):
    """Structured response produced by the Proponent agent."""

    argument: str = Field(
        description="The main argument supporting the proposition."
    )

    claims: list[str] = Field(
        description="Specific claims made in support of the proposition."
    )

    reasoning: str = Field(
        description="The reasoning that connects the claims to the proposition."
    )

    evidence_used: list[str] = Field(
        description="Evidence from the provided sources that supports the claims."
    )

    limitations: list[str] = Field(
        description="Important limitations, uncertainties, or weaknesses in the available evidence."
    )


PROPONENT_SYSTEM_PROMPT = """
You are the Proponent in an AI debate system.

Your role is to argue IN FAVOR of the proposition.

Your objective is to construct the strongest possible argument
using only the available evidence.

You must:

- Clearly support the proposition.
- Make specific and logically connected claims.
- Explain the reasoning behind your claims.
- Use the provided evidence whenever possible.
- Identify which evidence supports which claims.
- Distinguish evidence from your own reasoning.
- Never invent facts, statistics, sources, quotations, or citations.
- Never claim that evidence exists when it was not provided.
- If the available evidence is insufficient, explicitly acknowledge this.
- Identify important limitations or uncertainties in the evidence.
- Do not argue against the proposition. The Opponent agent will handle that role.
"""


def build_proponent_agent():
    """Build and return the Proponent agent chain."""

    model = get_proponent_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROPONENT_SYSTEM_PROMPT),
            (
                "human",
                """
Proposition:
{question}

Available evidence:
{evidence}

Using the available evidence, construct the strongest
evidence-supported argument IN FAVOR of the proposition.

Return your response according to the required structured format.
""",
            ),
        ]
    )

    structured_model = model.with_structured_output(ProponentResponse)

    return prompt | structured_model