"""LangGraph wiring: proponent -> opponent -> moderator -+-> proponent / -> synthesize -> END."""

from __future__ import annotations

import json
import operator
from typing import Annotated, Callable, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from engine.agents.moderator import Moderator
from engine.agents.opponent import Opponent
from engine.agents.proponent import Proponent
from engine.schemas import (
    ConsensusReport,
    DebateConfig,
    ModeratorVerdict,
    Role,
    Turn,
)

# Active RAG retriever for the current run_debate() call. LangGraph state must
# stay JSON-serializable, so the callable is held here instead of in state.
_active_retriever: Optional[Callable[[str], list[str]]] = None


class DebateState(TypedDict):
    config: DebateConfig
    turns: Annotated[list[Turn], operator.add]
    verdicts: Annotated[list[ModeratorVerdict], operator.add]
    current_round: int
    report: Optional[ConsensusReport]


def _transcript(turns: list[Turn]) -> str:
    return "\n\n".join(f"[{t.role.value} R{t.round_number}]: {t.content}" for t in turns)


def _maybe_attach_citations(turn: Turn, config: DebateConfig) -> Turn:
    if config.use_rag and _active_retriever is not None:
        turn.citations = _active_retriever(turn.content) or []
    return turn


def proponent_node(state: DebateState) -> dict:
    config = state["config"]
    rnd = state["current_round"]
    turn = Proponent().run(
        topic=config.topic,
        transcript=_transcript(state.get("turns", [])),
        persona=config.proponent_persona,
        temperature=config.temperature,
    )
    turn.round_number = rnd
    turn.role = Role.PROPONENT
    return {"turns": [_maybe_attach_citations(turn, config)]}


def opponent_node(state: DebateState) -> dict:
    config = state["config"]
    rnd = state["current_round"]
    turn = Opponent().run(
        topic=config.topic,
        transcript=_transcript(state.get("turns", [])),
        persona=config.opponent_persona,
        temperature=config.temperature,
    )
    turn.round_number = rnd
    turn.role = Role.OPPONENT
    return {"turns": [_maybe_attach_citations(turn, config)]}


def moderator_node(state: DebateState) -> dict:
    config = state["config"]
    rnd = state["current_round"]
    round_turns = [t for t in state.get("turns", []) if t.round_number == rnd]
    verdict = Moderator().run(
        topic=config.topic,
        transcript=_transcript(round_turns),
        round_number=rnd,
    )
    return {"verdicts": [verdict], "current_round": rnd + 1}


def synthesize_node(state: DebateState) -> dict:
    """Build the final ConsensusReport deterministically (no extra LLM call,
    keeping the budget at exactly 3 LLM calls per round)."""
    config = state["config"]
    turns = state.get("turns", [])
    verdicts = state.get("verdicts", [])
    commentaries = " | ".join(v.commentary for v in verdicts)
    fallacy_types = sorted({f.type for v in verdicts for f in v.fallacies_detected})
    agreement = [f"Both sides addressed the thesis with scored rigor ({commentaries[:200]}...)"] if verdicts else []
    # Additional deterministic fields (kept minimal; see tests for exact shape):
    _ = fallacy_types
    report = ConsensusReport(
        topic=config.topic,
        total_rounds=config.rounds,
        points_of_agreement=agreement or ["No verdicts recorded."],
        points_of_conflict=[f"Fallacy flagged: {ft}" for ft in fallacy_types]
        or ["Proponent and opponent disagreed on the thesis across all rounds."],
        recommended_actions=["Review flagged fallacies and strengthen cited evidence."],
        turns=turns,
        verdicts=verdicts,
    )
    return {"report": report}


def _route_after_moderator(state: DebateState) -> str:
    # current_round was already incremented in moderator_node.
    if state["current_round"] <= state["config"].rounds:
        return "proponent"
    return "synthesize"


_builder = StateGraph(DebateState)
_builder.add_node("proponent", proponent_node)
_builder.add_node("opponent", opponent_node)
_builder.add_node("moderator", moderator_node)
_builder.add_node("synthesize", synthesize_node)
_builder.set_entry_point("proponent")
_builder.add_edge("proponent", "opponent")
_builder.add_edge("opponent", "moderator")
_builder.add_conditional_edges("moderator", _route_after_moderator, ["proponent", "synthesize"])
_builder.add_edge("synthesize", END)

debate_app = _builder.compile()


def run_debate(
    config: DebateConfig,
    retriever: Optional[Callable[[str], list[str]]] = None,
) -> ConsensusReport:
    global _active_retriever
    _active_retriever = retriever
    try:
        out = debate_app.invoke(
            {"config": config, "turns": [], "verdicts": [], "current_round": 1}
        )
    finally:
        _active_retriever = None
    report = out.get("report")
    if isinstance(report, ConsensusReport):
        return report
    return ConsensusReport(**report)


if __name__ == "__main__":
    demo_config = DebateConfig(topic="Should AI be regulated by governments?", rounds=2)
    result = run_debate(demo_config)
    print(json.dumps(result.model_dump(), indent=2, default=str))
