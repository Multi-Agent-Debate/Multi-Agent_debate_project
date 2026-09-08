from src.agents.moderator import build_moderator_agent


moderator = build_moderator_agent()

result = moderator.invoke(
    {
        "question": "Should college students use AI for studying?",
        "evidence": """
A hypothetical study reports that students using AI tutoring tools
received faster explanations and additional practice opportunities.
The study also notes that excessive dependence on AI may reduce
independent problem-solving practice.
""",
        "debate_history": """
PROPONENT:
AI can improve study efficiency by providing faster explanations
and additional opportunities for practice.

Claims:
- AI can provide faster explanations.
- AI can provide additional practice opportunities.
- AI can improve study efficiency.

OPPONENT:
The benefits of AI may be outweighed by reduced independent
problem-solving if students become overly dependent on AI.

Weaknesses:
- Faster explanations do not necessarily prove better learning outcomes.
- The evidence does not establish long-term academic improvement.

Counterclaims:
- Convenience does not necessarily result in deeper understanding.
- Excessive AI dependence may reduce independent problem-solving practice.
""",
    }
)

print("\n=== MODERATOR RESPONSE ===\n")

print("Summary:")
print(result.summary)

print("\nAssessment:")
print(result.assessment)

print("\nConclusion:")
print(result.conclusion)

print("\nConfidence:")
print(result.confidence)
