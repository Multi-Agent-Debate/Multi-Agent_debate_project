from langchain_groq import ChatGroq

from src.config.settings import settings


def get_proponent_model():
    """Return the Groq model used by the Proponent agent."""
    return ChatGroq(
        model=settings.proponent_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
    )


def get_opponent_model():
    """Return the Groq model used by the Opponent agent."""
    return ChatGroq(
        model=settings.opponent_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
    )


def get_moderator_model():
    """Return the Groq model used by the Moderator agent."""
    return ChatGroq(
        model=settings.moderator_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
    )