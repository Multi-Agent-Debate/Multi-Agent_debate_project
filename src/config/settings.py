"""Application settings and environment configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Multi-Agent Debate Engine and RAG subsystem."""

    # API Keys
    # Defaults prevent validation crashes when running in offline mode.
    groq_api_key: str = ""
    mistral_api_key: str = ""
    openai_api_key: str = ""

    # Model Configuration
    proponent_model: str = "openai/gpt-oss-120b"
    opponent_model: str = "qwen/qwen3.6-27b"
    moderator_model: str = "openai/gpt-oss-20b"
    default_model_name: str = "gpt-4o-mini"
    max_debate_rounds: int = 3

    # RAG Pipeline Configuration
    chroma_persist_directory: str = "./data/vector_db"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retriever_top_k: int = 5

    # Logging Configuration
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()