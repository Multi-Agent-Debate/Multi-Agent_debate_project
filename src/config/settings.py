from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    groq_api_key: str
    mistral_api_key: str

    # Model Configuration
    proponent_model: str = "openai/gpt-oss-120b"
    opponent_model: str = "qwen/qwen3.6-27b"
    moderator_model: str = "mistral-small-latest"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()