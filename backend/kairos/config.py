"""Runtime configuration. Everything comes from the environment; nothing is hardcoded."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- app ---------------------------------------------------------------
    kairos_env: str = "dev"
    kairos_log_level: str = "INFO"
    kairos_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- postgres ----------------------------------------------------------
    postgres_user: str = "kairos"
    postgres_password: str = "kairos"
    postgres_db: str = "kairos"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # --- mlflow ------------------------------------------------------------
    # MLflow is a SOFT dependency: a dead tracking server must never take down
    # a run. Training logs best-effort and degrades to Postgres.
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "kairos"

    # --- llm ---------------------------------------------------------------
    # Absent key is a supported mode, not an error: diagnosis falls back to
    # rules and narration to deterministic templates.
    anthropic_api_key: str = ""
    kairos_llm_model: str = "claude-sonnet-5"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def checkpointer_url(self) -> str:
        """Plain libpq URL — langgraph's Postgres checkpointer wants psycopg, not SQLAlchemy."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.kairos_cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
