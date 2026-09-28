"""Runtime configuration loaded from environment variables (prefix ``RUBRICOPS_``).

Every setting has a safe local default so the test suite and the demo run with no
environment at all. Production deployments override them via env vars or a ``.env``
file (see ``.env.example``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_JWT_SECRET = "dev-only-insecure-secret-change-me"  # noqa: S105 - documented dev default

Environment = Literal["dev", "test", "prod"]


class Settings(BaseSettings):
    """Application settings.

    ``database_url`` accepts any SQLAlchemy URL: SQLite by default, Postgres via
    ``postgresql+psycopg://user:pass@host/db``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RUBRICOPS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Environment = "dev"
    database_url: str = "sqlite:///./var/rubricops.db"
    jwt_secret: SecretStr = SecretStr(DEV_JWT_SECRET)
    jwt_ttl_minutes: int = Field(default=60, ge=1, le=24 * 60)
    qa_sample_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    review_sla_hours: int = Field(default=24, ge=1)
    random_seed: int = 20260929

    @field_validator("database_url")
    @classmethod
    def _require_scheme(cls, value: str) -> str:
        if "://" not in value:
            msg = "database_url must be a SQLAlchemy URL such as sqlite:///path.db"
            raise ValueError(msg)
        return value

    def assert_safe_for_env(self) -> None:
        """Refuse to run in prod with the well-known development JWT secret."""
        if self.env == "prod" and self.jwt_secret.get_secret_value() == DEV_JWT_SECRET:
            msg = "RUBRICOPS_JWT_SECRET must be set to a real secret when RUBRICOPS_ENV=prod"
            raise RuntimeError(msg)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    settings = Settings()
    settings.assert_safe_for_env()
    return settings
