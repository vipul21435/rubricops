from __future__ import annotations

import pytest
from pydantic import ValidationError

from rubricops.settings import DEV_JWT_SECRET, Settings, get_settings


def test_defaults_are_local_and_safe() -> None:
    settings = Settings()
    assert settings.env == "dev"
    assert settings.database_url.startswith("sqlite:///")
    assert settings.jwt_secret.get_secret_value() == DEV_JWT_SECRET
    assert 0.0 <= settings.qa_sample_rate <= 1.0


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUBRICOPS_DATABASE_URL", "postgresql+psycopg://u:p@db/rubricops")
    monkeypatch.setenv("RUBRICOPS_QA_SAMPLE_RATE", "0.25")
    settings = get_settings()
    assert settings.database_url == "postgresql+psycopg://u:p@db/rubricops"
    assert settings.qa_sample_rate == pytest.approx(0.25)


def test_secret_is_masked_in_dumps() -> None:
    assert DEV_JWT_SECRET not in repr(Settings())


@pytest.mark.parametrize("rate", ["-0.1", "1.5"])
def test_rejects_out_of_range_sample_rate(monkeypatch: pytest.MonkeyPatch, rate: str) -> None:
    monkeypatch.setenv("RUBRICOPS_QA_SAMPLE_RATE", rate)
    with pytest.raises(ValidationError):
        Settings()


def test_rejects_database_url_without_scheme() -> None:
    with pytest.raises(ValidationError, match="SQLAlchemy URL"):
        Settings(database_url="rubricops.db")


def test_prod_refuses_dev_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUBRICOPS_ENV", "prod")
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        get_settings()


def test_prod_accepts_real_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUBRICOPS_ENV", "prod")
    monkeypatch.setenv("RUBRICOPS_JWT_SECRET", "a-real-secret-from-the-vault")
    assert get_settings().env == "prod"
