"""Shared fixtures: every test starts with a clean settings cache and environment."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rubricops.settings import get_settings


@pytest.fixture(autouse=True)
def _isolate_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    for name in (
        "ENV",
        "DATABASE_URL",
        "JWT_SECRET",
        "JWT_TTL_MINUTES",
        "QA_SAMPLE_RATE",
    ):
        monkeypatch.delenv(f"RUBRICOPS_{name}", raising=False)
    # Never pick up a developer's local .env during tests.
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
