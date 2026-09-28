"""Alembic: upgrade, downgrade, upgrade again, and no drift from the models."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, text

from rubricops.db.engine import make_engine
from rubricops.db.migrate import current_revision, downgrade, upgrade, upgrade_sql
from rubricops.db.models import Base

TABLES = {
    "assignments",
    "audit_events",
    "gold_items",
    "reviews",
    "rubric_versions",
    "rubrics",
    "submissions",
    "users",
}


def _tables(url: str) -> set[str]:
    engine = make_engine(url)
    try:
        return set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()


def _triggers(url: str) -> dict[str, str]:
    engine = make_engine(url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT name, sql FROM sqlite_master WHERE type='trigger'"))
            return {name: " ".join(sql.split()) for name, sql in rows}
    finally:
        engine.dispose()


@pytest.fixture
def url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'migrated.db'}"


def test_upgrade_downgrade_upgrade(url: str) -> None:
    assert current_revision(url) is None
    upgrade(url)
    assert current_revision(url) == "0001"
    assert _tables(url) == TABLES
    assert len(_triggers(url)) == 4
    downgrade(url, "base")
    assert current_revision(url) is None
    assert _tables(url) == set()
    assert _triggers(url) == {}
    upgrade(url)
    assert _tables(url) == TABLES


def test_no_drift_between_migrations_and_models(url: str) -> None:
    upgrade(url)
    engine = make_engine(url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn, opts={"compare_type": True})
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()


def test_migrated_triggers_match_create_all(url: str, tmp_path: Path) -> None:
    upgrade(url)
    created = f"sqlite:///{tmp_path / 'created.db'}"
    engine = make_engine(created)
    Base.metadata.create_all(engine)
    engine.dispose()
    assert _triggers(url) == _triggers(created)
    assert set(_triggers(url)) == {
        "audit_events_no_update",
        "audit_events_no_delete",
        "rubric_versions_no_update",
        "rubric_versions_no_delete",
    }


def test_url_falls_back_to_settings(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("RUBRICOPS_DATABASE_URL", url)
    upgrade()
    assert current_revision() == "0001"


def test_offline_sql(capsys: pytest.CaptureFixture[str]) -> None:
    upgrade_sql("sqlite:///offline.db")
    sql = capsys.readouterr().out
    assert "CREATE TABLE audit_events" in sql
    assert "CREATE TRIGGER audit_events_no_update" in sql


def test_percent_in_url_survives_config_interpolation(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'with%25percent.db'}"
    upgrade(url)
    assert current_revision(url) == "0001"
