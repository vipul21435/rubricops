from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import Column, MetaData, String, Table, create_engine, insert, select, text
from sqlalchemy.exc import StatementError

from rubricops.db import engine as engine_module
from rubricops.db.engine import make_engine, make_session_factory, session_scope
from rubricops.db.models import Rubric
from rubricops.db.types import UTCDateTime
from tests.db.conftest import T0


def test_file_sqlite_gets_pragmas_and_its_directory(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "dir" / "r.db"
    engine = make_engine(f"sqlite:///{db}")
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 5000
    engine.dispose()
    assert db.exists()


def test_memory_sqlite_is_one_shared_database() -> None:
    engine = make_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (x INTEGER)"))
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    with engine.connect() as other:
        assert other.execute(text("SELECT count(*) FROM t")).scalar() == 0
    engine.dispose()


def test_other_backends_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_create_engine(url: str, **kwargs: object) -> str:
        calls.append((url, kwargs))
        return "engine"

    monkeypatch.setattr(engine_module, "create_engine", fake_create_engine)
    url = "postgresql+psycopg://user:pw@localhost/rubricops"
    assert make_engine(url) == "engine"  # type: ignore[comparison-overlap]
    assert calls == [(url, {"echo": False})]


def test_session_scope_commits_or_rolls_back() -> None:
    engine = make_engine("sqlite://")
    Rubric.metadata.create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        session.add(Rubric(id="kept", title="Kept", created_at=T0))

    def failing_unit_of_work() -> None:
        with session_scope(factory) as session:
            session.add(Rubric(id="dropped", title="Dropped", created_at=T0))
            session.flush()
            raise RuntimeError

    with pytest.raises(RuntimeError):
        failing_unit_of_work()
    with session_scope(factory) as session:
        assert session.scalars(select(Rubric.id)).all() == ["kept"]


def test_utc_datetime_round_trips_as_aware_utc() -> None:
    metadata = MetaData()
    table = Table("t", metadata, Column("k", String, primary_key=True), Column("at", UTCDateTime))
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    ist = timezone(timedelta(hours=5, minutes=30))
    with engine.begin() as conn:
        conn.execute(insert(table), [{"k": "a", "at": datetime(2026, 9, 1, 15, 0, tzinfo=ist)}])
        conn.execute(insert(table), [{"k": "b", "at": None}])
        rows = dict(conn.execute(select(table.c.k, table.c.at)).all())  # type: ignore[arg-type]
    assert rows["a"] == datetime(2026, 9, 1, 9, 30, tzinfo=UTC)
    assert rows["a"].tzinfo is UTC
    assert rows["b"] is None
    with pytest.raises(StatementError, match="naive"), engine.begin() as conn:
        conn.execute(insert(table), [{"k": "c", "at": datetime(2026, 9, 1)}])  # noqa: DTZ001


def test_utc_datetime_converts_aware_results() -> None:
    kind = UTCDateTime()
    ist = timezone(timedelta(hours=5, minutes=30))
    engine = create_engine("sqlite://")
    value = kind.process_result_value(datetime(2026, 9, 1, 15, 0, tzinfo=ist), engine.dialect)
    assert value == datetime(2026, 9, 1, 9, 30, tzinfo=UTC)
    assert value is not None
    assert value.tzinfo is UTC
