from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from rubricops.db.engine import make_engine, make_session_factory
from rubricops.db.models import Base

T0 = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)


@pytest.fixture
def engine() -> Iterator[Engine]:
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with make_session_factory(engine)() as session:
        yield session
