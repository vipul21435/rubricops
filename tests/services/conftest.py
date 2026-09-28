from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from rubricops.db.engine import make_engine, make_session_factory
from rubricops.db.models import Base
from rubricops.domain.clock import SteppingClock
from rubricops.domain.pipeline import Role
from rubricops.services.pipeline import PipelineService
from tests.factories import make_rubric


@dataclass
class World:
    engine: Engine
    sessions: sessionmaker[Session]
    service: PipelineService
    author: int
    reviewer1: int
    reviewer2: int
    lead1: int
    lead2: int
    rubric_v1: int


def build_world() -> World:
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    sessions = make_session_factory(engine)
    service = PipelineService(sessions, SteppingClock(datetime(2026, 9, 1, 9, 0, tzinfo=UTC)))
    lead1 = service.add_user("lead-1", Role.LEAD)
    return World(
        engine=engine,
        sessions=sessions,
        service=service,
        author=service.add_user("author-1", Role.AUTHOR, ["python"]),
        reviewer1=service.add_user("reviewer-1", Role.REVIEWER, ["python"]),
        reviewer2=service.add_user("reviewer-2", Role.REVIEWER),
        lead1=lead1,
        lead2=service.add_user("lead-2", Role.LEAD),
        rubric_v1=service.publish_rubric(make_rubric(), "first version", actor_id=lead1),
    )


@pytest.fixture
def world() -> Iterator[World]:
    built = build_world()
    yield built
    built.engine.dispose()
