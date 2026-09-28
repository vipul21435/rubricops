from __future__ import annotations

import pytest
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rubricops.db.models import (
    GENESIS_HASH,
    AppendOnlyError,
    AuditEvent,
    Rubric,
    RubricVersion,
    Submission,
    User,
)
from tests.db.conftest import T0


def _event(seq: int = 1) -> AuditEvent:
    return AuditEvent(
        seq=seq,
        occurred_at=T0,
        actor_id=None,
        action="create",
        entity="submission",
        entity_id=1,
        from_status=None,
        to_status="queued",
        data={},
        prev_hash=GENESIS_HASH if seq == 1 else f"{seq - 1:064d}",
        hash=f"{seq:064d}",
    )


def _version(session: Session) -> RubricVersion:
    session.add(Rubric(id="r", title="R", created_at=T0))
    version = RubricVersion(
        rubric_id="r", version=1, content_hash="a" * 64, body="{}", message="m", created_at=T0
    )
    session.add(version)
    session.flush()
    return version


def test_role_check_constraint(session: Session) -> None:
    session.add(User(handle="x", role="superuser", created_at=T0))
    with pytest.raises(IntegrityError, match="ck_users_role"):
        session.flush()


def test_status_check_constraint(session: Session) -> None:
    author = User(handle="a", role="author", created_at=T0)
    session.add(author)
    version = _version(session)
    session.add(
        Submission(
            author_id=author.id,
            rubric_version_id=version.id,
            payload={},
            status="lost",
            created_at=T0,
            updated_at=T0,
        )
    )
    with pytest.raises(IntegrityError, match="ck_submissions_status"):
        session.flush()


def test_foreign_keys_are_enforced(session: Session) -> None:
    session.add(
        Submission(
            author_id=404,
            rubric_version_id=404,
            payload={},
            status="queued",
            created_at=T0,
            updated_at=T0,
        )
    )
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        session.flush()


def test_rubric_versions_are_unique_per_rubric(session: Session) -> None:
    _version(session)
    session.add(
        RubricVersion(
            rubric_id="r", version=1, content_hash="b" * 64, body="{}", message="m", created_at=T0
        )
    )
    with pytest.raises(IntegrityError, match="UNIQUE"):
        session.flush()


@pytest.mark.parametrize("model", ["event", "version"])
def test_orm_refuses_update_and_delete(session: Session, model: str) -> None:
    row: AuditEvent | RubricVersion
    if model == "event":
        row = _event()
        session.add(row)
        session.flush()
    else:
        row = _version(session)
    session.commit()
    if isinstance(row, AuditEvent):
        row.action = "edited"
    else:
        row.message = "edited"
    with pytest.raises(AppendOnlyError, match="append-only: UPDATE refused"):
        session.flush()
    session.rollback()
    session.delete(row)
    with pytest.raises(AppendOnlyError, match="append-only: DELETE refused"):
        session.flush()


def test_orm_refuses_bulk_update_and_delete(session: Session) -> None:
    session.add(_event())
    session.commit()
    with pytest.raises(AppendOnlyError, match="audit_events is append-only: UPDATE"):
        session.execute(update(AuditEvent).values(action="edited"))
    with pytest.raises(AppendOnlyError, match="audit_events is append-only: DELETE"):
        session.execute(delete(AuditEvent))
    # Bulk writes to ordinary tables are unaffected.
    session.execute(update(User).values(handle="unused"))


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE audit_events SET action = 'edited'",
        "DELETE FROM audit_events",
        "UPDATE rubric_versions SET message = 'edited'",
        "DELETE FROM rubric_versions",
    ],
)
def test_database_triggers_refuse_raw_sql(session: Session, statement: str) -> None:
    session.add(_event())
    _version(session)
    session.commit()
    with pytest.raises(IntegrityError, match="is append-only"):
        session.execute(text(statement))


def test_audit_chain_cannot_fork(session: Session) -> None:
    session.add(_event(1))
    session.commit()
    fork = _event(2)
    fork.prev_hash = GENESIS_HASH  # a second child of the genesis link
    session.add(fork)
    with pytest.raises(IntegrityError, match="UNIQUE"):
        session.flush()


def test_timestamps_come_back_aware(session: Session) -> None:
    session.add(_event())
    session.commit()
    session.expire_all()
    stored = session.scalars(select(AuditEvent)).one()
    assert stored.occurred_at == T0
    assert stored.occurred_at.tzinfo is not None
