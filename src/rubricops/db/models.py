"""SQLAlchemy 2 models for review operations.

Every table is plain relational data with CHECK constraints for its enumerations,
so the database refuses a bad status even if a bug slips past the service layer.
Two tables are append-only, :class:`RubricVersion` and :class:`AuditEvent`: the ORM
refuses to UPDATE or DELETE them (per object and in bulk), and database triggers
refuse the same at the SQL level (created by both ``create_all`` and the Alembic
migration).

Timestamps are timezone-aware UTC on every backend (:class:`UTCDateTime`).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DDL,
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Mapper,
    ORMExecuteState,
    Session,
    mapped_column,
)
from sqlalchemy.schema import ExecutableDDLElement

from rubricops.db.types import UTCDateTime
from rubricops.domain.pipeline import Role, Stage, Status

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

#: sha256 of the empty history: the ``prev_hash`` of the first audit event.
GENESIS_HASH = "0" * 64


def _one_of(column: str, values: type[StrEnum]) -> str:
    allowed = ", ".join(f"'{member.value}'" for member in values)
    return f"{column} IN ({allowed})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy reads this class attribute
        datetime: UTCDateTime(),
        dict[str, Any]: JSON(),
        dict[str, int]: JSON(),
        list[str]: JSON(),
    }


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_one_of("role", Role), name="role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    handle: Mapped[str] = mapped_column(String(64), unique=True)
    role: Mapped[str] = mapped_column(String(16))
    skill_tags: Mapped[list[str]] = mapped_column(default=list)
    created_at: Mapped[datetime]


class Rubric(Base):
    __tablename__ = "rubrics"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime]


class RubricVersion(Base):
    """One immutable published version: the canonical JSON body and its sha256."""

    __tablename__ = "rubric_versions"
    __table_args__ = (
        UniqueConstraint("rubric_id", "version"),
        CheckConstraint("version >= 1", name="version_positive"),
        CheckConstraint("length(content_hash) = 64", name="content_hash_length"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    rubric_id: Mapped[str] = mapped_column(ForeignKey("rubrics.id"))
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    body: Mapped[str] = mapped_column(Text)
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]


class Submission(Base):
    """An item under review, pinned to the rubric version it is judged against.

    ``version`` is the optimistic-lock counter: SQLAlchemy adds ``WHERE version = n``
    to every UPDATE and bumps it, so two writers that read the same state cannot
    both apply a transition. ``round`` starts at 1 and grows each time the author
    resubmits, so reviews of an earlier draft never count towards the current one.
    """

    __tablename__ = "submissions"
    __table_args__ = (
        CheckConstraint(_one_of("status", Status), name="status"),
        CheckConstraint("round >= 1", name="round_positive"),
        CheckConstraint(
            f"final_stage IS NULL OR {_one_of('final_stage', Stage)}", name="final_stage"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    rubric_version_id: Mapped[int] = mapped_column(ForeignKey("rubric_versions.id"))
    payload: Mapped[dict[str, Any]]
    skill_tags: Mapped[list[str]] = mapped_column(default=list)
    status: Mapped[str] = mapped_column(String(32), index=True)
    round: Mapped[int] = mapped_column(Integer, default=1)
    final_score: Mapped[float | None] = mapped_column(Float)
    final_passed: Mapped[bool | None] = mapped_column(Boolean)
    final_stage: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    __mapper_args__ = {"version_id_col": version}  # noqa: RUF012 - SQLAlchemy mapper config


class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (CheckConstraint(_one_of("stage", Stage), name="stage"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), index=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    stage: Mapped[str] = mapped_column(String(16))
    round: Mapped[int] = mapped_column(Integer)
    assigned_at: Mapped[datetime]
    due_at: Mapped[datetime | None]
    released_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]


class Review(Base):
    """A scored review, pinned to the rubric version its scores were validated against."""

    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("submission_id", "round", "stage"),
        CheckConstraint(_one_of("stage", Stage), name="stage"),
        CheckConstraint("score >= 0 AND score <= 1", name="score_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), index=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    stage: Mapped[str] = mapped_column(String(16))
    round: Mapped[int] = mapped_column(Integer)
    rubric_version_id: Mapped[int] = mapped_column(ForeignKey("rubric_versions.id"))
    scores: Mapped[dict[str, int]]
    score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]


class GoldItem(Base):
    """An item with known per-criterion scores, used to calibrate reviewers."""

    __tablename__ = "gold_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    rubric_version_id: Mapped[int] = mapped_column(ForeignKey("rubric_versions.id"))
    title: Mapped[str] = mapped_column(String(200))
    payload: Mapped[dict[str, Any]]
    expected_scores: Mapped[dict[str, int]]
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime]


class AuditEvent(Base):
    """One link of the hash chain: ``hash = sha256(prev_hash + canonical event)``.

    ``seq`` is assigned by the writer (previous seq + 1) rather than by the database,
    so it can be part of the hashed content; the UNIQUE constraints on ``prev_hash``
    and ``hash`` make a forked chain impossible to commit.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        CheckConstraint("seq >= 1", name="seq_positive"),
        Index("ix_audit_events_entity", "entity", "entity_id"),
    )

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    occurred_at: Mapped[datetime]
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(32))
    entity: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[int] = mapped_column(Integer)
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str | None] = mapped_column(String(32))
    data: Mapped[dict[str, Any]]
    prev_hash: Mapped[str] = mapped_column(String(64), unique=True)
    hash: Mapped[str] = mapped_column(String(64), unique=True)


APPEND_ONLY_MODELS: tuple[type[Base], ...] = (RubricVersion, AuditEvent)


class AppendOnlyError(RuntimeError):
    """An attempt to change or remove a row of an append-only table."""


def _refuse(kind: str, table: str) -> AppendOnlyError:
    return AppendOnlyError(f"{table} is append-only: {kind} refused")


def _install_orm_guards(model: type[Base]) -> None:
    table = model.__tablename__

    @event.listens_for(model, "before_update")
    def _no_update(_mapper: Mapper[Any], _connection: object, _target: object) -> None:
        raise _refuse("UPDATE", table)

    @event.listens_for(model, "before_delete")
    def _no_delete(_mapper: Mapper[Any], _connection: object, _target: object) -> None:
        raise _refuse("DELETE", table)


for _model in APPEND_ONLY_MODELS:
    _install_orm_guards(_model)


@event.listens_for(Session, "do_orm_execute")
def _no_bulk_writes(state: ORMExecuteState) -> None:
    """Refuse ``session.execute(update(AuditEvent))`` and friends, which skip mapper events."""
    if not (state.is_update or state.is_delete):
        return
    mapper = state.bind_mapper
    if mapper is not None and mapper.class_ in APPEND_ONLY_MODELS:
        kind = "UPDATE" if state.is_update else "DELETE"
        raise _refuse(kind, mapper.class_.__tablename__)


def sqlite_trigger_sql(table: str) -> tuple[str, ...]:
    """The SQLite triggers that make ``table`` append-only."""
    return tuple(
        f"CREATE TRIGGER {table}_no_{op.lower()} BEFORE {op} ON {table} "
        f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
        for op in ("UPDATE", "DELETE")
    )


POSTGRES_FUNCTION_SQL = (
    "CREATE OR REPLACE FUNCTION rubricops_append_only() RETURNS trigger "
    "LANGUAGE plpgsql AS $$ BEGIN "
    "RAISE EXCEPTION USING MESSAGE = TG_TABLE_NAME || ' is append-only'; END $$"
)


def postgres_trigger_sql(table: str) -> tuple[str, ...]:
    """The Postgres function and trigger that make ``table`` append-only."""
    return (
        POSTGRES_FUNCTION_SQL,
        f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
        "FOR EACH ROW EXECUTE FUNCTION rubricops_append_only()",
    )


def _ddl(statement: str, dialect: str) -> ExecutableDDLElement:
    return DDL(statement).execute_if(dialect=dialect)  # type: ignore[no-untyped-call]


def _attach_triggers(table: Table) -> None:
    for statement in sqlite_trigger_sql(table.name):
        event.listen(table, "after_create", _ddl(statement, "sqlite"))
    for statement in postgres_trigger_sql(table.name):
        event.listen(table, "after_create", _ddl(statement, "postgresql"))


for _model in APPEND_ONLY_MODELS:
    _attach_triggers(_model.__table__)  # type: ignore[arg-type]
