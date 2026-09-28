"""Initial schema: users, rubrics and versions, submissions, reviews, gold items, audit log.

rubric_versions and audit_events are append-only: triggers refuse UPDATE and DELETE
on SQLite and Postgres. The trigger SQL is copied here rather than imported, so this
revision stays what it was even if the models change; a test checks that a migrated
database and a create_all database end up with the same triggers.

Revision ID: 0001
Revises:
Create Date: 2026-09-29 03:28:04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APPEND_ONLY_TABLES = ("rubric_versions", "audit_events")


def _trigger_statements(dialect: str, table: str) -> list[str]:
    if dialect == "sqlite":
        return [
            f"CREATE TRIGGER {table}_no_{op_name.lower()} BEFORE {op_name} ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
            for op_name in ("UPDATE", "DELETE")
        ]
    if dialect == "postgresql":
        return [
            "CREATE OR REPLACE FUNCTION rubricops_append_only() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN "
            "RAISE EXCEPTION USING MESSAGE = TG_TABLE_NAME || ' is append-only'; END $$",
            f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION rubricops_append_only()",
        ]
    return []  # pragma: no cover - other backends rely on the ORM guards alone


def _drop_trigger_statements(dialect: str, table: str) -> list[str]:
    if dialect == "sqlite":
        return [f"DROP TRIGGER IF EXISTS {table}_no_{name}" for name in ("update", "delete")]
    if dialect == "postgresql":
        return [f"DROP TRIGGER IF EXISTS {table}_append_only ON {table}"]
    return []  # pragma: no cover


def upgrade() -> None:
    op.create_table(
        "rubrics",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rubrics")),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("handle", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("skill_tags", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('author', 'reviewer', 'lead')", name=op.f("ck_users_role")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("handle", name=op.f("uq_users_handle")),
    )
    op.create_table(
        "audit_events",
        sa.Column("seq", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("entity", sa.String(length=32), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("from_status", sa.String(length=32), nullable=True),
        sa.Column("to_status", sa.String(length=32), nullable=True),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint("seq >= 1", name=op.f("ck_audit_events_seq_positive")),
        sa.ForeignKeyConstraint(
            ["actor_id"], ["users.id"], name=op.f("fk_audit_events_actor_id_users")
        ),
        sa.PrimaryKeyConstraint("seq", name=op.f("pk_audit_events")),
        sa.UniqueConstraint("hash", name=op.f("uq_audit_events_hash")),
        sa.UniqueConstraint("prev_hash", name=op.f("uq_audit_events_prev_hash")),
    )
    with op.batch_alter_table("audit_events", schema=None) as batch_op:
        batch_op.create_index("ix_audit_events_entity", ["entity", "entity_id"], unique=False)

    op.create_table(
        "rubric_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("rubric_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(content_hash) = 64", name=op.f("ck_rubric_versions_content_hash_length")
        ),
        sa.CheckConstraint("version >= 1", name=op.f("ck_rubric_versions_version_positive")),
        sa.ForeignKeyConstraint(
            ["rubric_id"], ["rubrics.id"], name=op.f("fk_rubric_versions_rubric_id_rubrics")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rubric_versions")),
        sa.UniqueConstraint(
            "rubric_id", "version", name=op.f("uq_rubric_versions_rubric_id_version")
        ),
    )
    with op.batch_alter_table("rubric_versions", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_rubric_versions_content_hash"), ["content_hash"], unique=False
        )

    op.create_table(
        "gold_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("rubric_version_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("expected_scores", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name=op.f("fk_gold_items_created_by_users")
        ),
        sa.ForeignKeyConstraint(
            ["rubric_version_id"],
            ["rubric_versions.id"],
            name=op.f("fk_gold_items_rubric_version_id_rubric_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_gold_items")),
    )
    op.create_table(
        "submissions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.Integer(), nullable=False),
        sa.Column("rubric_version_id", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("skill_tags", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=True),
        sa.Column("final_passed", sa.Boolean(), nullable=True),
        sa.Column("final_stage", sa.String(length=16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "final_stage IS NULL OR final_stage IN ('primary', 'qa', 'adjudication')",
            name=op.f("ck_submissions_final_stage"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'assigned', 'in_review', 'reviewed', 'qa_pending', 'qa_passed', 'qa_failed', 'in_adjudication', 'finalized', 'returned_to_author')",
            name=op.f("ck_submissions_status"),
        ),
        sa.CheckConstraint("round >= 1", name=op.f("ck_submissions_round_positive")),
        sa.ForeignKeyConstraint(
            ["author_id"], ["users.id"], name=op.f("fk_submissions_author_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["rubric_version_id"],
            ["rubric_versions.id"],
            name=op.f("fk_submissions_rubric_version_id_rubric_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_submissions")),
    )
    with op.batch_alter_table("submissions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_submissions_author_id"), ["author_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_submissions_status"), ["status"], unique=False)

    op.create_table(
        "assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("submission_id", sa.Integer(), nullable=False),
        sa.Column("reviewer_id", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "stage IN ('primary', 'qa', 'adjudication')", name=op.f("ck_assignments_stage")
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_id"], ["users.id"], name=op.f("fk_assignments_reviewer_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["submission_id"],
            ["submissions.id"],
            name=op.f("fk_assignments_submission_id_submissions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_assignments")),
    )
    with op.batch_alter_table("assignments", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_assignments_reviewer_id"), ["reviewer_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_assignments_submission_id"), ["submission_id"], unique=False
        )

    op.create_table(
        "reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("submission_id", sa.Integer(), nullable=False),
        sa.Column("reviewer_id", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("rubric_version_id", sa.Integer(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "stage IN ('primary', 'qa', 'adjudication')", name=op.f("ck_reviews_stage")
        ),
        sa.CheckConstraint("score >= 0 AND score <= 1", name=op.f("ck_reviews_score_range")),
        sa.ForeignKeyConstraint(
            ["reviewer_id"], ["users.id"], name=op.f("fk_reviews_reviewer_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["rubric_version_id"],
            ["rubric_versions.id"],
            name=op.f("fk_reviews_rubric_version_id_rubric_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["submission_id"], ["submissions.id"], name=op.f("fk_reviews_submission_id_submissions")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reviews")),
        sa.UniqueConstraint(
            "submission_id", "round", "stage", name=op.f("uq_reviews_submission_id_round_stage")
        ),
    )
    with op.batch_alter_table("reviews", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_reviews_reviewer_id"), ["reviewer_id"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_reviews_submission_id"), ["submission_id"], unique=False
        )

    dialect = op.get_context().dialect.name
    for table in APPEND_ONLY_TABLES:
        for statement in _trigger_statements(dialect, table):
            op.execute(statement)


def downgrade() -> None:
    dialect = op.get_context().dialect.name
    for table in APPEND_ONLY_TABLES:
        for statement in _drop_trigger_statements(dialect, table):
            op.execute(statement)
    if dialect == "postgresql":
        op.execute("DROP FUNCTION IF EXISTS rubricops_append_only()")

    with op.batch_alter_table("reviews", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_reviews_submission_id"))
        batch_op.drop_index(batch_op.f("ix_reviews_reviewer_id"))

    op.drop_table("reviews")
    with op.batch_alter_table("assignments", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_assignments_submission_id"))
        batch_op.drop_index(batch_op.f("ix_assignments_reviewer_id"))

    op.drop_table("assignments")
    with op.batch_alter_table("submissions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_submissions_status"))
        batch_op.drop_index(batch_op.f("ix_submissions_author_id"))

    op.drop_table("submissions")
    op.drop_table("gold_items")
    with op.batch_alter_table("rubric_versions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_rubric_versions_content_hash"))

    op.drop_table("rubric_versions")
    with op.batch_alter_table("audit_events", schema=None) as batch_op:
        batch_op.drop_index("ix_audit_events_entity")

    op.drop_table("audit_events")
    op.drop_table("users")
    op.drop_table("rubrics")
