"""The review pipeline service: the pure state machine applied to the database.

Every public method is one transaction. :meth:`PipelineService.apply` loads the
submission, checks the caller's ``expected_version`` (optimistic locking), gathers
the facts of the current review round, asks :func:`rubricops.domain.pipeline.decide`
for the next status, and then, all or nothing:

- writes a :class:`~rubricops.db.models.Review` for reviewing actions, scored with
  exact arithmetic against the rubric version the submission is pinned to (never
  the rubric's latest version);
- records the final score when the item is finalized: the primary review's when QA
  passed or was not sampled, the adjudicated review's otherwise;
- appends one hash-chained :class:`~rubricops.db.models.AuditEvent`.

A second writer that read the same version gets :class:`StaleSubmission`, whether
it arrives after the first commit (the version check) or races it: SQLAlchemy's
version counter makes the UPDATE match no row, and a racing reviewing action that
loses on the ``reviews`` unique key (or a racing append on ``audit_events``) is
reported the same way, since every other constraint is checked before writing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.exc import StaleDataError

from rubricops.db.engine import session_scope
from rubricops.db.models import Assignment, AuditEvent, Review, Submission, User
from rubricops.db.models import Rubric as RubricRow
from rubricops.db.models import RubricVersion as RubricVersionRow
from rubricops.domain.clock import Clock
from rubricops.domain.pipeline import (
    REVIEW_STAGE,
    TRANSITIONS,
    Action,
    Actor,
    Command,
    Facts,
    IllegalTransition,
    PipelinePolicy,
    Role,
    Stage,
    Status,
    decide,
)
from rubricops.domain.rubric import Rubric
from rubricops.domain.scoring import ScoreResult, exact_score, score_review
from rubricops.domain.sla import SlaPolicy
from rubricops.domain.versioning import UnchangedRubricError, canonical_json, content_hash
from rubricops.services.audit import append_event


class NotFoundError(LookupError):
    """No row with that id."""


class StaleSubmission(Exception):  # noqa: N818 - named for what the caller sees
    """The submission changed since the caller read it; reload and retry."""

    def __init__(self, submission_id: int, expected: int, actual: int | None) -> None:
        self.submission_id = submission_id
        self.expected = expected
        self.actual = actual
        now = "changed concurrently" if actual is None else f"is at version {actual}"
        super().__init__(
            f"submission {submission_id} {now}, not version {expected}; reload and retry"
        )


class RoleRequiredError(PermissionError):
    """A setup action (publishing a rubric, submitting an item) by the wrong role."""


class ScoresRequired(ValueError):  # noqa: N818
    """A reviewing action was sent without scores, or scores were sent with another action."""


class StoredDataError(RuntimeError):
    """Stored data does not match its own hash."""


@dataclass(frozen=True, slots=True)
class TransitionResult:
    submission_id: int
    action: Action
    from_status: Status
    to_status: Status
    version: int
    audit_seq: int
    review_id: int | None = None
    score: float | None = None
    passed: bool | None = None


@dataclass(frozen=True, slots=True)
class _Round:
    assignment: Assignment | None
    primary: Review | None
    qa: Review | None


class PipelineService:
    """Users, rubric versions, submissions and transitions, each in one transaction."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        clock: Clock,
        *,
        policy: PipelinePolicy | None = None,
        review_sla: timedelta = timedelta(hours=24),
        sla: SlaPolicy | None = None,
    ) -> None:
        """``sla`` (default turnaround plus per-rubric overrides) wins over ``review_sla``."""
        self._sessions = sessions
        self._clock = clock
        self._policy = policy or PipelinePolicy()
        self._sla = sla or SlaPolicy(review_sla)

    # -- setup ------------------------------------------------------------------

    def add_user(self, handle: str, role: Role, skill_tags: Sequence[str] = ()) -> int:
        with session_scope(self._sessions) as session:
            user = User(
                handle=handle, role=role.value, skill_tags=list(skill_tags), created_at=self._now()
            )
            session.add(user)
            session.flush()
            append_event(
                session,
                self._clock,
                actor_id=None,
                action="add_user",
                entity="user",
                entity_id=user.id,
                data={"handle": handle, "role": role.value},
            )
            return user.id

    def publish_rubric(self, rubric: Rubric, message: str, *, actor_id: int) -> int:
        """Store ``rubric`` as the next immutable version and return its row id.

        Only a lead publishes. An unchanged head is refused, as in the in-memory
        registry.
        """
        with session_scope(self._sessions) as session:
            actor = self._actor(session, actor_id)
            if actor.role is not Role.LEAD:
                msg = f"user {actor_id} is a {actor.role.value}; only a lead publishes rubrics"
                raise RoleRequiredError(msg)
            digest = content_hash(rubric)
            head = session.scalars(
                select(RubricVersionRow)
                .where(RubricVersionRow.rubric_id == rubric.id)
                .order_by(RubricVersionRow.version.desc())
                .limit(1)
            ).first()
            if head is None:
                session.add(RubricRow(id=rubric.id, title=rubric.title, created_at=self._now()))
            elif head.content_hash == digest:
                msg = f"{rubric.id} is unchanged since v{head.version}; nothing to publish"
                raise UnchangedRubricError(msg)
            row = RubricVersionRow(
                rubric_id=rubric.id,
                version=1 if head is None else head.version + 1,
                content_hash=digest,
                body=canonical_json(rubric),
                message=message,
                created_at=self._now(),
            )
            session.add(row)
            session.flush()
            append_event(
                session,
                self._clock,
                actor_id=actor_id,
                action="publish_rubric",
                entity="rubric_version",
                entity_id=row.id,
                data={"rubric_id": rubric.id, "version": row.version, "content_hash": digest},
            )
            return row.id

    def submit(
        self,
        *,
        author_id: int,
        rubric_version_id: int,
        payload: Mapping[str, Any],
        skill_tags: Sequence[str] = (),
    ) -> int:
        """Queue a new submission pinned to ``rubric_version_id``; only authors submit."""
        with session_scope(self._sessions) as session:
            author = self._actor(session, author_id)
            if author.role is not Role.AUTHOR:
                msg = f"user {author_id} is a {author.role.value}; only an author submits"
                raise RoleRequiredError(msg)
            self._rubric_version(session, rubric_version_id)
            now = self._now()
            submission = Submission(
                author_id=author_id,
                rubric_version_id=rubric_version_id,
                payload=dict(payload),
                skill_tags=list(skill_tags),
                status=Status.QUEUED.value,
                round=1,
                created_at=now,
                updated_at=now,
            )
            session.add(submission)
            session.flush()
            append_event(
                session,
                self._clock,
                actor_id=author_id,
                action="submit",
                entity="submission",
                entity_id=submission.id,
                to_status=Status.QUEUED.value,
                data={"rubric_version_id": rubric_version_id, "version": submission.version},
            )
            return submission.id

    # -- transitions ------------------------------------------------------------

    def apply(
        self,
        submission_id: int,
        action: Action,
        *,
        actor_id: int,
        expected_version: int,
        assignee_id: int | None = None,
        scores: Mapping[str, object] | None = None,
        comment: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> TransitionResult:
        """Apply ``action`` to a submission the caller last saw at ``expected_version``.

        ``context`` is stored on the audit event under ``"context"``, for example the
        queue policy that chose an assignee or the reasons behind a QA decision.
        """
        try:
            with session_scope(self._sessions) as session:
                return self._apply(
                    session,
                    submission_id,
                    action,
                    actor_id=actor_id,
                    expected_version=expected_version,
                    assignee_id=assignee_id,
                    scores=scores,
                    comment=comment,
                    context=context,
                )
        except (StaleDataError, IntegrityError) as exc:
            # IntegrityError: a concurrent writer inserted this round's review (or the
            # next audit link) first. Users, rubric versions and scores are all
            # checked before any write, so a unique key is the only way to get here.
            raise StaleSubmission(submission_id, expected_version, None) from exc

    def _apply(
        self,
        session: Session,
        submission_id: int,
        action: Action,
        *,
        actor_id: int,
        expected_version: int,
        assignee_id: int | None,
        scores: Mapping[str, object] | None,
        comment: str | None,
        context: Mapping[str, Any] | None,
    ) -> TransitionResult:
        submission = session.get(Submission, submission_id, with_for_update=True)
        if submission is None:
            raise NotFoundError(f"no submission {submission_id}")
        if submission.version != expected_version:
            raise StaleSubmission(submission_id, expected_version, submission.version)
        source = Status(submission.status)
        if (source, action) not in TRANSITIONS:
            raise IllegalTransition(source, action)

        stage = REVIEW_STAGE.get(action)
        if (stage is None) != (scores is None):
            needs = "needs" if stage is not None else "does not take"
            raise ScoresRequired(f"{action.value} {needs} criterion scores")

        actor = self._actor(session, actor_id)
        assignee = self._actor(session, assignee_id) if assignee_id is not None else None
        current = self._round(session, submission)
        result, exact, primary_exact = self._score(session, submission, current, scores)

        facts = Facts(
            status=source,
            author_id=submission.author_id,
            assignee_id=current.assignment.reviewer_id if current.assignment else None,
            primary_reviewer_id=current.primary.reviewer_id if current.primary else None,
            primary_score=primary_exact,
            primary_passed=current.primary.passed if current.primary else None,
            qa_reviewer_id=current.qa.reviewer_id if current.qa else None,
        )
        command = Command(
            action,
            actor,
            assignee=assignee,
            score=exact,
            passed=result.passed if result else None,
        )
        target = decide(facts, command, self._policy)

        now = self._now()
        data: dict[str, Any] = {"round": submission.round}
        if context:
            data["context"] = dict(context)
        review = None
        if stage is not None and result is not None:
            review = self._write_review(
                session,
                submission,
                actor=actor,
                stage=stage,
                result=result,
                comment=comment,
                now=now,
            )
            data.update(
                review_id=review.id,
                stage=stage.value,
                rubric_version_id=submission.rubric_version_id,
                content_hash=result.content_hash,
                scores=review.scores,
                score=result.score,
                passed=result.passed,
            )
        self._track_assignment(
            session,
            submission,
            action=action,
            assignee=assignee,
            current=current,
            now=now,
            data=data,
        )
        if target is Status.FINALIZED:
            final = review if action is Action.ADJUDICATE else current.primary
            if final is None:  # pragma: no cover - the table makes this unreachable
                msg = f"submission {submission.id} reached finalized without a review"
                raise StoredDataError(msg)
            submission.final_score = final.score
            submission.final_passed = final.passed
            submission.final_stage = final.stage
            data.update(
                final_review_id=final.id, final_score=final.score, final_passed=final.passed
            )

        submission.status = target.value
        submission.updated_at = now
        session.flush()  # bumps submission.version; raises StaleDataError on a race
        data["version"] = submission.version
        event = append_event(
            session,
            self._clock,
            actor_id=actor.id,
            action=action.value,
            entity="submission",
            entity_id=submission.id,
            from_status=source.value,
            to_status=target.value,
            data=data,
        )
        return TransitionResult(
            submission_id=submission.id,
            action=action,
            from_status=source,
            to_status=target,
            version=submission.version,
            audit_seq=event.seq,
            review_id=review.id if review else None,
            score=result.score if result else None,
            passed=result.passed if result else None,
        )

    def _score(
        self,
        session: Session,
        submission: Submission,
        current: _Round,
        scores: Mapping[str, object] | None,
    ) -> tuple[ScoreResult | None, Fraction | None, Fraction | None]:
        """Score a reviewing action; return the result, its exact total and the primary's."""
        if scores is None:
            return None, None, None
        rubric = self._pinned_rubric(session, submission.rubric_version_id)
        primary_exact = None
        if current.primary is not None:
            # Re-scored exactly from the stored map: the stored float is rounded.
            primary_rubric = self._pinned_rubric(session, current.primary.rubric_version_id)
            primary_exact = exact_score(primary_rubric, current.primary.scores)
        return score_review(rubric, scores), exact_score(rubric, scores), primary_exact

    @staticmethod
    def _write_review(
        session: Session,
        submission: Submission,
        *,
        actor: Actor,
        stage: Stage,
        result: ScoreResult,
        comment: str | None,
        now: datetime,
    ) -> Review:
        review = Review(
            submission_id=submission.id,
            reviewer_id=actor.id,
            stage=stage.value,
            round=submission.round,
            rubric_version_id=submission.rubric_version_id,
            scores={c.criterion_id: c.score for c in result.criteria},
            score=result.score,
            passed=result.passed,
            comment=comment,
            created_at=now,
        )
        session.add(review)
        session.flush()
        return review

    def _track_assignment(
        self,
        session: Session,
        submission: Submission,
        *,
        action: Action,
        assignee: Actor | None,
        current: _Round,
        now: datetime,
        data: dict[str, Any],
    ) -> None:
        """Open, close or release the primary assignment, and start a new round on resubmit."""
        open_assignment = current.assignment
        if action is Action.ASSIGN and assignee is not None:
            rubric_id = self._rubric_version(session, submission.rubric_version_id).rubric_id
            session.add(
                Assignment(
                    submission_id=submission.id,
                    reviewer_id=assignee.id,
                    stage=Stage.PRIMARY.value,
                    round=submission.round,
                    assigned_at=now,
                    due_at=self._sla.due_at(now, rubric_id),
                )
            )
            data["assignee_id"] = assignee.id
        elif open_assignment is not None and open_assignment.completed_at is None:
            if action in (Action.RELEASE, Action.RETURN_TO_AUTHOR):
                open_assignment.released_at = now
            elif action is Action.SUBMIT_PRIMARY:
                open_assignment.completed_at = now
        elif action is Action.RESUBMIT:
            submission.round += 1
            data["round"] = submission.round

    # -- reads ------------------------------------------------------------------

    def get_submission(self, submission_id: int) -> Submission:
        with session_scope(self._sessions) as session:
            submission = session.get(Submission, submission_id)
            if submission is None:
                raise NotFoundError(f"no submission {submission_id}")
            return submission

    def reviews(self, submission_id: int) -> list[Review]:
        with session_scope(self._sessions) as session:
            return list(
                session.scalars(
                    select(Review).where(Review.submission_id == submission_id).order_by(Review.id)
                )
            )

    def count_events(self) -> int:
        with session_scope(self._sessions) as session:
            return session.scalar(select(func.count()).select_from(AuditEvent)) or 0

    # -- helpers ----------------------------------------------------------------

    def _now(self) -> datetime:
        return self._clock.now()

    @staticmethod
    def _actor(session: Session, user_id: int) -> Actor:
        user = session.get(User, user_id)
        if user is None:
            raise NotFoundError(f"no user {user_id}")
        return Actor(user.id, Role(user.role))

    @staticmethod
    def _rubric_version(session: Session, rubric_version_id: int) -> RubricVersionRow:
        row = session.get(RubricVersionRow, rubric_version_id)
        if row is None:
            raise NotFoundError(f"no rubric version {rubric_version_id}")
        return row

    def _pinned_rubric(self, session: Session, rubric_version_id: int) -> Rubric:
        row = self._rubric_version(session, rubric_version_id)
        rubric = Rubric.model_validate_json(row.body)
        if content_hash(rubric) != row.content_hash:
            msg = f"rubric version {row.id} body does not match its stored hash"
            raise StoredDataError(msg)
        return rubric

    @staticmethod
    def _round(session: Session, submission: Submission) -> _Round:
        assignment = session.scalars(
            select(Assignment)
            .where(
                Assignment.submission_id == submission.id,
                Assignment.round == submission.round,
                Assignment.stage == Stage.PRIMARY.value,
                Assignment.released_at.is_(None),
            )
            .order_by(Assignment.id.desc())
            .limit(1)
        ).first()
        reviews = {
            review.stage: review
            for review in session.scalars(
                select(Review).where(
                    Review.submission_id == submission.id, Review.round == submission.round
                )
            )
        }
        return _Round(
            assignment=assignment,
            primary=reviews.get(Stage.PRIMARY.value),
            qa=reviews.get(Stage.QA.value),
        )
