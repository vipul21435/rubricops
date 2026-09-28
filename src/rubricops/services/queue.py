"""The review queue on the database: assignment, SLA sweeps and QA sampling.

:class:`QueueService` reads the facts the pure queue code needs (reviewers and
their open load, open assignments, the primary review and the item's other scores),
asks the domain policy, and then acts through :class:`PipelineService`, so every
decision it takes is a normal pipeline transition with one audit event. The audit
event's ``context`` records why:

- ``assign_next`` stores the policy name and the round-robin cursor it used;
- ``sample_for_qa`` stores the sampling reasons, the draw and the rate, then sends
  the item to QA (``send_to_qa``) or finalizes it (``finalize``).

The round-robin cursor needs no table of its own: it is the reviewer of the most
recent primary assignment, which the ``assignments`` table already records.
Primary assignments only; a QA audit is taken by any eligible reviewer through the
pipeline's own guards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from rubricops.db.engine import session_scope
from rubricops.db.models import Assignment, Review, Submission, User
from rubricops.db.models import RubricVersion as RubricVersionRow
from rubricops.domain.clock import Clock
from rubricops.domain.pipeline import Action, Role, Stage, Status
from rubricops.domain.queue import (
    AssignmentPolicy,
    NoEligibleReviewer,
    QueueItem,
    Reviewer,
    RoundRobin,
)
from rubricops.domain.rubric import Rubric
from rubricops.domain.sampling import QaSampler, SampleCandidate, SampleDecision
from rubricops.domain.sla import OpenAssignment, Overdue, SlaPolicy, find_overdue
from rubricops.services.pipeline import NotFoundError, PipelineService


class NotReviewedError(ValueError):
    """QA sampling was asked for an item that is not waiting for the QA decision."""


@dataclass(frozen=True, slots=True)
class AssignOutcome:
    submission_id: int
    reviewer_id: int
    policy: str
    due_at: datetime | None


def _never_flagged(_reviewer_id: int) -> bool:
    return False


class QueueService:
    """Assign, sweep and sample through the pipeline service."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        clock: Clock,
        pipeline: PipelineService,
        *,
        sla: SlaPolicy,
        sampler: QaSampler,
        is_flagged: Callable[[int], bool] = _never_flagged,
    ) -> None:
        self._sessions = sessions
        self._clock = clock
        self._pipeline = pipeline
        self._sla = sla
        self._sampler = sampler
        self._is_flagged = is_flagged  # calibration hook, filled by slice 5

    # -- assignment -------------------------------------------------------------

    def assign_next(self, policy: AssignmentPolicy, *, actor_id: int) -> AssignOutcome | None:
        """Assign the oldest queued submission somebody can take; None when the queue is empty.

        Items are tried oldest first, and an item the policy refuses is skipped, so one
        item nobody can take does not block the rest of the queue. When the policy
        refuses every queued item, the oldest item's
        :class:`~rubricops.domain.queue.NoEligibleReviewer` is raised and all items
        stay queued. A :class:`RoundRobin` cursor only moves once the assignment is
        written: if the pipeline refuses it, the cursor is put back.
        """
        with session_scope(self._sessions) as session:
            queued = session.scalars(
                select(Submission)
                .where(Submission.status == Status.QUEUED.value)
                .order_by(Submission.created_at, Submission.id)
            ).all()
            if not queued:
                return None
            reviewers = self._reviewers(session)
            if isinstance(policy, RoundRobin) and policy.cursor is None:
                policy.cursor = session.scalars(
                    select(Assignment.reviewer_id)
                    .where(Assignment.stage == Stage.PRIMARY.value)
                    .order_by(Assignment.id.desc())
                    .limit(1)
                ).first()
            cursor = policy.cursor if isinstance(policy, RoundRobin) else None
            refusals: list[NoEligibleReviewer] = []
            for submission in queued:
                item = QueueItem(
                    submission.id,
                    submission.author_id,
                    Stage.PRIMARY,
                    frozenset(submission.skill_tags),
                )
                try:
                    chosen = policy.choose(item, reviewers)
                except NoEligibleReviewer as exc:
                    refusals.append(exc)
                    continue
                version = submission.version
                break
            else:
                raise refusals[0]  # every queued item was refused: report the oldest
        context: dict[str, object] = {"policy": policy.name}
        if cursor is not None:
            context["cursor"] = cursor
        try:
            self._pipeline.apply(
                item.submission_id,
                Action.ASSIGN,
                actor_id=actor_id,
                expected_version=version,
                assignee_id=chosen.id,
                context=context,
            )
        except Exception:
            if isinstance(policy, RoundRobin):
                policy.cursor = cursor  # nothing was assigned, so a retry picks the same one
            raise
        with session_scope(self._sessions) as session:
            due_at = session.scalars(
                select(Assignment.due_at)
                .where(Assignment.submission_id == item.submission_id)
                .order_by(Assignment.id.desc())
                .limit(1)
            ).one()
        return AssignOutcome(item.submission_id, chosen.id, policy.name, due_at)

    @staticmethod
    def _reviewers(session: Session) -> list[Reviewer]:
        open_counts: dict[int, int] = dict(
            session.execute(
                select(Assignment.reviewer_id, func.count())
                .where(Assignment.completed_at.is_(None), Assignment.released_at.is_(None))
                .group_by(Assignment.reviewer_id)
            ).all()
        )
        users = session.scalars(
            select(User).where(User.role == Role.REVIEWER.value).order_by(User.id)
        )
        return [
            Reviewer(u.id, u.handle, frozenset(u.skill_tags), open_counts.get(u.id, 0))
            for u in users
        ]

    # -- SLA --------------------------------------------------------------------

    def sweep_overdue(self) -> list[Overdue]:
        """Open assignments past their due time at the injected clock, most late first."""
        with session_scope(self._sessions) as session:
            rows = session.execute(
                select(Assignment, RubricVersionRow.rubric_id)
                .join(Submission, Submission.id == Assignment.submission_id)
                .join(RubricVersionRow, RubricVersionRow.id == Submission.rubric_version_id)
                .where(Assignment.completed_at.is_(None), Assignment.released_at.is_(None))
            ).all()
            assignments = [
                OpenAssignment(
                    submission_id=a.submission_id,
                    reviewer_id=a.reviewer_id,
                    assigned_at=a.assigned_at,
                    due_at=a.due_at or self._sla.due_at(a.assigned_at, rubric_id),
                    stage=Stage(a.stage),
                )
                for a, rubric_id in rows
            ]
        return find_overdue(assignments, self._clock)

    # -- QA sampling --------------------------------------------------------------

    def sample_for_qa(self, submission_id: int, *, actor_id: int) -> SampleDecision:
        """Decide finalize-or-QA for a reviewed item and apply it, recording the reasons."""
        with session_scope(self._sessions) as session:
            submission = session.get(Submission, submission_id)
            if submission is None:
                raise NotFoundError(f"no submission {submission_id}")
            if submission.status != Status.REVIEWED.value:
                msg = f"submission {submission_id} is {submission.status}, not reviewed"
                raise NotReviewedError(msg)
            reviews = list(
                session.scalars(select(Review).where(Review.submission_id == submission_id))
            )
            primary = next(
                r for r in reviews if r.round == submission.round and r.stage == Stage.PRIMARY.value
            )
            completed = session.scalar(
                select(func.count())
                .select_from(Review)
                .where(
                    Review.reviewer_id == primary.reviewer_id,
                    Review.stage == Stage.PRIMARY.value,
                    Review.id < primary.id,  # only reviews written before this one
                )
            )
            body = session.get_one(RubricVersionRow, submission.rubric_version_id).body
            candidate = SampleCandidate(
                submission_id=submission_id,
                reviewer_id=primary.reviewer_id,
                score=primary.score,
                pass_threshold=Rubric.model_validate_json(body).pass_threshold,
                reviewer_completed_reviews=completed or 0,
                round=submission.round,
                item_scores=tuple(r.score for r in reviews if r.id != primary.id),
                reviewer_flagged=self._is_flagged(primary.reviewer_id),
            )
            version = submission.version
        decision = self._sampler.decide(candidate)
        self._pipeline.apply(
            submission_id,
            Action.SEND_TO_QA if decision.sampled else Action.FINALIZE,
            actor_id=actor_id,
            expected_version=version,
            context={
                "qa_sampling": {
                    "sampled": decision.sampled,
                    "reasons": [r.value for r in decision.reasons],
                    "details": list(decision.details),
                    "draw": decision.draw,
                    "rate": decision.rate,
                    "seed": self._sampler.seed,
                    "reviewer_completed_reviews": candidate.reviewer_completed_reviews,
                }
            },
        )
        return decision
