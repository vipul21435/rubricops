"""Queue scenario files: a snapshot of reviewers, queued items, open assignments and
primary reviews, read by ``rubricops queue assign|overdue|sample``.

A scenario is plain data (YAML or JSON) so the queue policies can be run and
compared without a database. Timestamps must carry a timezone. Unknown keys are
errors, as in rubric files, and every reference (an assignment's reviewer, a
review's reviewer) must name a listed reviewer.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Annotated, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from rubricops.domain.pipeline import Stage
from rubricops.domain.queue import QueueItem, Reviewer
from rubricops.domain.sampling import SampleCandidate
from rubricops.domain.sla import OpenAssignment, SlaPolicy
from rubricops.loaders import DocumentError, format_validation_error, load_document


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReviewerSpec(_Strict):
    id: int
    handle: str = Field(min_length=1)
    skills: list[str] = Field(default_factory=list)
    open: int = Field(default=0, ge=0)
    capacity: int | None = Field(default=None, ge=1)
    completed_reviews: int = Field(default=0, ge=0)
    flagged: bool = False


class QueueItemSpec(_Strict):
    submission: int
    author: int
    stage: Stage = Stage.PRIMARY
    skills: list[str] = Field(default_factory=list)
    primary_reviewer: int | None = None


class AssignmentSpec(_Strict):
    submission: int
    reviewer: int
    rubric: str | None = None
    stage: Stage = Stage.PRIMARY
    assigned_at: AwareDatetime
    due_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    released_at: AwareDatetime | None = None


class ReviewSpec(_Strict):
    submission: int
    reviewer: int
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    round: int = Field(default=1, ge=1)
    item_scores: list[Annotated[float, Field(ge=0, le=1)]] = Field(default_factory=list)


class SlaSpec(_Strict):
    default_hours: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    rubrics: dict[str, Annotated[float, Field(gt=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )


class Scenario(_Strict):
    now: AwareDatetime | None = None
    sla: SlaSpec = Field(default_factory=SlaSpec)
    reviewers: list[ReviewerSpec] = Field(default_factory=list)
    queue: list[QueueItemSpec] = Field(default_factory=list)
    assignments: list[AssignmentSpec] = Field(default_factory=list)
    reviews: list[ReviewSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        ids = [r.id for r in self.reviewers]
        if len(set(ids)) != len(ids):
            msg = "reviewer ids must be unique"
            raise ValueError(msg)
        known = set(ids)
        for kind, refs in (
            ("assignments", [a.reviewer for a in self.assignments]),
            ("reviews", [r.reviewer for r in self.reviews]),
        ):
            unknown = sorted(set(refs) - known)
            if unknown:
                msg = f"{kind} name reviewers that are not listed: {unknown}"
                raise ValueError(msg)
        self.queue_items()  # the domain value objects check stage-specific fields
        try:  # the SLA hours must also make usable durations (not 0 once rounded, no overflow)
            self.sla_policy(1.0)
        except OverflowError as exc:
            msg = f"sla hours are too large: {exc}"
            raise ValueError(msg) from exc
        rounds = [(r.submission, r.round) for r in self.reviews]
        repeated = sorted({key for key in rounds if rounds.count(key) > 1})
        if repeated:
            msg = f"reviews repeat a (submission, round) pair: {repeated}"
            raise ValueError(msg)
        for a in self.assignments:
            if a.due_at is not None and a.due_at < a.assigned_at:
                msg = f"assignment for submission {a.submission}: due_at is before assigned_at"
                raise ValueError(msg)
        return self

    # -- domain views -----------------------------------------------------------

    def reviewer_pool(self) -> list[Reviewer]:
        return [
            Reviewer(r.id, r.handle, frozenset(r.skills), r.open, r.capacity)
            for r in self.reviewers
        ]

    def handles(self) -> dict[int, str]:
        return {r.id: r.handle for r in self.reviewers}

    def queue_items(self) -> list[QueueItem]:
        return [
            QueueItem(q.submission, q.author, q.stage, frozenset(q.skills), q.primary_reviewer)
            for q in self.queue
        ]

    def sla_policy(
        self, fallback_hours: float, *, override_hours: float | None = None
    ) -> SlaPolicy:
        """Default SLA: ``override_hours``, else the file's ``default_hours``, else the fallback."""
        default = override_hours or self.sla.default_hours or fallback_hours
        return SlaPolicy.from_hours(default, self.sla.rubrics)

    def open_assignments(self, policy: SlaPolicy) -> list[OpenAssignment]:
        return [
            OpenAssignment(
                submission_id=a.submission,
                reviewer_id=a.reviewer,
                assigned_at=a.assigned_at,
                due_at=a.due_at or policy.due_at(a.assigned_at, a.rubric),
                stage=a.stage,
                completed_at=a.completed_at,
                released_at=a.released_at,
            )
            for a in self.assignments
        ]

    def sample_candidates(self, flagged_handles: Collection[str] = ()) -> list[SampleCandidate]:
        """One candidate per review; ``flagged_handles`` adds calibration flags."""
        by_id = {r.id: r for r in self.reviewers}
        flagged = set(flagged_handles)
        return [
            SampleCandidate(
                submission_id=r.submission,
                reviewer_id=r.reviewer,
                score=r.score,
                pass_threshold=r.threshold,
                reviewer_completed_reviews=by_id[r.reviewer].completed_reviews,
                round=r.round,
                item_scores=tuple(r.item_scores),
                reviewer_flagged=by_id[r.reviewer].flagged or by_id[r.reviewer].handle in flagged,
            )
            for r in self.reviews
        ]


def load_scenario(path: Path) -> Scenario:
    """Load and validate a queue scenario; problems raise :class:`DocumentError`."""
    data = load_document(path)
    if not isinstance(data, dict):
        raise DocumentError(path, [f"expected a scenario mapping, got {type(data).__name__}"])
    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise DocumentError(path, format_validation_error(exc)) from exc
