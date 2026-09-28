"""Review-queue assignment policies.

An :class:`AssignmentPolicy` picks one reviewer for one queued item. Every policy
applies the same independence and capacity rules first (:func:`exclusion`):

- nobody reviews their own submission (``author``);
- the QA auditor is never the item's primary reviewer (``primary_reviewer``);
- a reviewer with a capacity cap and that many open assignments takes no more
  (``at_capacity``).

and then chooses among the reviewers left:

- :class:`RoundRobin` walks reviewer ids in ascending order from a cursor that is
  the id of the last reviewer it assigned. The cursor is a plain id, not a list
  index, so it survives reviewers joining or leaving and can be persisted between
  runs (``policy.cursor``).
- :class:`SkillTagMatch` keeps reviewers whose skill tags are a superset of the
  item's tags, then takes the lowest open load, then the lowest id.
- :class:`LoadBalanced` takes the reviewer with the fewest open assignments, then
  the lowest id.

When nobody fits, the policy raises :class:`NoEligibleReviewer`, which carries the
reason each candidate was excluded. :func:`assign_batch` runs a policy over a list
of items and counts each assignment towards the reviewer's open load as it goes, so
capacity caps and load balancing hold across the whole batch. Nothing here touches a
database or a clock.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Protocol

from rubricops.domain.pipeline import Stage


class Exclusion(StrEnum):
    """Why a reviewer cannot take an item."""

    AUTHOR = "author"
    PRIMARY_REVIEWER = "primary_reviewer"
    AT_CAPACITY = "at_capacity"
    MISSING_SKILLS = "missing_skills"


@dataclass(frozen=True, slots=True)
class Reviewer:
    """A candidate reviewer and their current open load."""

    id: int
    handle: str
    skill_tags: frozenset[str] = frozenset()
    open_assignments: int = 0
    capacity: int | None = None

    def __post_init__(self) -> None:
        if self.open_assignments < 0:
            msg = f"reviewer {self.id} has a negative open load ({self.open_assignments})"
            raise ValueError(msg)
        if self.capacity is not None and self.capacity < 1:
            msg = f"reviewer {self.id} capacity must be at least 1, got {self.capacity}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class QueueItem:
    """A submission waiting for a primary review or a QA audit."""

    submission_id: int
    author_id: int
    stage: Stage = Stage.PRIMARY
    skill_tags: frozenset[str] = frozenset()
    primary_reviewer_id: int | None = None

    def __post_init__(self) -> None:
        if self.stage is Stage.ADJUDICATION:
            msg = "adjudication is done by a lead, not assigned from the review queue"
            raise ValueError(msg)
        if self.stage is Stage.QA and self.primary_reviewer_id is None:
            msg = f"QA item {self.submission_id} needs its primary_reviewer_id"
            raise ValueError(msg)


class NoEligibleReviewer(LookupError):  # noqa: N818 - named for what the caller sees
    """Nobody can take the item; ``excluded`` maps each candidate id to the reason."""

    def __init__(self, item: QueueItem, policy: str, excluded: Mapping[int, Exclusion]) -> None:
        self.item = item
        self.policy = policy
        self.excluded: Mapping[int, Exclusion] = MappingProxyType(dict(excluded))
        detail = (
            ", ".join(f"{rid}:{why.value}" for rid, why in sorted(self.excluded.items()))
            or "no candidates"
        )
        super().__init__(
            f"no eligible {item.stage.value} reviewer for submission "
            f"{item.submission_id} under {policy} ({detail})"
        )


def exclusion(item: QueueItem, reviewer: Reviewer) -> Exclusion | None:
    """The first rule that stops ``reviewer`` from taking ``item``, or None."""
    if reviewer.id == item.author_id:
        return Exclusion.AUTHOR
    if item.stage is Stage.QA and reviewer.id == item.primary_reviewer_id:
        return Exclusion.PRIMARY_REVIEWER
    if reviewer.capacity is not None and reviewer.open_assignments >= reviewer.capacity:
        return Exclusion.AT_CAPACITY
    return None


def _split(
    item: QueueItem, candidates: Iterable[Reviewer]
) -> tuple[list[Reviewer], dict[int, Exclusion]]:
    eligible: list[Reviewer] = []
    excluded: dict[int, Exclusion] = {}
    seen: set[int] = set()
    for reviewer in candidates:
        if reviewer.id in seen:
            msg = f"reviewer {reviewer.id} is listed twice"
            raise ValueError(msg)
        seen.add(reviewer.id)
        why = exclusion(item, reviewer)
        if why is None:
            eligible.append(reviewer)
        else:
            excluded[reviewer.id] = why
    return eligible, excluded


class AssignmentPolicy(Protocol):
    """Chooses one reviewer for one item, or raises :class:`NoEligibleReviewer`."""

    name: ClassVar[str]

    def choose(self, item: QueueItem, candidates: Sequence[Reviewer]) -> Reviewer: ...


class RoundRobin:
    """Next eligible reviewer id after the cursor, wrapping around."""

    name: ClassVar[str] = "round-robin"

    def __init__(self, cursor: int | None = None) -> None:
        self.cursor = cursor

    def choose(self, item: QueueItem, candidates: Sequence[Reviewer]) -> Reviewer:
        eligible, excluded = _split(item, candidates)
        if not eligible:
            raise NoEligibleReviewer(item, self.name, excluded)
        ordered = sorted(eligible, key=lambda r: r.id)
        after = [r for r in ordered if self.cursor is None or r.id > self.cursor]
        chosen = after[0] if after else ordered[0]
        self.cursor = chosen.id
        return chosen


class SkillTagMatch:
    """Reviewers whose tags cover the item's tags; lowest open load, then lowest id."""

    name: ClassVar[str] = "skill-match"

    def choose(self, item: QueueItem, candidates: Sequence[Reviewer]) -> Reviewer:
        eligible, excluded = _split(item, candidates)
        skilled = []
        for reviewer in eligible:
            if item.skill_tags <= reviewer.skill_tags:
                skilled.append(reviewer)
            else:
                excluded[reviewer.id] = Exclusion.MISSING_SKILLS
        if not skilled:
            raise NoEligibleReviewer(item, self.name, excluded)
        return min(skilled, key=lambda r: (r.open_assignments, r.id))


class LoadBalanced:
    """The eligible reviewer with the fewest open assignments, then the lowest id."""

    name: ClassVar[str] = "load-balanced"

    def choose(self, item: QueueItem, candidates: Sequence[Reviewer]) -> Reviewer:
        eligible, excluded = _split(item, candidates)
        if not eligible:
            raise NoEligibleReviewer(item, self.name, excluded)
        return min(eligible, key=lambda r: (r.open_assignments, r.id))


POLICY_NAMES = (RoundRobin.name, SkillTagMatch.name, LoadBalanced.name)


def make_policy(name: str, *, cursor: int | None = None) -> AssignmentPolicy:
    """Build a policy by its CLI name; only round-robin takes a cursor."""
    if name == RoundRobin.name:
        return RoundRobin(cursor)
    if cursor is not None:
        msg = f"only {RoundRobin.name} takes a cursor, not {name}"
        raise ValueError(msg)
    if name == SkillTagMatch.name:
        return SkillTagMatch()
    if name == LoadBalanced.name:
        return LoadBalanced()
    msg = f"unknown policy {name!r}; expected one of {', '.join(POLICY_NAMES)}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AssignmentDecision:
    """The outcome for one item: the chosen reviewer, or why nobody was chosen."""

    item: QueueItem
    reviewer: Reviewer | None
    refusal: NoEligibleReviewer | None = None


@dataclass(slots=True)
class BatchResult:
    decisions: list[AssignmentDecision] = field(default_factory=list)
    loads: dict[int, int] = field(default_factory=dict)

    @property
    def assigned(self) -> int:
        return sum(1 for d in self.decisions if d.reviewer is not None)


def assign_batch(
    policy: AssignmentPolicy, items: Iterable[QueueItem], reviewers: Sequence[Reviewer]
) -> BatchResult:
    """Assign ``items`` in order, adding each assignment to the reviewer's open load."""
    pool = {r.id: r for r in reviewers}
    if len(pool) != len(reviewers):
        msg = "reviewer ids must be unique"
        raise ValueError(msg)
    result = BatchResult()
    for item in items:
        try:
            chosen = policy.choose(item, list(pool.values()))
        except NoEligibleReviewer as refusal:
            result.decisions.append(AssignmentDecision(item, None, refusal))
            continue
        pool[chosen.id] = replace(chosen, open_assignments=chosen.open_assignments + 1)
        result.decisions.append(AssignmentDecision(item, chosen))
    result.loads = {rid: r.open_assignments for rid, r in sorted(pool.items())}
    return result
