"""The review pipeline as an explicit, pure state machine.

A submission moves through a primary review, an optional QA audit and, when the
audit disagrees, adjudication by a lead::

    queued -> assigned -> in_review -> reviewed -+-> finalized  (not sampled for QA)
                                                 |
                                                 +-> qa_pending -+-> qa_passed -> finalized
                                                                 |
                                                                 +-> qa_failed -> in_adjudication
                                                                                   -> finalized
    side exits:
      assigned -> queued                        release (the assignee or a lead)
      in_review -> returned_to_author           return_to_author (the assignee)
      in_adjudication -> returned_to_author     return_to_author (a lead)
      returned_to_author -> queued              resubmit (the author), which starts a new round

:data:`TRANSITIONS` maps every legal ``(status, action)`` pair to a :class:`Rule`:
the target status, the roles allowed to take the action, and named guards. Anything
not in the table raises :class:`IllegalTransition`; a legal action by the wrong
person raises :class:`Forbidden` naming the rule that failed. The guards encode the
independence rules of expert review:

- nobody reviews, audits or adjudicates their own submission;
- only the assigned reviewer starts, submits or returns a primary review;
- the QA auditor is not the primary reviewer;
- the adjudicator is a lead and is neither the primary reviewer nor the QA auditor.

``submit_qa`` is the one branching action. The audit passes when the QA score is
within ``PipelinePolicy.qa_tolerance`` of the primary score *and* both reviews reach
the same pass/fail verdict; otherwise the item needs adjudication. Nothing here
touches a database or a clock, so the whole table is tested exhaustively.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from rubricops.domain.rubric import exact_decimal


class Role(StrEnum):
    AUTHOR = "author"
    REVIEWER = "reviewer"
    LEAD = "lead"


class Status(StrEnum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    REVIEWED = "reviewed"
    QA_PENDING = "qa_pending"
    QA_PASSED = "qa_passed"
    QA_FAILED = "qa_failed"
    IN_ADJUDICATION = "in_adjudication"
    FINALIZED = "finalized"
    RETURNED_TO_AUTHOR = "returned_to_author"


class Action(StrEnum):
    ASSIGN = "assign"
    RELEASE = "release"
    START = "start"
    SUBMIT_PRIMARY = "submit_primary"
    RETURN_TO_AUTHOR = "return_to_author"
    SEND_TO_QA = "send_to_qa"
    FINALIZE = "finalize"
    SUBMIT_QA = "submit_qa"
    ESCALATE = "escalate"
    ADJUDICATE = "adjudicate"
    RESUBMIT = "resubmit"


class Stage(StrEnum):
    """The kind of review a reviewing action records."""

    PRIMARY = "primary"
    QA = "qa"
    ADJUDICATION = "adjudication"


#: Actions that record a scored review, and the stage each one records.
REVIEW_STAGE: Mapping[Action, Stage] = MappingProxyType(
    {
        Action.SUBMIT_PRIMARY: Stage.PRIMARY,
        Action.SUBMIT_QA: Stage.QA,
        Action.ADJUDICATE: Stage.ADJUDICATION,
    }
)

TERMINAL_STATUSES = frozenset({Status.FINALIZED})


class PipelineError(Exception):
    """Base class for state-machine errors."""


class IllegalTransition(PipelineError):  # noqa: N818 - the name reads as the domain event
    """The action is not defined for the submission's current status."""

    def __init__(self, status: Status, action: Action) -> None:
        self.status = status
        self.action = action
        allowed = ", ".join(sorted(a.value for a in allowed_actions(status))) or "none"
        super().__init__(
            f"cannot {action.value} a submission that is {status.value} (allowed: {allowed})"
        )


class Forbidden(PipelineError):  # noqa: N818 - mirrors the HTTP status it maps to
    """The action is legal here, but not for this actor."""

    def __init__(self, action: Action, rule: str, detail: str) -> None:
        self.action = action
        self.rule = rule
        super().__init__(f"{action.value} forbidden by {rule}: {detail}")


@dataclass(frozen=True, slots=True)
class Actor:
    id: int
    role: Role


@dataclass(frozen=True, slots=True)
class Facts:
    """What the state machine needs to know about a submission in its current round."""

    status: Status
    author_id: int
    assignee_id: int | None = None
    primary_reviewer_id: int | None = None
    primary_score: float | None = None
    primary_passed: bool | None = None
    qa_reviewer_id: int | None = None


@dataclass(frozen=True, slots=True)
class Command:
    """An actor asking to take an action.

    ``assignee`` is required for ``assign``. ``score`` and ``passed`` are the result
    of the review being recorded and are required for ``submit_qa``, whose target
    depends on them.
    """

    action: Action
    actor: Actor
    assignee: Actor | None = None
    score: float | None = None
    passed: bool | None = None


class PipelinePolicy(BaseModel):
    """Tunable pipeline rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Largest |QA score - primary score| (both normalised to [0, 1]) that still passes QA.
    qa_tolerance: float = Field(default=0.1, ge=0.0, le=1.0)


#: A guard returns ``None`` when satisfied, or a sentence saying why not.
Guard = Callable[[Facts, Command], str | None]


def _assignee_is_eligible(facts: Facts, command: Command) -> str | None:
    assignee = command.assignee
    if assignee is None:
        return "assign needs an assignee"
    if assignee.role not in (Role.REVIEWER, Role.LEAD):
        return f"user {assignee.id} is a {assignee.role.value}, not a reviewer or lead"
    if assignee.id == facts.author_id:
        return f"user {assignee.id} is the author of this submission"
    return None


def _actor_is_assignee(facts: Facts, command: Command) -> str | None:
    if command.actor.id != facts.assignee_id:
        return f"user {command.actor.id} is not the assigned reviewer ({facts.assignee_id})"
    return None


def _lead_or_assignee(facts: Facts, command: Command) -> str | None:
    if command.actor.role is Role.LEAD:
        return None
    return _actor_is_assignee(facts, command)


def _actor_is_not_author(facts: Facts, command: Command) -> str | None:
    if command.actor.id == facts.author_id:
        return f"user {command.actor.id} is the author of this submission"
    return None


def _auditor_is_not_primary(facts: Facts, command: Command) -> str | None:
    if command.actor.id == facts.primary_reviewer_id:
        return f"user {command.actor.id} wrote the primary review and cannot audit it"
    return None


def _qa_result_given(_facts: Facts, command: Command) -> str | None:
    if command.score is None or command.passed is None:
        return "submit_qa needs the QA review's score and verdict"
    return None


def _adjudicator_is_independent(facts: Facts, command: Command) -> str | None:
    if command.actor.id in (facts.primary_reviewer_id, facts.qa_reviewer_id):
        return f"user {command.actor.id} already reviewed this submission"
    return None


def _actor_is_author(facts: Facts, command: Command) -> str | None:
    if command.actor.id != facts.author_id:
        return f"only the author ({facts.author_id}) can resubmit"
    return None


_REVIEWERS = frozenset({Role.REVIEWER, Role.LEAD})
_LEADS = frozenset({Role.LEAD})
_AUTHORS = frozenset({Role.AUTHOR})


@dataclass(frozen=True, slots=True)
class Rule:
    """One row of the transition table.

    ``targets`` has one status, except for ``submit_qa`` whose two targets are the
    passed and failed outcomes, in that order.
    """

    targets: tuple[Status, ...]
    roles: frozenset[Role]
    guards: tuple[Guard, ...] = ()


TRANSITIONS: Mapping[tuple[Status, Action], Rule] = MappingProxyType(
    {
        (Status.QUEUED, Action.ASSIGN): Rule((Status.ASSIGNED,), _LEADS, (_assignee_is_eligible,)),
        (Status.ASSIGNED, Action.RELEASE): Rule((Status.QUEUED,), _REVIEWERS, (_lead_or_assignee,)),
        (Status.ASSIGNED, Action.START): Rule(
            (Status.IN_REVIEW,), _REVIEWERS, (_actor_is_assignee,)
        ),
        (Status.IN_REVIEW, Action.SUBMIT_PRIMARY): Rule(
            (Status.REVIEWED,), _REVIEWERS, (_actor_is_assignee,)
        ),
        (Status.IN_REVIEW, Action.RETURN_TO_AUTHOR): Rule(
            (Status.RETURNED_TO_AUTHOR,), _REVIEWERS, (_actor_is_assignee,)
        ),
        (Status.REVIEWED, Action.SEND_TO_QA): Rule((Status.QA_PENDING,), _LEADS),
        (Status.REVIEWED, Action.FINALIZE): Rule((Status.FINALIZED,), _LEADS),
        (Status.QA_PENDING, Action.SUBMIT_QA): Rule(
            (Status.QA_PASSED, Status.QA_FAILED),
            _REVIEWERS,
            (_actor_is_not_author, _auditor_is_not_primary, _qa_result_given),
        ),
        (Status.QA_PASSED, Action.FINALIZE): Rule((Status.FINALIZED,), _LEADS),
        (Status.QA_FAILED, Action.ESCALATE): Rule((Status.IN_ADJUDICATION,), _LEADS),
        (Status.IN_ADJUDICATION, Action.ADJUDICATE): Rule(
            (Status.FINALIZED,), _LEADS, (_actor_is_not_author, _adjudicator_is_independent)
        ),
        (Status.IN_ADJUDICATION, Action.RETURN_TO_AUTHOR): Rule(
            (Status.RETURNED_TO_AUTHOR,), _LEADS, (_actor_is_not_author,)
        ),
        (Status.RETURNED_TO_AUTHOR, Action.RESUBMIT): Rule(
            (Status.QUEUED,), _AUTHORS, (_actor_is_author,)
        ),
    }
)


def allowed_actions(status: Status) -> frozenset[Action]:
    """Every action defined for ``status`` (ignoring who is asking)."""
    return frozenset(action for (source, action) in TRANSITIONS if source is status)


def qa_agrees(
    primary_score: float,
    primary_passed: bool,
    qa_score: float,
    qa_passed: bool,
    policy: PipelinePolicy,
) -> bool:
    """True when a QA review confirms the primary one: same verdict, score within tolerance."""
    if primary_passed != qa_passed:
        return False
    # Compared as the decimals the scores print as, so 0.8 vs 0.7 is exactly 0.1 apart.
    distance = abs(exact_decimal(qa_score) - exact_decimal(primary_score))
    return distance <= exact_decimal(policy.qa_tolerance)


def decide(facts: Facts, command: Command, policy: PipelinePolicy | None = None) -> Status:
    """Return the status ``command`` moves the submission to, or raise.

    Raises :class:`IllegalTransition` when the action is not defined for the current
    status and :class:`Forbidden` when the actor's role or a guard rules it out.
    """
    rule = TRANSITIONS.get((facts.status, command.action))
    if rule is None:
        raise IllegalTransition(facts.status, command.action)
    if command.actor.role not in rule.roles:
        roles = "/".join(sorted(role.value for role in rule.roles))
        raise Forbidden(
            command.action,
            "role",
            f"needs {roles}, user {command.actor.id} is {command.actor.role.value}",
        )
    for guard in rule.guards:
        problem = guard(facts, command)
        if problem is not None:
            raise Forbidden(command.action, guard.__name__.strip("_"), problem)

    if command.action is not Action.SUBMIT_QA:
        return rule.targets[0]
    # The guards above ensured score and verdict are present.
    assert command.score is not None  # noqa: S101
    assert command.passed is not None  # noqa: S101
    if facts.primary_score is None or facts.primary_passed is None:
        msg = "a submission awaiting QA must carry its primary score and verdict"
        raise PipelineError(msg)
    passed, failed = rule.targets
    agrees = qa_agrees(
        facts.primary_score,
        facts.primary_passed,
        command.score,
        command.passed,
        policy or PipelinePolicy(),
    )
    return passed if agrees else failed
