"""The review state machine, checked over every (status, action) pair."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from rubricops.domain.pipeline import (
    REVIEW_STAGE,
    TRANSITIONS,
    Action,
    Actor,
    Command,
    Facts,
    Forbidden,
    IllegalTransition,
    PipelineError,
    PipelinePolicy,
    Role,
    Stage,
    Status,
    allowed_actions,
    decide,
    qa_agrees,
)

AUTHOR = Actor(1, Role.AUTHOR)
PRIMARY = Actor(2, Role.REVIEWER)
AUDITOR = Actor(3, Role.REVIEWER)
LEAD = Actor(4, Role.LEAD)
OTHER_REVIEWER = Actor(6, Role.REVIEWER)

S = Status
A = Action

# Written out independently of the implementation: the full legal table.
EXPECTED: dict[tuple[Status, Action], tuple[Status, ...]] = {
    (S.QUEUED, A.ASSIGN): (S.ASSIGNED,),
    (S.ASSIGNED, A.RELEASE): (S.QUEUED,),
    (S.ASSIGNED, A.START): (S.IN_REVIEW,),
    (S.IN_REVIEW, A.SUBMIT_PRIMARY): (S.REVIEWED,),
    (S.IN_REVIEW, A.RETURN_TO_AUTHOR): (S.RETURNED_TO_AUTHOR,),
    (S.REVIEWED, A.SEND_TO_QA): (S.QA_PENDING,),
    (S.REVIEWED, A.FINALIZE): (S.FINALIZED,),
    (S.QA_PENDING, A.SUBMIT_QA): (S.QA_PASSED, S.QA_FAILED),
    (S.QA_PASSED, A.FINALIZE): (S.FINALIZED,),
    (S.QA_FAILED, A.ESCALATE): (S.IN_ADJUDICATION,),
    (S.IN_ADJUDICATION, A.ADJUDICATE): (S.FINALIZED,),
    (S.IN_ADJUDICATION, A.RETURN_TO_AUTHOR): (S.RETURNED_TO_AUTHOR,),
    (S.RETURNED_TO_AUTHOR, A.RESUBMIT): (S.QUEUED,),
}

EXPECTED_ROLES: dict[Action, set[Role]] = {
    A.ASSIGN: {Role.LEAD},
    A.RELEASE: {Role.REVIEWER, Role.LEAD},
    A.START: {Role.REVIEWER, Role.LEAD},
    A.SUBMIT_PRIMARY: {Role.REVIEWER, Role.LEAD},
    A.RETURN_TO_AUTHOR: {Role.REVIEWER, Role.LEAD},
    A.SEND_TO_QA: {Role.LEAD},
    A.FINALIZE: {Role.LEAD},
    A.SUBMIT_QA: {Role.REVIEWER, Role.LEAD},
    A.ESCALATE: {Role.LEAD},
    A.ADJUDICATE: {Role.LEAD},
    A.RESUBMIT: {Role.AUTHOR},
}


def expected_roles(status: Status, action: Action) -> set[Role]:
    # Sending an item back from adjudication is a lead's call, unlike from in_review.
    if (status, action) == (S.IN_ADJUDICATION, A.RETURN_TO_AUTHOR):
        return {Role.LEAD}
    return EXPECTED_ROLES[action]


def facts(status: Status, **overrides: object) -> Facts:
    """Facts for a submission by AUTHOR, assigned to and reviewed by PRIMARY."""
    base: dict[str, object] = {
        "status": status,
        "author_id": AUTHOR.id,
        "assignee_id": PRIMARY.id,
        "primary_reviewer_id": PRIMARY.id,
        "primary_score": 0.8,
        "primary_passed": True,
        "qa_reviewer_id": AUDITOR.id,
    }
    base.update(overrides)
    return Facts(**base)  # type: ignore[arg-type]


def permitted_actor(status: Status, action: Action) -> Command:
    """A command every guard for (status, action) accepts."""
    actor = {
        A.ASSIGN: LEAD,
        A.RELEASE: PRIMARY,
        A.START: PRIMARY,
        A.SUBMIT_PRIMARY: PRIMARY,
        A.RETURN_TO_AUTHOR: PRIMARY if status is S.IN_REVIEW else LEAD,
        A.SEND_TO_QA: LEAD,
        A.FINALIZE: LEAD,
        A.SUBMIT_QA: AUDITOR,
        A.ESCALATE: LEAD,
        A.ADJUDICATE: LEAD,
        A.RESUBMIT: AUTHOR,
    }[action]
    return Command(action, actor, assignee=PRIMARY, score=0.8, passed=True)


def test_table_matches_the_documented_design() -> None:
    assert {key: rule.targets for key, rule in TRANSITIONS.items()} == EXPECTED
    for (status, action), rule in TRANSITIONS.items():
        assert set(rule.roles) == expected_roles(status, action), (status, action)


@pytest.mark.parametrize(("status", "action"), list(itertools.product(Status, Action)))
def test_every_status_action_pair(status: Status, action: Action) -> None:
    command = permitted_actor(status, action)
    if (status, action) not in EXPECTED:
        with pytest.raises(IllegalTransition) as info:
            decide(facts(status), command)
        assert info.value.status is status
        assert info.value.action is action
        return
    assert decide(facts(status), command) == EXPECTED[(status, action)][0]


@pytest.mark.parametrize(("key", "role"), list(itertools.product(EXPECTED, Role)))
def test_role_matrix(key: tuple[Status, Action], role: Role) -> None:
    status, action = key
    if role in expected_roles(status, action):
        return
    command = Command(action, Actor(99, role), assignee=PRIMARY, score=0.8, passed=True)
    with pytest.raises(Forbidden) as info:
        decide(facts(status), command)
    assert info.value.rule == "role"


def test_finalized_is_terminal() -> None:
    assert allowed_actions(S.FINALIZED) == frozenset()
    message = str(IllegalTransition(S.FINALIZED, A.ASSIGN))
    assert "allowed: none" in message
    assert "allowed: finalize, send_to_qa" in str(IllegalTransition(S.REVIEWED, A.ASSIGN))


def test_every_status_except_finalized_has_a_way_forward() -> None:
    for status in Status:
        assert bool(allowed_actions(status)) is (status is not S.FINALIZED)


def test_review_stage_map() -> None:
    assert dict(REVIEW_STAGE) == {
        A.SUBMIT_PRIMARY: Stage.PRIMARY,
        A.SUBMIT_QA: Stage.QA,
        A.ADJUDICATE: Stage.ADJUDICATION,
    }


def test_qa_auditor_is_not_the_primary_reviewer() -> None:
    command = Command(A.SUBMIT_QA, PRIMARY, score=0.8, passed=True)
    with pytest.raises(Forbidden, match="wrote the primary review") as info:
        decide(facts(S.QA_PENDING), command)
    assert info.value.rule == "auditor_is_not_primary"


def test_a_lead_cannot_audit_their_own_primary_review() -> None:
    lead_facts = facts(S.QA_PENDING, assignee_id=LEAD.id, primary_reviewer_id=LEAD.id)
    with pytest.raises(Forbidden, match="auditor_is_not_primary"):
        decide(lead_facts, Command(A.SUBMIT_QA, LEAD, score=0.8, passed=True))


def test_nobody_audits_their_own_submission() -> None:
    own = facts(S.QA_PENDING, author_id=AUDITOR.id)
    with pytest.raises(Forbidden, match="actor_is_not_author"):
        decide(own, Command(A.SUBMIT_QA, AUDITOR, score=0.8, passed=True))


def test_qa_needs_the_review_result() -> None:
    with pytest.raises(Forbidden, match="qa_result_given"):
        decide(facts(S.QA_PENDING), Command(A.SUBMIT_QA, AUDITOR))


def test_qa_without_a_primary_result_is_an_invariant_violation() -> None:
    broken = facts(S.QA_PENDING, primary_score=None)
    with pytest.raises(PipelineError, match="primary score"):
        decide(broken, Command(A.SUBMIT_QA, AUDITOR, score=0.8, passed=True))


def test_adjudicator_is_a_lead() -> None:
    with pytest.raises(Forbidden, match="needs lead, user 6 is reviewer"):
        decide(facts(S.IN_ADJUDICATION), Command(A.ADJUDICATE, OTHER_REVIEWER))


@pytest.mark.parametrize("reviewer_field", ["primary_reviewer_id", "qa_reviewer_id"])
def test_adjudicator_did_not_review_the_item(reviewer_field: str) -> None:
    conflicted = facts(S.IN_ADJUDICATION, **{reviewer_field: LEAD.id})
    with pytest.raises(Forbidden, match="already reviewed"):
        decide(conflicted, Command(A.ADJUDICATE, LEAD))


def test_adjudicator_is_not_the_author() -> None:
    with pytest.raises(Forbidden, match="actor_is_not_author"):
        decide(facts(S.IN_ADJUDICATION, author_id=LEAD.id), Command(A.ADJUDICATE, LEAD))
    with pytest.raises(Forbidden, match="actor_is_not_author"):
        decide(facts(S.IN_ADJUDICATION, author_id=LEAD.id), Command(A.RETURN_TO_AUTHOR, LEAD))


@pytest.mark.parametrize("action", [A.START, A.SUBMIT_PRIMARY, A.RETURN_TO_AUTHOR])
def test_only_the_assignee_works_the_primary_review(action: Action) -> None:
    status = S.ASSIGNED if action is A.START else S.IN_REVIEW
    with pytest.raises(Forbidden, match="not the assigned reviewer") as info:
        decide(facts(status), Command(action, OTHER_REVIEWER))
    assert info.value.rule == "actor_is_assignee"
    # Being a lead does not let you submit someone else's review.
    with pytest.raises(Forbidden, match="not the assigned reviewer"):
        decide(facts(status), Command(action, LEAD))


def test_release_by_assignee_or_any_lead() -> None:
    assert decide(facts(S.ASSIGNED), Command(A.RELEASE, PRIMARY)) is S.QUEUED
    assert decide(facts(S.ASSIGNED), Command(A.RELEASE, LEAD)) is S.QUEUED
    with pytest.raises(Forbidden, match="not the assigned reviewer"):
        decide(facts(S.ASSIGNED), Command(A.RELEASE, OTHER_REVIEWER))


@pytest.mark.parametrize(
    ("assignee", "match"),
    [
        (None, "needs an assignee"),
        (Actor(7, Role.AUTHOR), "is a author, not a reviewer or lead"),
        (Actor(AUTHOR.id, Role.REVIEWER), "is the author"),
    ],
)
def test_assignment_eligibility(assignee: Actor | None, match: str) -> None:
    with pytest.raises(Forbidden, match=match):
        decide(facts(S.QUEUED), Command(A.ASSIGN, LEAD, assignee=assignee))


def test_leads_can_be_assigned_primary_reviews() -> None:
    assert decide(facts(S.QUEUED), Command(A.ASSIGN, LEAD, assignee=LEAD)) is S.ASSIGNED


def test_only_the_author_resubmits() -> None:
    with pytest.raises(Forbidden, match="only the author"):
        decide(facts(S.RETURNED_TO_AUTHOR), Command(A.RESUBMIT, Actor(8, Role.AUTHOR)))


@pytest.mark.parametrize(
    ("qa_score", "qa_passed", "expected"),
    [
        (0.8, True, S.QA_PASSED),  # identical
        (0.7, True, S.QA_PASSED),  # exactly at the tolerance
        (0.9, True, S.QA_PASSED),
        (0.69, True, S.QA_FAILED),  # just outside
        (0.8, False, S.QA_FAILED),  # same score, different verdict (a gate failed)
    ],
)
def test_qa_outcome(qa_score: float, qa_passed: bool, expected: Status) -> None:
    command = Command(A.SUBMIT_QA, AUDITOR, score=qa_score, passed=qa_passed)
    policy = PipelinePolicy(qa_tolerance=0.1)
    assert decide(facts(S.QA_PENDING, primary_score=0.8), command, policy) is expected


def test_policy_bounds() -> None:
    assert PipelinePolicy().qa_tolerance == pytest.approx(0.1)
    with pytest.raises(ValidationError):
        PipelinePolicy(qa_tolerance=1.5)


@given(
    primary=st.floats(0, 1),
    qa=st.floats(0, 1),
    verdict=st.booleans(),
    tolerance=st.floats(0, 1),
)
def test_qa_agreement_is_symmetric_and_zero_distance_agrees(
    primary: float, qa: float, verdict: bool, tolerance: float
) -> None:
    policy = PipelinePolicy(qa_tolerance=tolerance)
    assert qa_agrees(primary, verdict, qa, verdict, policy) == qa_agrees(
        qa, verdict, primary, verdict, policy
    )
    assert qa_agrees(primary, verdict, primary, verdict, policy)
    assert not qa_agrees(primary, verdict, primary, not verdict, policy)
