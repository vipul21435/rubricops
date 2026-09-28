"""The queue service on SQLite: assignment through the pipeline, SLA sweeps, QA sampling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from rubricops.db.models import AuditEvent
from rubricops.domain.clock import FrozenClock
from rubricops.domain.pipeline import Action, Forbidden, Status
from rubricops.domain.queue import LoadBalanced, NoEligibleReviewer, RoundRobin, SkillTagMatch
from rubricops.domain.sampling import QaSampler, Reason, SamplingRules
from rubricops.domain.sla import SlaPolicy
from rubricops.services.audit import verify_audit_chain
from rubricops.services.pipeline import NotFoundError, PipelineService
from rubricops.services.queue import NotReviewedError, QueueService
from tests.services.conftest import World

HIGH = {"accuracy": 4, "clarity": 2}  # score 1.0
AT_THRESHOLD = {"accuracy": 3, "clarity": 1}  # score 0.6, the rubric's threshold
T0 = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


def _queue(
    world: World,
    *,
    rate: float = 0.0,
    min_reviews: int = 0,
    clock: FrozenClock | None = None,
    flagged: frozenset[int] = frozenset(),
) -> QueueService:
    sampler = QaSampler(SamplingRules(rate=rate, new_reviewer_min_reviews=min_reviews), 7)
    sla = SlaPolicy(timedelta(hours=24))
    at = clock or FrozenClock(T0)
    if not flagged:  # the default hook: nobody is flagged until calibration exists
        return QueueService(world.sessions, at, world.service, sla=sla, sampler=sampler)
    return QueueService(
        world.sessions,
        at,
        world.service,
        sla=sla,
        sampler=sampler,
        is_flagged=lambda rid: rid in flagged,
    )


def _submit(world: World, tags: tuple[str, ...] = ()) -> int:
    return world.service.submit(
        author_id=world.author,
        rubric_version_id=world.rubric_v1,
        payload={"answer": "42"},
        skill_tags=tags,
    )


def _step(world: World, sid: int, action: Action, actor: int, **kwargs: object) -> None:
    version = world.service.get_submission(sid).version
    world.service.apply(sid, action, actor_id=actor, expected_version=version, **kwargs)  # type: ignore[arg-type]


def _review(world: World, sid: int, reviewer: int, scores: dict[str, int]) -> None:
    _step(world, sid, Action.START, reviewer)
    _step(world, sid, Action.SUBMIT_PRIMARY, reviewer, scores=scores)


def _last_context(world: World) -> dict[str, object]:
    with world.sessions() as session:
        event = session.scalars(select(AuditEvent).order_by(AuditEvent.seq.desc())).first()
        assert event is not None
        return dict(event.data["context"])


# -- assign_next --------------------------------------------------------------


def test_assign_next_takes_the_oldest_item_and_records_the_policy(world: World) -> None:
    queue = _queue(world)
    first, second = _submit(world), _submit(world)
    outcome = queue.assign_next(LoadBalanced(), actor_id=world.lead1)
    assert outcome is not None
    assert outcome.submission_id == first
    assert outcome.reviewer_id == world.reviewer1  # both idle: lowest id
    assert outcome.due_at is not None
    assert world.service.get_submission(first).status == Status.ASSIGNED.value
    assert _last_context(world) == {"policy": "load-balanced"}
    # reviewer1 now has one open assignment, so load balancing moves on.
    outcome = queue.assign_next(LoadBalanced(), actor_id=world.lead1)
    assert outcome is not None
    assert (outcome.submission_id, outcome.reviewer_id) == (second, world.reviewer2)
    assert queue.assign_next(LoadBalanced(), actor_id=world.lead1) is None
    assert verify_audit_chain(world.sessions()).ok


def test_round_robin_resumes_from_the_last_assignment_in_the_database(world: World) -> None:
    queue = _queue(world)
    sids = [_submit(world) for _ in range(3)]
    picks = []
    for _ in sids:
        # A fresh policy each time: the cursor comes from the assignments table.
        outcome = queue.assign_next(RoundRobin(), actor_id=world.lead1)
        assert outcome is not None
        picks.append(outcome.reviewer_id)
    assert picks == [world.reviewer1, world.reviewer2, world.reviewer1]
    assert _last_context(world) == {"policy": "round-robin", "cursor": world.reviewer2}


def test_skill_match_and_refusal_leave_the_item_queued(world: World) -> None:
    queue = _queue(world)
    sid = _submit(world, ("python",))
    outcome = queue.assign_next(SkillTagMatch(), actor_id=world.lead1)
    assert outcome is not None
    assert outcome.reviewer_id == world.reviewer1  # the only python reviewer
    rust = _submit(world, ("rust",))
    with pytest.raises(NoEligibleReviewer) as info:
        queue.assign_next(SkillTagMatch(), actor_id=world.lead1)
    assert info.value.item.submission_id == rust
    assert world.service.get_submission(rust).status == Status.QUEUED.value
    assert world.service.get_submission(sid).status == Status.ASSIGNED.value


def test_only_reviewers_are_in_the_pool(world: World) -> None:
    queue = _queue(world)
    for _ in range(4):
        _submit(world)
    chosen = set()
    for _ in range(4):
        outcome = queue.assign_next(LoadBalanced(), actor_id=world.lead1)
        assert outcome is not None
        chosen.add(outcome.reviewer_id)
    assert chosen == {world.reviewer1, world.reviewer2}  # never the author or a lead


# -- sweep_overdue --------------------------------------------------------------


def test_sweep_uses_the_per_rubric_sla_and_the_injected_clock(world: World) -> None:
    pipeline = PipelineService(
        world.sessions,
        FrozenClock(T0),
        sla=SlaPolicy(timedelta(hours=24), {"sample-rubric": timedelta(hours=2)}),
    )
    clock = FrozenClock(T0)
    queue = QueueService(
        world.sessions,
        clock,
        pipeline,
        sla=SlaPolicy(timedelta(hours=24)),
        sampler=QaSampler(SamplingRules(rate=0.0), 7),
    )
    sid = _submit(world)
    outcome = queue.assign_next(LoadBalanced(), actor_id=world.lead1)
    assert outcome is not None
    assert outcome.due_at == T0 + timedelta(hours=2)
    clock.set(T0 + timedelta(hours=2))
    assert queue.sweep_overdue() == []
    clock.advance(timedelta(minutes=30))
    [late] = queue.sweep_overdue()
    assert late.assignment.submission_id == sid
    assert late.lateness == timedelta(minutes=30)
    # Submitting the review completes the assignment, so it is no longer overdue.
    _review(world, sid, outcome.reviewer_id, HIGH)
    assert queue.sweep_overdue() == []


# -- sample_for_qa ------------------------------------------------------------


def _reviewed(world: World, scores: dict[str, int]) -> int:
    sid = _submit(world)
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _review(world, sid, world.reviewer1, scores)
    return sid


def test_a_safe_review_is_finalized_and_the_decision_is_audited(world: World) -> None:
    sid = _reviewed(world, HIGH)
    decision = _queue(world).sample_for_qa(sid, actor_id=world.lead1)
    assert not decision.sampled
    assert world.service.get_submission(sid).status == Status.FINALIZED.value
    context = _last_context(world)["qa_sampling"]
    assert isinstance(context, dict)
    assert context["sampled"] is False
    assert context["reasons"] == []
    assert context["seed"] == 7


def test_risky_reviews_go_to_qa_with_their_reasons(world: World) -> None:
    near = _reviewed(world, AT_THRESHOLD)
    decision = _queue(world).sample_for_qa(near, actor_id=world.lead1)
    assert decision.reasons == (Reason.NEAR_THRESHOLD,)
    assert world.service.get_submission(near).status == Status.QA_PENDING.value
    context = _last_context(world)["qa_sampling"]
    assert isinstance(context, dict)
    assert context["reasons"] == ["near_threshold"]

    newcomer = _reviewed(world, HIGH)  # reviewer1 has one earlier primary review
    decision = _queue(world, min_reviews=2).sample_for_qa(newcomer, actor_id=world.lead1)
    assert decision.reasons == (Reason.NEW_REVIEWER,)
    assert decision.details == ("1 completed reviews < 2",)

    flagged = _reviewed(world, HIGH)
    decision = _queue(world, flagged=frozenset({world.reviewer1})).sample_for_qa(
        flagged, actor_id=world.lead1
    )
    assert decision.reasons == (Reason.CALIBRATION_FLAG,)


def test_earlier_round_scores_count_towards_item_agreement(world: World) -> None:
    sid = _submit(world)
    # Round 1: primary 0.0, QA 1.0, escalated, and sent back to the author by a lead.
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _review(world, sid, world.reviewer1, {"accuracy": 1, "clarity": 0})
    _step(world, sid, Action.SEND_TO_QA, world.lead1)
    _step(world, sid, Action.SUBMIT_QA, world.reviewer2, scores=HIGH)
    _step(world, sid, Action.ESCALATE, world.lead1)
    _step(world, sid, Action.RETURN_TO_AUTHOR, world.lead2)
    _step(world, sid, Action.RESUBMIT, world.author)
    # Round 2: a clean primary review, but the item's scores so far span 0.0..1.0.
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _review(world, sid, world.reviewer1, HIGH)
    decision = _queue(world).sample_for_qa(sid, actor_id=world.lead1)
    assert decision.reasons == (Reason.LOW_AGREEMENT,)
    assert decision.details == ("item scores span 1.0000 > 0.25",)
    assert world.service.get_submission(sid).status == Status.QA_PENDING.value


def test_sampling_is_deterministic_for_a_seed(world: World) -> None:
    sids = [_reviewed(world, HIGH) for _ in range(20)]
    queue = _queue(world, rate=0.5)
    decisions = [queue.sample_for_qa(sid, actor_id=world.lead1) for sid in sids]
    replay = QaSampler(SamplingRules(rate=0.5, new_reviewer_min_reviews=0), 7)
    assert [d.draw for d in decisions] == [replay.draw(sid) for sid in sids]
    assert 0 < sum(d.sampled for d in decisions) < 20


def test_sampling_refuses_items_that_are_not_reviewed(world: World) -> None:
    queue = _queue(world)
    sid = _submit(world)
    with pytest.raises(NotReviewedError, match="is queued, not reviewed"):
        queue.sample_for_qa(sid, actor_id=world.lead1)
    with pytest.raises(NotFoundError):
        queue.sample_for_qa(999, actor_id=world.lead1)


# -- review regressions -----------------------------------------------------------


def test_an_unassignable_item_does_not_block_the_rest_of_the_queue(world: World) -> None:
    queue = _queue(world)
    rust = _submit(world, ("rust",))  # nobody has rust, and it is the oldest item
    python = [_submit(world, ("python",)) for _ in range(3)]
    picked = []
    for _ in python:
        outcome = queue.assign_next(SkillTagMatch(), actor_id=world.lead1)
        assert outcome is not None
        picked.append(outcome.submission_id)
    assert picked == python
    assert world.service.get_submission(rust).status == Status.QUEUED.value
    # Only the refused item is left: its refusal is raised, and it stays queued.
    with pytest.raises(NoEligibleReviewer) as info:
        queue.assign_next(SkillTagMatch(), actor_id=world.lead1)
    assert info.value.item.submission_id == rust
    assert world.service.get_submission(rust).status == Status.QUEUED.value


def test_a_failed_assignment_does_not_move_the_round_robin_cursor(world: World) -> None:
    queue = _queue(world)
    sid = _submit(world)
    rr = RoundRobin()
    with pytest.raises(Forbidden):
        queue.assign_next(rr, actor_id=world.reviewer2)  # only a lead may assign
    assert rr.cursor is None  # nothing in the assignments table yet
    outcome = queue.assign_next(rr, actor_id=world.lead1)
    assert outcome is not None
    assert (outcome.submission_id, outcome.reviewer_id) == (sid, world.reviewer1)
    assert rr.cursor == world.reviewer1


def test_new_reviewer_counts_only_reviews_written_before_the_sampled_one(world: World) -> None:
    sids = [_reviewed(world, HIGH) for _ in range(3)]
    queue = _queue(world, min_reviews=2)
    # A batched sweep after all three reviews gives the same answers as sampling
    # each review as soon as it is written.
    decisions = [queue.sample_for_qa(sid, actor_id=world.lead1) for sid in sids]
    assert [d.reasons for d in decisions] == [
        (Reason.NEW_REVIEWER,),
        (Reason.NEW_REVIEWER,),
        (),
    ]
    context = _last_context(world)["qa_sampling"]
    assert isinstance(context, dict)
    assert context["reviewer_completed_reviews"] == 2  # recorded even when the rule is quiet
