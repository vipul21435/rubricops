"""Random walks through the pipeline: whatever happens, the invariants hold."""

from __future__ import annotations

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st
from sqlalchemy import select

from rubricops.db.models import Review
from rubricops.domain.pipeline import (
    REVIEW_STAGE,
    Action,
    Forbidden,
    IllegalTransition,
    Status,
    allowed_actions,
)
from rubricops.services.audit import verify_audit_chain
from tests.services.conftest import World, build_world

ACTIONS = sorted(Action, key=lambda a: a.value)

step = st.tuples(
    # Mostly pick from the currently legal actions (and try every user until one is
    # allowed), sometimes any action by the drawn user, so walks get deep and also
    # probe refusals.
    st.sampled_from([True, True, True, False]),
    st.integers(0, 10),  # which action
    st.integers(0, 4),  # which user acts
    st.integers(0, 4),  # which user is assigned (for assign)
    st.integers(1, 4),  # accuracy score
    st.integers(0, 2),  # clarity score
)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(steps=st.lists(step, min_size=5, max_size=40))
def test_random_walk_keeps_the_invariants(
    steps: list[tuple[bool, int, int, int, int, int]],
) -> None:
    world = build_world()
    try:
        users = [world.author, world.reviewer1, world.reviewer2, world.lead1, world.lead2]
        sid = world.service.submit(
            author_id=world.author, rubric_version_id=world.rubric_v1, payload={}
        )
        applied = 0
        events = world.service.count_events()
        for legal_only, which, actor, assignee, accuracy, clarity in steps:
            submission = world.service.get_submission(sid)
            status = Status(submission.status)
            choices = sorted(allowed_actions(status), key=lambda a: a.value) if legal_only else []
            action = (choices or ACTIONS)[which % len(choices or ACTIONS)]
            scores = {"accuracy": accuracy, "clarity": clarity} if action in REVIEW_STAGE else None
            order = users[actor:] + users[:actor] if legal_only else [users[actor]]
            succeeded = False
            for user in order:
                try:
                    world.service.apply(
                        sid,
                        action,
                        actor_id=user,
                        expected_version=submission.version,
                        assignee_id=users[assignee] if action is Action.ASSIGN else None,
                        scores=scores,
                    )
                except (IllegalTransition, Forbidden):
                    # A refused transition writes nothing.
                    assert world.service.count_events() == events
                    assert world.service.get_submission(sid).version == submission.version
                    continue
                succeeded = True
                break
            if not succeeded:
                continue
            applied += 1
            events += 1
            assert world.service.count_events() == events
            assert world.service.get_submission(sid).version == submission.version + 1

        _check_final_state(world, sid, events=events, applied=applied)
    finally:
        world.engine.dispose()


def _check_final_state(world: World, sid: int, *, events: int, applied: int) -> None:
    final = world.service.get_submission(sid)
    event(f"final status {final.status}")
    with world.sessions() as session:
        report = verify_audit_chain(session)
        reviews = list(session.scalars(select(Review).where(Review.submission_id == sid)))
    # One audit event per applied transition, all on one intact chain.
    assert report.ok
    assert report.checked == events
    assert final.version == 1 + applied
    # Every review is pinned to the submission's rubric version.
    assert all(r.rubric_version_id == final.rubric_version_id for r in reviews)
    current = {r.stage: r for r in reviews if r.round == final.round}
    if final.status == Status.FINALIZED.value:
        # Nothing is finalized without a primary review in the current round ...
        assert "primary" in current
        # ... and the final score is the adjudicated one if there is one, else the primary.
        source = current.get("adjudication", current["primary"])
        assert final.final_stage == source.stage
        assert final.final_score == source.score
        assert final.final_passed == source.passed
        # The QA auditor and the adjudicator are independent of the primary reviewer.
        if "qa" in current:
            assert current["qa"].reviewer_id != current["primary"].reviewer_id
        if "adjudication" in current:
            others = {current["primary"].reviewer_id, current["qa"].reviewer_id}
            assert current["adjudication"].reviewer_id not in others
    else:
        assert final.final_score is None
