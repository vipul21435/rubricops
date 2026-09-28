"""The pipeline service end to end on SQLite: reviews, final scores, locking, rollback."""

from __future__ import annotations

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.orm import Session

from rubricops.db.models import AuditEvent, Review, Submission
from rubricops.domain.pipeline import Action, Forbidden, IllegalTransition, Status
from rubricops.domain.scoring import InvalidScoresError
from rubricops.domain.versioning import UnchangedRubricError
from rubricops.services.audit import verify_audit_chain
from rubricops.services.pipeline import (
    NotFoundError,
    RoleRequiredError,
    ScoresRequired,
    StaleSubmission,
    StoredDataError,
    TransitionResult,
)
from tests.factories import criterion_dict, make_rubric
from tests.services.conftest import World

HIGH = {"accuracy": 4, "clarity": 2}  # score 1.0, pass
AT_THRESHOLD = {"accuracy": 3, "clarity": 1}  # 0.6 * 2/3 + 0.4 * 1/2 = 0.6 exactly, pass
GATED = {"accuracy": 1, "clarity": 2}  # 0.4, and accuracy is below its gate of 2


def _submit(world: World) -> int:
    return world.service.submit(
        author_id=world.author, rubric_version_id=world.rubric_v1, payload={"answer": "42"}
    )


def _step(world: World, sid: int, action: Action, actor: int, **kwargs: object) -> TransitionResult:
    version = world.service.get_submission(sid).version
    return world.service.apply(sid, action, actor_id=actor, expected_version=version, **kwargs)  # type: ignore[arg-type]


def _to_reviewed(world: World, scores: dict[str, int] = HIGH) -> int:
    sid = _submit(world)
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _step(world, sid, Action.START, world.reviewer1)
    _step(world, sid, Action.SUBMIT_PRIMARY, world.reviewer1, scores=scores)
    return sid


def test_primary_only_path_finalizes_with_the_primary_score(world: World) -> None:
    sid = _to_reviewed(world, AT_THRESHOLD)
    result = _step(world, sid, Action.FINALIZE, world.lead1)
    assert result.from_status is Status.REVIEWED
    assert result.to_status is Status.FINALIZED
    submission = world.service.get_submission(sid)
    assert submission.status == "finalized"
    assert submission.final_score == pytest.approx(0.6)
    assert submission.final_passed is True
    assert submission.final_stage == "primary"
    assert submission.version == 5  # created at 1, four transitions
    report = verify_audit_chain(world.sessions())
    assert report.ok
    # 5 users + 1 rubric version + 1 submission + 4 transitions
    assert report.checked == world.service.count_events() == 11


def test_same_verdict_but_a_score_gap_beyond_tolerance_fails_qa(world: World) -> None:
    sid = _to_reviewed(world, HIGH)
    _step(world, sid, Action.SEND_TO_QA, world.lead1)
    qa = _step(world, sid, Action.SUBMIT_QA, world.reviewer2, scores={"accuracy": 4, "clarity": 1})
    # Both reviews pass, but 0.8 vs 1.0 is further apart than the 0.1 tolerance.
    assert qa.passed is True
    assert qa.score == pytest.approx(0.8)
    assert qa.to_status is Status.QA_FAILED


def test_qa_disagreement_goes_to_adjudication_and_the_adjudicated_score_wins(
    world: World,
) -> None:
    sid = _to_reviewed(world, HIGH)
    _step(world, sid, Action.SEND_TO_QA, world.lead1)
    qa = _step(world, sid, Action.SUBMIT_QA, world.reviewer2, scores=GATED)
    assert qa.to_status is Status.QA_FAILED
    assert qa.passed is False
    _step(world, sid, Action.ESCALATE, world.lead1)
    with pytest.raises(Forbidden, match="needs lead"):
        _step(world, sid, Action.ADJUDICATE, world.reviewer1, scores=AT_THRESHOLD)
    final = _step(world, sid, Action.ADJUDICATE, world.lead2, scores=AT_THRESHOLD)
    assert final.to_status is Status.FINALIZED
    submission = world.service.get_submission(sid)
    assert submission.final_stage == "adjudication"
    assert submission.final_score == pytest.approx(0.6)
    stages = [r.stage for r in world.service.reviews(sid)]
    assert stages == ["primary", "qa", "adjudication"]
    assert verify_audit_chain(world.sessions()).ok


def test_qa_within_tolerance_passes(world: World) -> None:
    sid = _to_reviewed(world, HIGH)
    _step(world, sid, Action.SEND_TO_QA, world.lead1)
    qa = _step(world, sid, Action.SUBMIT_QA, world.reviewer2, scores=HIGH)
    assert qa.to_status is Status.QA_PASSED
    _step(world, sid, Action.FINALIZE, world.lead1)
    assert world.service.get_submission(sid).final_score == pytest.approx(1.0)


def test_qa_auditor_cannot_be_the_primary_reviewer(world: World) -> None:
    sid = _to_reviewed(world)
    _step(world, sid, Action.SEND_TO_QA, world.lead1)
    before = world.service.count_events()
    with pytest.raises(Forbidden, match="auditor_is_not_primary"):
        _step(world, sid, Action.SUBMIT_QA, world.reviewer1, scores=HIGH)
    # A refused transition writes nothing: no review, no event, no version bump.
    assert world.service.count_events() == before
    assert [r.stage for r in world.service.reviews(sid)] == ["primary"]
    assert world.service.get_submission(sid).status == "qa_pending"


def test_reviews_stay_pinned_to_the_submissions_rubric_version(world: World) -> None:
    sid = _submit(world)
    v2 = world.service.publish_rubric(
        make_rubric(
            criterion_dict("accuracy", weight=0.5),
            criterion_dict("clarity", weight=0.3, low=0, high=2),
            criterion_dict("safety", weight=0.2),
        ),
        "add safety",
        actor_id=world.lead1,
    )
    assert v2 != world.rubric_v1
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _step(world, sid, Action.START, world.reviewer1)
    with pytest.raises(InvalidScoresError, match="unknown criteria"):
        _step(world, sid, Action.SUBMIT_PRIMARY, world.reviewer1, scores={**HIGH, "safety": 4})
    _step(world, sid, Action.SUBMIT_PRIMARY, world.reviewer1, scores=HIGH)
    (review,) = world.service.reviews(sid)
    assert review.rubric_version_id == world.rubric_v1
    with world.sessions() as session:
        last = session.scalars(select(AuditEvent).order_by(AuditEvent.seq.desc())).first()
        assert last is not None
        assert last.data["rubric_version_id"] == world.rubric_v1
        assert last.data["scores"] == HIGH


def test_double_submit_is_rejected_by_the_version_check(world: World) -> None:
    sid = _submit(world)
    world.service.apply(
        sid, Action.ASSIGN, actor_id=world.lead1, expected_version=1, assignee_id=world.reviewer1
    )
    with pytest.raises(StaleSubmission, match="is at version 2, not version 1") as info:
        world.service.apply(
            sid,
            Action.ASSIGN,
            actor_id=world.lead1,
            expected_version=1,
            assignee_id=world.reviewer2,
        )
    assert (info.value.expected, info.value.actual) == (1, 2)


def test_a_racing_writer_is_rejected_by_the_version_column(world: World) -> None:
    sid = _submit(world)
    before = world.service.count_events()

    def other_writer_commits_first(session: Session, _context: object, _objects: object) -> None:
        # Simulates a concurrent transaction that committed after this one read the row.
        session.connection().execute(
            text("UPDATE submissions SET version = version + 1 WHERE id = :id"), {"id": sid}
        )

    event.listen(Session, "before_flush", other_writer_commits_first, once=True)
    try:
        with pytest.raises(StaleSubmission, match="changed concurrently"):
            world.service.apply(
                sid,
                Action.ASSIGN,
                actor_id=world.lead1,
                expected_version=1,
                assignee_id=world.reviewer1,
            )
    finally:
        if event.contains(Session, "before_flush", other_writer_commits_first):
            event.remove(Session, "before_flush", other_writer_commits_first)
    assert world.service.count_events() == before
    assert world.service.get_submission(sid).status == "queued"


def test_return_and_resubmit_starts_a_new_round(world: World) -> None:
    sid = _submit(world)
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _step(world, sid, Action.RELEASE, world.reviewer1)
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer2)
    with pytest.raises(Forbidden, match="not the assigned reviewer"):
        _step(world, sid, Action.START, world.reviewer1)
    _step(world, sid, Action.START, world.reviewer2)
    _step(world, sid, Action.RETURN_TO_AUTHOR, world.reviewer2)
    resubmitted = _step(world, sid, Action.RESUBMIT, world.author)
    assert resubmitted.to_status is Status.QUEUED
    assert world.service.get_submission(sid).round == 2
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _step(world, sid, Action.START, world.reviewer1)
    _step(world, sid, Action.SUBMIT_PRIMARY, world.reviewer1, scores=HIGH)
    _step(world, sid, Action.SEND_TO_QA, world.lead1)
    _step(world, sid, Action.SUBMIT_QA, world.reviewer2, scores=GATED)
    _step(world, sid, Action.ESCALATE, world.lead1)
    _step(world, sid, Action.RETURN_TO_AUTHOR, world.lead1)
    assert world.service.get_submission(sid).status == "returned_to_author"
    assert verify_audit_chain(world.sessions()).ok


def test_scores_are_required_exactly_for_reviewing_actions(world: World) -> None:
    sid = _submit(world)
    with pytest.raises(ScoresRequired, match="assign does not take criterion scores"):
        _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1, scores=HIGH)
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _step(world, sid, Action.START, world.reviewer1)
    with pytest.raises(ScoresRequired, match="submit_primary needs criterion scores"):
        _step(world, sid, Action.SUBMIT_PRIMARY, world.reviewer1)


def test_illegal_action_is_reported_before_scores_are_checked(world: World) -> None:
    sid = _submit(world)
    with pytest.raises(IllegalTransition, match="cannot adjudicate a submission that is queued"):
        _step(world, sid, Action.ADJUDICATE, world.lead1, scores={"bogus": 1})


def test_missing_rows(world: World) -> None:
    with pytest.raises(NotFoundError, match="no submission 999"):
        world.service.get_submission(999)
    with pytest.raises(NotFoundError, match="no submission 999"):
        world.service.apply(999, Action.ASSIGN, actor_id=world.lead1, expected_version=1)
    sid = _submit(world)
    with pytest.raises(NotFoundError, match="no user 999"):
        world.service.apply(sid, Action.ASSIGN, actor_id=999, expected_version=1)
    with pytest.raises(NotFoundError, match="no rubric version 999"):
        world.service.submit(author_id=world.author, rubric_version_id=999, payload={})


def test_setup_roles_and_unchanged_rubric(world: World) -> None:
    with pytest.raises(RoleRequiredError, match="only a lead publishes"):
        world.service.publish_rubric(make_rubric(threshold=0.5), "x", actor_id=world.reviewer1)
    with pytest.raises(RoleRequiredError, match="only an author submits"):
        world.service.submit(author_id=world.lead1, rubric_version_id=world.rubric_v1, payload={})
    with pytest.raises(UnchangedRubricError, match="unchanged since v1"):
        world.service.publish_rubric(make_rubric(), "same again", actor_id=world.lead1)


def test_a_rubric_body_that_no_longer_matches_its_hash_is_refused(world: World) -> None:
    sid = _submit(world)
    _step(world, sid, Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    _step(world, sid, Action.START, world.reviewer1)
    with world.sessions() as session:
        session.execute(text("DROP TRIGGER rubric_versions_no_update"))
        body = session.scalar(text("SELECT body FROM rubric_versions WHERE id = :id"), {"id": 1})
        assert isinstance(body, str)
        session.execute(
            text("UPDATE rubric_versions SET body = :b WHERE id = :id"),
            {"b": body.replace('"pass_threshold":0.6', '"pass_threshold":0.1'), "id": 1},
        )
        session.commit()
    with pytest.raises(StoredDataError, match="does not match its stored hash"):
        _step(world, sid, Action.SUBMIT_PRIMARY, world.reviewer1, scores=GATED)


def test_final_fields_and_reviews_are_consistent(world: World) -> None:
    sid = _to_reviewed(world, GATED)
    _step(world, sid, Action.FINALIZE, world.lead1)
    with world.sessions() as session:
        submission = session.get(Submission, sid)
        assert submission is not None
        assert submission.final_passed is False  # the gate failed despite a 0.4 total
        primary = session.scalars(select(Review).where(Review.stage == "primary")).one()
        assert primary.passed is False
        assert primary.score == pytest.approx(0.4)
