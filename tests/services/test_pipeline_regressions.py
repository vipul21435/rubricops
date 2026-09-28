"""Regression tests for review findings on the pipeline service.

- QA tolerance is compared on exact rubric scores, not on rounded floats.
- A racing double submit of a reviewing action is a StaleSubmission, not a raw
  IntegrityError.
- Assignment bookkeeping (completed_at, released_at, due_at) and round isolation.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest
from sqlalchemy import select

import rubricops.services.pipeline as pipeline_module
from rubricops.db.engine import make_engine, make_session_factory
from rubricops.db.migrate import upgrade
from rubricops.db.models import Assignment, Base, Submission
from rubricops.domain.clock import SteppingClock
from rubricops.domain.pipeline import (
    Action,
    Actor,
    Command,
    Facts,
    PipelinePolicy,
    Role,
    Status,
    decide,
    qa_agrees,
)
from rubricops.domain.rubric import Rubric
from rubricops.domain.scoring import exact_score, score_review
from rubricops.loaders import load_rubric
from rubricops.services.pipeline import PipelineService, StaleSubmission, TransitionResult
from tests.services.conftest import World

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "rubrics"
PRIMARY = {"correctness": 3, "completeness": 4, "clarity": 2, "safe_advice": 2}  # 11/15
AUDIT = {"correctness": 4, "completeness": 2, "clarity": 4, "safe_advice": 2}  # 5/6


def _all_maps(rubric: Rubric) -> Iterator[dict[str, int]]:
    ranges = [range(c.scale.low, c.scale.high + 1) for c in rubric.criteria]
    for values in itertools.product(*ranges):
        yield dict(zip(rubric.criterion_ids, values, strict=True))


def test_exact_score_is_the_unrounded_total_of_score_review() -> None:
    rubric = load_rubric(EXAMPLES / "code-explanation.yaml")
    assert exact_score(rubric, PRIMARY) == Fraction(11, 15)
    assert exact_score(rubric, AUDIT) == Fraction(5, 6)
    for scores in _all_maps(rubric):
        assert float(exact_score(rubric, scores)) == score_review(rubric, scores).score


@pytest.mark.parametrize(
    "name", ["code-explanation.yaml", "code-explanation.v1.yaml", "action-items.yaml"]
)
@pytest.mark.parametrize("tolerance", [0.05, 0.1, 0.2])
def test_every_pair_exactly_at_the_tolerance_agrees(name: str, tolerance: float) -> None:
    rubric = load_rubric(EXAMPLES / name)
    policy = PipelinePolicy(qa_tolerance=tolerance)
    limit = Fraction(repr(tolerance))
    scored = {
        (s, bool(score_review(rubric, dict(scores)).passed))
        for scores in _all_maps(rubric)
        for s in [exact_score(rubric, scores)]
    }
    at_limit = inside = outside = 0
    for (a, verdict_a), (b, verdict_b) in itertools.product(scored, scored):
        if verdict_a != verdict_b:
            continue
        agrees = qa_agrees(a, verdict_a, b, verdict_b, policy)
        distance = abs(a - b)
        assert agrees == (distance <= limit), (a, b)
        at_limit += distance == limit
        inside += distance < limit
        outside += distance > limit
    assert inside
    assert outside


def test_decide_passes_qa_for_scores_exactly_one_tenth_apart() -> None:
    facts = Facts(
        status=Status.QA_PENDING,
        author_id=1,
        primary_reviewer_id=2,
        primary_score=Fraction(11, 15),
        primary_passed=True,
    )
    command = Command(Action.SUBMIT_QA, Actor(3, Role.REVIEWER), score=Fraction(5, 6), passed=True)
    assert decide(facts, command, PipelinePolicy(qa_tolerance=0.1)) is Status.QA_PASSED
    # The rounded floats would have failed it.
    assert not qa_agrees(
        float(Fraction(11, 15)), True, float(Fraction(5, 6)), True, PipelinePolicy()
    )


def _service(url: str) -> tuple[PipelineService, dict[str, int]]:
    engine = make_engine(url)
    if url == "sqlite://":
        Base.metadata.create_all(engine)
    service = PipelineService(
        make_session_factory(engine), SteppingClock(datetime(2026, 9, 1, 9, 0, tzinfo=UTC))
    )
    lead = service.add_user("lead", Role.LEAD)
    ids = {
        "lead": lead,
        "author": service.add_user("author", Role.AUTHOR),
        "r1": service.add_user("r1", Role.REVIEWER),
        "r2": service.add_user("r2", Role.REVIEWER),
        "r3": service.add_user("r3", Role.REVIEWER),
        "rv": service.publish_rubric(
            load_rubric(EXAMPLES / "code-explanation.yaml"), "v1", actor_id=lead
        ),
    }
    return service, ids


def _step(
    service: PipelineService, sid: int, action: Action, actor: int, **kw: object
) -> TransitionResult:
    version = service.get_submission(sid).version
    return service.apply(sid, action, actor_id=actor, expected_version=version, **kw)  # type: ignore[arg-type]


def _to_in_review(service: PipelineService, ids: dict[str, int]) -> int:
    sid = service.submit(author_id=ids["author"], rubric_version_id=ids["rv"], payload={"a": 1})
    _step(service, sid, Action.ASSIGN, ids["lead"], assignee_id=ids["r1"])
    _step(service, sid, Action.START, ids["r1"])
    return sid


def test_service_passes_qa_when_the_exact_scores_are_at_the_tolerance() -> None:
    service, ids = _service("sqlite://")
    sid = _to_in_review(service, ids)
    primary = _step(service, sid, Action.SUBMIT_PRIMARY, ids["r1"], scores=PRIMARY)
    assert primary.score == pytest.approx(11 / 15)
    _step(service, sid, Action.SEND_TO_QA, ids["lead"])
    qa = _step(service, sid, Action.SUBMIT_QA, ids["r2"], scores=AUDIT)
    assert qa.to_status is Status.QA_PASSED
    final = _step(service, sid, Action.FINALIZE, ids["lead"])
    assert final.to_status is Status.FINALIZED
    assert service.get_submission(sid).final_stage == "primary"


def _race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: Action, actors: tuple[str, str]
) -> list[object]:
    url = f"sqlite:///{tmp_path / 'race.db'}"
    upgrade(url)
    service, ids = _service(url)
    sid = _to_in_review(service, ids)
    scores = PRIMARY
    if action is Action.SUBMIT_QA:
        _step(service, sid, Action.SUBMIT_PRIMARY, ids["r1"], scores=PRIMARY)
        _step(service, sid, Action.SEND_TO_QA, ids["lead"])
        scores = AUDIT
    version = service.get_submission(sid).version

    first_done = threading.Event()
    real_decide = pipeline_module.decide

    def paused_decide(*args: object, **kwargs: object) -> Status:
        target = real_decide(*args, **kwargs)  # type: ignore[arg-type]
        if threading.current_thread().name == "second":
            first_done.wait(timeout=10)
        return target

    monkeypatch.setattr(pipeline_module, "decide", paused_decide)
    outcomes: dict[str, object] = {}

    def run(name: str, actor: str) -> None:
        try:
            outcomes[name] = service.apply(
                sid, action, actor_id=ids[actor], expected_version=version, scores=scores
            )
        except Exception as exc:
            outcomes[name] = exc
        finally:
            if name == "first":
                first_done.set()

    second = threading.Thread(target=run, args=("second", actors[1]), name="second")
    second.start()
    first = threading.Thread(target=run, args=("first", actors[0]), name="first")
    first.start()
    first.join(timeout=20)
    second.join(timeout=20)
    return [outcomes["first"], outcomes["second"]]


def test_a_racing_double_primary_submit_is_a_stale_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _race(tmp_path, monkeypatch, Action.SUBMIT_PRIMARY, ("r1", "r1"))
    assert isinstance(first, TransitionResult)
    assert first.to_status is Status.REVIEWED
    assert isinstance(second, StaleSubmission)
    assert "changed concurrently" in str(second)


def test_two_racing_auditors_get_a_stale_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _race(tmp_path, monkeypatch, Action.SUBMIT_QA, ("r2", "r3"))
    assert isinstance(first, TransitionResult)
    assert isinstance(second, StaleSubmission)


def _assignments(world: World, sid: int) -> list[Assignment]:
    with world.sessions() as session:
        return list(
            session.scalars(
                select(Assignment).where(Assignment.submission_id == sid).order_by(Assignment.id)
            )
        )


def test_assignment_rows_record_due_completed_and_released_times(world: World) -> None:
    service = world.service
    sid = service.submit(author_id=world.author, rubric_version_id=world.rubric_v1, payload={})
    v = service.get_submission(sid).version
    service.apply(
        sid, Action.ASSIGN, actor_id=world.lead1, expected_version=v, assignee_id=world.reviewer1
    )
    v = service.get_submission(sid).version
    service.apply(sid, Action.RELEASE, actor_id=world.reviewer1, expected_version=v)
    v = service.get_submission(sid).version
    service.apply(
        sid, Action.ASSIGN, actor_id=world.lead1, expected_version=v, assignee_id=world.reviewer2
    )
    v = service.get_submission(sid).version
    service.apply(sid, Action.START, actor_id=world.reviewer2, expected_version=v)
    v = service.get_submission(sid).version
    service.apply(
        sid,
        Action.SUBMIT_PRIMARY,
        actor_id=world.reviewer2,
        expected_version=v,
        scores={"accuracy": 4, "clarity": 2},
    )
    released, completed = _assignments(world, sid)
    assert released.reviewer_id == world.reviewer1
    assert released.released_at is not None
    assert released.completed_at is None
    assert released.due_at is not None
    assert released.due_at - released.assigned_at == timedelta(hours=24)
    assert completed.reviewer_id == world.reviewer2
    assert completed.completed_at is not None
    assert completed.released_at is None
    assert completed.due_at - completed.assigned_at == timedelta(hours=24)


def test_reviews_of_an_earlier_round_never_count_in_the_next_one(world: World) -> None:
    service = world.service
    sid = service.submit(author_id=world.author, rubric_version_id=world.rubric_v1, payload={})

    def step(action: Action, actor: int, **kw: object) -> TransitionResult:
        v = service.get_submission(sid).version
        return service.apply(sid, action, actor_id=actor, expected_version=v, **kw)  # type: ignore[arg-type]

    # Round 1: reviewer1 passes it, reviewer2's audit fails a gate, a lead sends it back.
    step(Action.ASSIGN, world.lead1, assignee_id=world.reviewer1)
    step(Action.START, world.reviewer1)
    step(Action.SUBMIT_PRIMARY, world.reviewer1, scores={"accuracy": 4, "clarity": 2})
    step(Action.SEND_TO_QA, world.lead1)
    audit = step(Action.SUBMIT_QA, world.reviewer2, scores={"accuracy": 1, "clarity": 2})
    assert audit.to_status is Status.QA_FAILED
    step(Action.ESCALATE, world.lead1)
    step(Action.RETURN_TO_AUTHOR, world.lead1)
    step(Action.RESUBMIT, world.author)

    with world.sessions() as session:
        submission = session.get(Submission, sid)
        assert submission is not None
        assert submission.round == 2
        fresh = PipelineService._round(session, submission)
    assert fresh.assignment is None
    assert fresh.primary is None
    assert fresh.qa is None

    # Round 2: roles swap. reviewer2 (round 1's auditor) does the primary review and
    # reviewer1 (round 1's primary reviewer) may audit it.
    step(Action.ASSIGN, world.lead1, assignee_id=world.reviewer2)
    step(Action.START, world.reviewer2)
    primary = step(Action.SUBMIT_PRIMARY, world.reviewer2, scores={"accuracy": 1, "clarity": 2})
    assert primary.passed is False
    step(Action.SEND_TO_QA, world.lead1)
    qa = step(Action.SUBMIT_QA, world.reviewer1, scores={"accuracy": 1, "clarity": 2})
    assert qa.to_status is Status.QA_PASSED
    step(Action.FINALIZE, world.lead1)
    submission = service.get_submission(sid)
    assert submission.final_passed is False
    assert submission.final_score == pytest.approx(primary.score)
    rounds = [(r.round, r.stage, r.reviewer_id) for r in service.reviews(sid)]
    assert rounds == [
        (1, "primary", world.reviewer1),
        (1, "qa", world.reviewer2),
        (2, "primary", world.reviewer2),
        (2, "qa", world.reviewer1),
    ]
