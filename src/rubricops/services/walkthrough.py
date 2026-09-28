"""A scripted, deterministic run of the review pipeline against a real database.

It migrates a fresh database, adds five users and one rubric, then takes three
submissions down the three routes through the pipeline:

1. primary review only (not sampled for QA);
2. primary review confirmed by a QA audit;
3. a QA audit that disagrees, escalated and settled by a lead's adjudication.

Along the way it shows two refusals, an audit by the primary reviewer and a stale
double submit, and ends by verifying the audit chain. A :class:`SteppingClock`
starting at a fixed instant makes every timestamp, and so every hash, the same on
each run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select

from rubricops.db import migrate
from rubricops.db.engine import make_engine, make_session_factory
from rubricops.db.models import User
from rubricops.domain.clock import SteppingClock
from rubricops.domain.pipeline import Action, PipelineError, Role
from rubricops.domain.rubric import Rubric
from rubricops.services.audit import ChainReport, verify_audit_chain
from rubricops.services.pipeline import PipelineService, StaleSubmission, TransitionResult

WALKTHROUGH_START = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


class WalkthroughError(RuntimeError):
    """The target database is not empty."""


Echo = Callable[[str], None]


class _Runner:
    def __init__(self, service: PipelineService, names: Mapping[int, str], echo: Echo) -> None:
        self.service = service
        self.names = names
        self.echo = echo

    def step(
        self,
        sid: int,
        action: Action,
        actor: int,
        *,
        assignee: int | None = None,
        scores: Mapping[str, int] | None = None,
    ) -> TransitionResult:
        version = self.service.get_submission(sid).version
        result = self.service.apply(
            sid,
            action,
            actor_id=actor,
            expected_version=version,
            assignee_id=assignee,
            scores=scores,
        )
        detail = ""
        if result.score is not None:
            verdict = "PASS" if result.passed else "FAIL"
            detail = f"  score {result.score:.4f} {verdict}"
        if assignee is not None:
            detail = f"  -> {self.names[assignee]}"
        self.echo(
            f"  {action.value:<15} {result.from_status.value:>15} -> "
            f"{result.to_status.value:<16} by {self.names[actor]:<10} "
            f"v{result.version} event #{result.audit_seq}{detail}"
        )
        return result

    def refused(self, sid: int, action: Action, actor: int, **kwargs: object) -> None:
        version = self.service.get_submission(sid).version
        try:
            self.service.apply(sid, action, actor_id=actor, expected_version=version, **kwargs)  # type: ignore[arg-type]
        except PipelineError as exc:
            self.echo(f"  refused: {exc}")
        else:  # pragma: no cover - the walkthrough only calls this for refused actions
            msg = f"{action.value} by {self.names[actor]} was expected to be refused"
            raise AssertionError(msg)


@dataclass(frozen=True, slots=True)
class _Cast:
    lead1: int
    lead2: int
    author: int
    rev1: int
    rev2: int
    rubric_version: int

    @property
    def names(self) -> dict[int, str]:
        return {
            self.lead1: "lead-1",
            self.lead2: "lead-2",
            self.author: "author-1",
            self.rev1: "reviewer-1",
            self.rev2: "reviewer-2",
        }


def _cast(service: PipelineService, rubric: Rubric, echo: Echo) -> _Cast:
    lead1 = service.add_user("lead-1", Role.LEAD)
    cast = _Cast(
        lead1=lead1,
        lead2=service.add_user("lead-2", Role.LEAD),
        author=service.add_user("author-1", Role.AUTHOR),
        rev1=service.add_user("reviewer-1", Role.REVIEWER),
        rev2=service.add_user("reviewer-2", Role.REVIEWER),
        rubric_version=service.publish_rubric(rubric, "initial version", actor_id=lead1),
    )
    echo(f"published rubric {rubric.id} v1 as rubric_version {cast.rubric_version}")
    return cast


def _scenarios(run: _Runner, cast: _Cast, scores: Mapping[str, Mapping[str, int]]) -> list[int]:
    service = run.service

    def submit(label: str) -> int:
        sid = service.submit(
            author_id=cast.author, rubric_version_id=cast.rubric_version, payload={"label": label}
        )
        run.echo(f"\nsubmission {sid} ({label}) queued by author-1")
        run.step(sid, Action.ASSIGN, cast.lead1, assignee=cast.rev1)
        run.step(sid, Action.START, cast.rev1)
        return sid

    first = submit("primary review only")
    run.step(first, Action.SUBMIT_PRIMARY, cast.rev1, scores=scores["good"])
    stale = service.get_submission(first).version
    run.step(first, Action.FINALIZE, cast.lead1)
    try:
        service.apply(first, Action.FINALIZE, actor_id=cast.lead1, expected_version=stale)
    except StaleSubmission as exc:
        run.echo(f"  refused: {exc}")

    second = submit("QA confirms")
    run.step(second, Action.SUBMIT_PRIMARY, cast.rev1, scores=scores["better"])
    run.step(second, Action.SEND_TO_QA, cast.lead1)
    run.step(second, Action.SUBMIT_QA, cast.rev2, scores=scores["close"])
    run.step(second, Action.FINALIZE, cast.lead1)

    third = submit("QA disagrees")
    run.step(third, Action.SUBMIT_PRIMARY, cast.rev1, scores=scores["better"])
    run.step(third, Action.SEND_TO_QA, cast.lead1)
    run.refused(third, Action.SUBMIT_QA, cast.rev1, scores=scores["gated"])
    run.step(third, Action.SUBMIT_QA, cast.rev2, scores=scores["gated"])
    run.step(third, Action.ESCALATE, cast.lead1)
    run.step(third, Action.ADJUDICATE, cast.lead2, scores=scores["settled"])
    return [first, second, third]


def run_walkthrough(
    url: str, rubric: Rubric, scores: Mapping[str, Mapping[str, int]], echo: Echo
) -> ChainReport:
    """Run the scripted scenario on the database at ``url`` and return the chain report.

    ``scores`` needs the keys ``good``, ``better``, ``close``, ``gated`` and ``settled``,
    each a complete score map for ``rubric``.
    """
    migrate.upgrade(url)
    engine = make_engine(url)
    try:
        sessions = make_session_factory(engine)
        with sessions() as session:
            if session.scalar(select(func.count()).select_from(User)):
                msg = "the database already has users; point --url at a new database"
                raise WalkthroughError(msg)
        service = PipelineService(sessions, SteppingClock(WALKTHROUGH_START))
        cast = _cast(service, rubric, echo)
        submissions = _scenarios(_Runner(service, cast.names, echo), cast, scores)

        echo("\nsubmission  status     final score  verdict  decided by")
        for sid in submissions:
            item = service.get_submission(sid)
            verdict = "PASS" if item.final_passed else "FAIL"
            echo(
                f"{sid:>10}  {item.status:<9}  {item.final_score or 0.0:>11.4f}  "
                f"{verdict:<7}  {item.final_stage} review"
            )
        with sessions() as session:
            report = verify_audit_chain(session)
        echo(f"\naudit chain {report.summary()}")
        return report
    finally:
        engine.dispose()
