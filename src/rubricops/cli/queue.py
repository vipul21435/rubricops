"""``rubricops queue assign|overdue|sample`` over a queue scenario file.

Exit codes: 0 success (items nobody could take are reported, not failures), 1 an
invalid scenario, 2 usage error.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from rubricops.domain.clock import Clock, FrozenClock, SystemClock
from rubricops.domain.queue import POLICY_NAMES, RoundRobin, assign_batch, make_policy
from rubricops.domain.sampling import QaSampler, SamplingRules
from rubricops.domain.sla import find_overdue, format_lateness
from rubricops.loaders import DocumentError
from rubricops.scenario import Scenario, load_scenario
from rubricops.settings import get_settings

queue_app = typer.Typer(
    help="Assign queued items, list overdue reviews and sample reviews for QA.",
    no_args_is_help=True,
)

ScenarioArg = Annotated[
    Path, typer.Argument(help="Queue scenario (YAML or JSON).", show_default=False)
]


def _load(path: Path) -> Scenario:
    try:
        return load_scenario(path)
    except DocumentError as exc:
        typer.echo(f"error: {exc.path}", err=True)
        for issue in exc.issues:
            typer.echo(f"  {issue}", err=True)
        raise typer.Exit(code=1) from exc


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M")


@queue_app.command("assign")
def assign(
    scenario_path: ScenarioArg,
    policy_name: Annotated[
        str, typer.Option("--policy", help=f"One of: {', '.join(POLICY_NAMES)}.")
    ] = "load-balanced",
    cursor: Annotated[
        int | None,
        typer.Option("--cursor", help="round-robin only: the last reviewer id assigned."),
    ] = None,
) -> None:
    """Assign every queued item in order with one policy, and print the new loads."""
    if policy_name not in POLICY_NAMES:
        raise typer.BadParameter(
            f"expected one of {', '.join(POLICY_NAMES)}", param_hint="--policy"
        )
    if cursor is not None and policy_name != RoundRobin.name:
        raise typer.BadParameter("only round-robin takes a cursor", param_hint="--cursor")
    scenario = _load(scenario_path)
    policy = make_policy(policy_name, cursor=cursor)
    handles = scenario.handles()
    result = assign_batch(policy, scenario.queue_items(), scenario.reviewer_pool())
    typer.echo(f"policy {policy_name}: {result.assigned} of {len(result.decisions)} assigned")
    typer.echo(f"  {'item':>6}  {'stage':<8} {'tags':<18} assigned to")
    for decision in result.decisions:
        item = decision.item
        tags = ",".join(sorted(item.skill_tags)) or "-"
        if decision.reviewer is not None:
            who = f"{decision.reviewer.handle} ({decision.reviewer.id})"
        else:
            assert decision.refusal is not None  # noqa: S101 - assign_batch pairs them
            reasons = ", ".join(
                f"{handles[rid]}={why.value}"
                for rid, why in sorted(decision.refusal.excluded.items())
            )
            who = f"UNASSIGNED ({reasons or 'no reviewers'})"
        typer.echo(f"  {item.submission_id:>6}  {item.stage.value:<8} {tags:<18} {who}")
    loads = ", ".join(f"{handles[rid]}={load}" for rid, load in result.loads.items())
    typer.echo(f"open loads after: {loads}")
    if isinstance(policy, RoundRobin):
        typer.echo(f"cursor: {policy.cursor}")


@queue_app.command("overdue")
def overdue(
    scenario_path: ScenarioArg,
    now: Annotated[
        datetime | None,
        typer.Option(
            "--now",
            formats=["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z"],
            help="Check as of this time, e.g. 2026-09-29T09:00+0000 (default: the "
            "scenario's now, else the system clock).",
        ),
    ] = None,
    sla_hours: Annotated[
        float | None,
        typer.Option(
            "--sla-hours",
            min=0.01,
            help="Default SLA (default: the scenario's, else RUBRICOPS_REVIEW_SLA_HOURS).",
        ),
    ] = None,
) -> None:
    """List open assignments past their due time, most late first."""
    scenario = _load(scenario_path)
    at = now or scenario.now
    clock: Clock = FrozenClock(at) if at is not None else SystemClock()
    policy = scenario.sla_policy(get_settings().review_sla_hours, override_hours=sla_hours)
    assignments = scenario.open_assignments(policy)
    late = find_overdue(assignments, clock)
    handles = scenario.handles()
    open_count = sum(a.is_open for a in assignments)
    typer.echo(
        f"as of {_stamp(clock.now())} UTC: {len(late)} of {open_count} open assignments "
        f"overdue (default SLA {policy.default.total_seconds() / 3600:g}h)"
    )
    if not late:
        return
    typer.echo(f"  {'item':>6}  {'stage':<8} {'reviewer':<12} {'due (UTC)':<17} late by")
    for entry in late:
        a = entry.assignment
        typer.echo(
            f"  {a.submission_id:>6}  {a.stage.value:<8} {handles[a.reviewer_id]:<12} "
            f"{_stamp(a.due_at):<17} {format_lateness(entry.lateness)}"
        )


@queue_app.command("sample")
def sample(
    scenario_path: ScenarioArg,
    *,
    rate: Annotated[
        float | None,
        typer.Option(
            "--rate", min=0.0, max=1.0, help="Random QA rate (default: RUBRICOPS_QA_SAMPLE_RATE)."
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed (default: RUBRICOPS_RANDOM_SEED).")
    ] = None,
    min_reviews: Annotated[
        int, typer.Option("--min-reviews", min=0, help="Reviewers below this are new.")
    ] = 20,
    margin: Annotated[
        float,
        typer.Option("--margin", min=0.0, max=1.0, help="Near-threshold margin."),
    ] = 0.05,
    max_spread: Annotated[
        float,
        typer.Option("--max-spread", min=0.0, max=1.0, help="Widest acceptable score spread."),
    ] = 0.25,
) -> None:
    """Decide which primary reviews go to QA, with the reasons for each decision."""
    scenario = _load(scenario_path)
    settings = get_settings()
    rules = SamplingRules(
        rate=settings.qa_sample_rate if rate is None else rate,
        new_reviewer_min_reviews=min_reviews,
        threshold_margin=margin,
        max_score_spread=max_spread,
    )
    sampler = QaSampler(rules, settings.random_seed if seed is None else seed)
    decisions = sampler.decide_all(scenario.sample_candidates())
    handles = scenario.handles()
    reviewer_of = {r.submission: r.reviewer for r in scenario.reviews}
    sampled = sum(d.sampled for d in decisions)
    typer.echo(
        f"seed {sampler.seed}, rate {rules.rate:g}: {sampled} of {len(decisions)} sent to QA"
    )
    for d in decisions:
        verdict = "QA      " if d.sampled else "finalize"
        who = handles[reviewer_of[d.submission_id]]
        codes = "+".join(r.value for r in d.reasons) or "none"
        why = "; ".join(d.details) if d.details else f"draw {d.draw:.4f} >= rate {d.rate:g}"
        typer.echo(f"  {d.submission_id:>6}  {verdict}  {who:<8} {codes:<31} {why}")
