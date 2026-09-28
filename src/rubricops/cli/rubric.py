"""``rubricops rubric``: validate, diff and score rubric files.

Exit codes: 0 success, 1 invalid input (a rubric or score map that fails
validation, or a scoring change under ``--fail-on-scoring-change``), 2 usage error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rubricops.domain.diff import diff_rubrics
from rubricops.domain.rubric import Rubric
from rubricops.domain.scoring import InvalidScoresError, ScoreResult, score_review
from rubricops.domain.versioning import content_hash
from rubricops.loaders import DocumentError, load_rubric, load_scores

rubric_app = typer.Typer(
    help="Validate, diff and score rubric files (YAML or JSON).",
    no_args_is_help=True,
)

JsonFlag = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]


def _fail(lines: list[str]) -> typer.Exit:
    for line in lines:
        typer.echo(line, err=True)
    return typer.Exit(code=1)


def _error_lines(exc: DocumentError) -> list[str]:
    return [f"error: {exc.path}", *(f"  {issue}" for issue in exc.issues)]


def _load_or_exit(path: Path) -> Rubric:
    try:
        return load_rubric(path)
    except DocumentError as exc:
        raise _fail(_error_lines(exc)) from exc


@rubric_app.command("validate")
def validate(
    paths: Annotated[list[Path], typer.Argument(help="Rubric files to check.")],
) -> None:
    """Check rubric files and print each one's id, size and content hash."""
    failed = False
    for path in paths:
        try:
            rubric = load_rubric(path)
        except DocumentError as exc:
            failed = True
            for line in _error_lines(exc):
                typer.echo(line, err=True)
            continue
        typer.echo(
            f"ok    {path}  id={rubric.id} criteria={len(rubric.criteria)} "
            f"pass_threshold={rubric.pass_threshold:g} sha256={content_hash(rubric)[:12]}"
        )
    if failed:
        raise typer.Exit(code=1)


@rubric_app.command("diff")
def diff(
    old: Annotated[Path, typer.Argument(help="The earlier rubric file.")],
    new: Annotated[Path, typer.Argument(help="The later rubric file.")],
    as_json: JsonFlag = False,
    fail_on_scoring_change: Annotated[
        bool,
        typer.Option(
            "--fail-on-scoring-change",
            help="Exit 1 when the change can alter a score or verdict (for CI gates).",
        ),
    ] = False,
) -> None:
    """Show what changed between two versions of a rubric."""
    before, after = _load_or_exit(old), _load_or_exit(new)
    result = diff_rubrics(before, after)
    if as_json:
        payload = {
            "old": {"id": before.id, "sha256": content_hash(before)},
            "new": {"id": after.id, "sha256": content_hash(after)},
            "affects_scoring": result.affects_scoring,
            "changes": result.model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(
            f"{before.id} ({content_hash(before)[:12]}) -> {after.id} ({content_hash(after)[:12]})"
        )
        for line in result.summary_lines():
            typer.echo(f"  {line}")
        typer.echo(f"scoring affected: {'yes' if result.affects_scoring else 'no'}")
    if fail_on_scoring_change and result.affects_scoring:
        raise typer.Exit(code=1)


def _parse_score_option(raw: str) -> tuple[str, object]:
    key, sep, value = raw.partition("=")
    if not sep or not key.strip():
        msg = f"expected CRITERION=SCORE, got {raw!r}"
        raise typer.BadParameter(msg, param_hint="--score")
    text = value.strip()
    try:
        return key.strip(), int(text)
    except ValueError:
        return key.strip(), text


def _render(result: ScoreResult) -> list[str]:
    width = max(len("criterion"), *(len(c.criterion_id) for c in result.criteria))
    lines = [
        f"rubric {result.rubric_id}  sha256 {result.content_hash[:12]}",
        f"{'criterion':<{width}}  score  scale   weight  contribution",
    ]
    lines += [
        f"{c.criterion_id:<{width}}  {c.score:>5}  {c.scale!s:<6}  {c.weight:>6.3f}  "
        f"{c.contribution:>12.4f}"
        for c in result.criteria
    ]
    lines += [
        f"gate failed: {g.criterion_id} scored {g.score}, needs at least {g.minimum}"
        for g in result.gating_failures
    ]
    verdict = "PASS" if result.passed else "FAIL"
    lines.append(f"score {result.score:.4f}  threshold {result.pass_threshold:g}  -> {verdict}")
    return lines


@rubric_app.command("score")
def score(
    rubric_path: Annotated[Path, typer.Argument(metavar="RUBRIC", help="Rubric file.")],
    scores_file: Annotated[
        Path | None,
        typer.Option("--scores", "-f", help="YAML or JSON map of criterion id to score."),
    ] = None,
    score_options: Annotated[
        list[str] | None,
        typer.Option(
            "--score", "-s", metavar="CRITERION=SCORE", help="One score; repeat per criterion."
        ),
    ] = None,
    as_json: JsonFlag = False,
) -> None:
    """Score one review against a rubric and print the breakdown and verdict."""
    rubric = _load_or_exit(rubric_path)
    scores: dict[str, object] = {}
    if scores_file is not None:
        try:
            scores.update(load_scores(scores_file))
        except DocumentError as exc:
            raise _fail(_error_lines(exc)) from exc
    scores.update(_parse_score_option(raw) for raw in score_options or [])
    if not scores:
        msg = "give scores with --scores FILE or --score CRITERION=SCORE"
        raise typer.BadParameter(msg)
    try:
        result = score_review(rubric, scores)
    except InvalidScoresError as exc:
        lines = [f"error: scores do not fit rubric {rubric.id}", *(f"  {i}" for i in exc.issues)]
        raise _fail(lines) from exc
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        for line in _render(result):
            typer.echo(line)
