"""``rubricops pipeline walkthrough`` and ``rubricops audit verify``.

Exit codes: 0 success, 1 a broken audit chain, a non-empty walkthrough database or
invalid input files, 2 usage error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter, ValidationError

from rubricops.cli.db import UrlOption, resolve_url
from rubricops.db.engine import make_engine, make_session_factory
from rubricops.domain.scoring import InvalidScoresError, validate_scores
from rubricops.loaders import DocumentError, format_validation_error, load_document, load_rubric
from rubricops.services.audit import verify_audit_chain
from rubricops.services.walkthrough import WalkthroughError, run_walkthrough

pipeline_app = typer.Typer(help="Run the review pipeline.", no_args_is_help=True)
audit_app = typer.Typer(help="Check the tamper-evident audit log.", no_args_is_help=True)

SCORE_KEYS = ("good", "better", "close", "gated", "settled")
_ScoreMaps = TypeAdapter(dict[str, dict[str, int]])


@pipeline_app.command("walkthrough")
def walkthrough(
    url: Annotated[str, typer.Option("--url", help="SQLAlchemy URL of a new, empty database.")],
    rubric_path: Annotated[Path, typer.Option("--rubric", help="Rubric file.")] = Path(
        "examples/rubrics/code-explanation.yaml"
    ),
    scores_path: Annotated[
        Path, typer.Option("--scores", help=f"Score maps named {', '.join(SCORE_KEYS)}.")
    ] = Path("examples/reviews/walkthrough-scores.yaml"),
) -> None:
    """Take three submissions through primary review, QA and adjudication, then verify."""
    try:
        rubric = load_rubric(rubric_path)
        maps = _ScoreMaps.validate_python(load_document(scores_path))
    except DocumentError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValidationError as exc:
        for line in [f"error: {scores_path}", *format_validation_error(exc)]:
            typer.echo(line, err=True)
        raise typer.Exit(code=1) from exc
    missing = [key for key in SCORE_KEYS if key not in maps]
    if missing:
        typer.echo(f"error: {scores_path} is missing score maps {missing}", err=True)
        raise typer.Exit(code=1)
    try:
        for key in SCORE_KEYS:
            validate_scores(rubric, maps[key])
    except InvalidScoresError as exc:
        typer.echo(f"error: {scores_path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        report = run_walkthrough(url, rubric, maps, typer.echo)
    except WalkthroughError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not report.ok:  # pragma: no cover - a fresh chain cannot be broken
        raise typer.Exit(code=1)


@audit_app.command("verify")
def verify(
    url: UrlOption = None,
    anchor: Annotated[
        str | None,
        typer.Option("--anchor", help="SEQ:HASH recorded earlier; also catches truncation."),
    ] = None,
) -> None:
    """Re-derive every hash in the audit log and report the first broken link."""
    pinned: tuple[int, str] | None = None
    if anchor is not None:
        seq, _, digest = anchor.partition(":")
        if not seq.isdigit() or len(digest) != 64:
            raise typer.BadParameter(
                "expected SEQ:HASH with a 64-character hash", param_hint="--anchor"
            )
        pinned = (int(seq), digest)
    engine = make_engine(resolve_url(url))
    try:
        with make_session_factory(engine)() as session:
            report = verify_audit_chain(session, anchor=pinned)
    finally:
        engine.dispose()
    typer.echo(report.summary())
    if report.ok:
        typer.echo(f"anchor: {report.head_seq}:{report.head_hash}")
    else:
        raise typer.Exit(code=1)
