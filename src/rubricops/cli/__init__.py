"""Command-line entry point: ``rubricops``."""

from __future__ import annotations

from typing import Annotated

import typer

from rubricops import __version__
from rubricops.cli.agreement import agreement
from rubricops.cli.db import db_app
from rubricops.cli.pipeline import audit_app, pipeline_app
from rubricops.cli.queue import queue_app
from rubricops.cli.rubric import rubric_app
from rubricops.settings import get_settings

app = typer.Typer(
    name="rubricops",
    help="Expert review and rubric-grading operations.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(rubric_app, name="rubric")
app.add_typer(db_app, name="db")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(audit_app, name="audit")
app.add_typer(queue_app, name="queue")
app.command("agreement")(agreement)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rubricops {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
) -> None:
    """RubricOps command-line interface."""


@app.command()
def config() -> None:
    """Print the effective configuration (secrets masked)."""
    settings = get_settings()
    for key, value in settings.model_dump().items():
        typer.echo(f"{key}={value}")
