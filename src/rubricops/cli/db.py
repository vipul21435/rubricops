"""``rubricops db``: apply and inspect Alembic migrations.

``--url`` defaults to ``RUBRICOPS_DATABASE_URL``. Passwords in the URL are masked in
everything printed.
"""

from __future__ import annotations

from typing import Annotated

import typer
from sqlalchemy.engine import make_url

from rubricops.db import migrate
from rubricops.settings import get_settings

db_app = typer.Typer(help="Apply and inspect database migrations.", no_args_is_help=True)

UrlOption = Annotated[
    str | None,
    typer.Option("--url", help="SQLAlchemy URL (default: RUBRICOPS_DATABASE_URL)."),
]


def _resolve(url: str | None) -> str:
    return url or get_settings().database_url


def _masked(url: str) -> str:
    return make_url(url).render_as_string(hide_password=True)


@db_app.command("upgrade")
def upgrade(
    url: UrlOption = None,
    revision: Annotated[str, typer.Argument(help="Target revision.")] = "head",
    sql: Annotated[bool, typer.Option("--sql", help="Print the SQL instead of running it.")] = (
        False
    ),
) -> None:
    """Migrate the database up to REVISION (default: head)."""
    target = _resolve(url)
    if sql:
        migrate.upgrade_sql(target, revision)
        return
    migrate.upgrade(target, revision)
    typer.echo(f"{_masked(target)} is at revision {migrate.current_revision(target)}")


@db_app.command("downgrade")
def downgrade(
    revision: Annotated[str, typer.Argument(help="Target revision, for example base.")],
    url: UrlOption = None,
) -> None:
    """Migrate the database down to REVISION."""
    target = _resolve(url)
    migrate.downgrade(target, revision)
    current = migrate.current_revision(target) or "base (empty)"
    typer.echo(f"{_masked(target)} is at revision {current}")


@db_app.command("current")
def current(url: UrlOption = None) -> None:
    """Print the revision the database is at."""
    target = _resolve(url)
    typer.echo(f"{_masked(target)} is at revision {migrate.current_revision(target) or 'none'}")
