"""Alembic, driven from code: ``upgrade``, ``downgrade`` and ``current`` for a URL.

The migration scripts ship inside the package (``rubricops/db/migrations``), so an
installed wheel or the Docker image can migrate a database with no ``alembic.ini``.
The URL comes from the caller, falling back to ``RUBRICOPS_DATABASE_URL``.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext

from rubricops.db.engine import make_engine, missing_sqlite_file
from rubricops.settings import get_settings

MIGRATIONS_DIR = Path(__file__).with_name("migrations")


def alembic_config(url: str | None = None) -> Config:
    """An Alembic config pointing at the packaged migrations and ``url``."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    # configparser interpolation would read a percent-encoded password as a reference.
    target = url or get_settings().database_url
    config.set_main_option("sqlalchemy.url", target.replace("%", "%%"))
    return config


def upgrade(url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)


def upgrade_sql(url: str | None = None, revision: str = "head") -> None:
    """Print the upgrade as SQL instead of running it (Alembic offline mode)."""
    command.upgrade(alembic_config(url), revision, sql=True)


def current_revision(url: str | None = None) -> str | None:
    """The revision the database at ``url`` is at, or ``None`` if it is unversioned."""
    target = url or get_settings().database_url
    if missing_sqlite_file(target) is not None:
        return None  # do not create the file just to report that it is empty
    engine = make_engine(target)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()
