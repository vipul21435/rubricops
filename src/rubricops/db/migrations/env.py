"""Alembic environment: migrate the URL in the config, online or as SQL (offline)."""

from __future__ import annotations

from alembic import context
from sqlalchemy.engine import Connection

from rubricops.db.engine import make_engine
from rubricops.db.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if not url:  # pragma: no cover - alembic_config() always sets it
        msg = "sqlalchemy.url is not set; use rubricops.db.migrate.alembic_config()"
        raise RuntimeError(msg)
    return url


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
    )


def run_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = make_engine(_url())
    try:
        with engine.connect() as connection:
            _configure(connection)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
