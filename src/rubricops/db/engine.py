"""Engine and session factories built from a SQLAlchemy URL.

SQLite connections get ``PRAGMA foreign_keys=ON`` (off by default in SQLite, which
would let a review point at a submission that does not exist) and, for file
databases, ``journal_mode=WAL`` so readers do not block the single writer. Any other
SQLAlchemy URL (for example ``postgresql+psycopg://...``) is passed through as is.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def _is_memory_sqlite(database: str | None) -> bool:
    return database in (None, "", ":memory:") or (database or "").startswith("file::memory:")


def make_engine(url: str, *, echo: bool = False) -> Engine:
    """Create an engine for ``url``; SQLite gets foreign keys, WAL and a busy timeout.

    An in-memory SQLite URL uses a single shared connection (``StaticPool``), so every
    session sees the same database instead of a fresh empty one per connection. The
    parent directory of a SQLite file is created if it is missing.
    """
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return create_engine(url, echo=echo)

    memory = _is_memory_sqlite(parsed.database)
    kwargs: dict[str, Any] = {"echo": echo}
    if memory:
        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    elif parsed.database:
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection: Any, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            if not memory:
                cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """A session factory whose objects stay readable after commit."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """One unit of work: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
