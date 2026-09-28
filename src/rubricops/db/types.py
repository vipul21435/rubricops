"""Column types that behave the same on SQLite and Postgres."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Dialect
from sqlalchemy.types import TypeDecorator

from rubricops.domain.clock import ensure_utc


class UTCDateTime(TypeDecorator[datetime]):
    """A timezone-aware UTC datetime on every backend.

    SQLite has no timezone support and hands back naive values, so this type stores
    UTC and re-attaches ``UTC`` on the way out. Naive datetimes are rejected on the
    way in rather than guessed, the same rule as :func:`ensure_utc`.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        aware = ensure_utc(value)
        return aware.replace(tzinfo=None) if dialect.name == "sqlite" else aware

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
