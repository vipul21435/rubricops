"""Injectable time source.

Domain code never calls ``datetime.now()`` directly; it asks a :class:`Clock`. The
application wires in :class:`SystemClock`, and tests use :class:`FrozenClock` so
SLA and history logic can be checked at exact instants.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """Anything that can tell the current time as a timezone-aware UTC datetime."""

    def now(self) -> datetime: ...


def ensure_utc(moment: datetime) -> datetime:
    """Return ``moment`` converted to UTC; naive datetimes are rejected, not guessed."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        msg = f"expected a timezone-aware datetime, got naive {moment.isoformat()}"
        raise ValueError(msg)
    return moment.astimezone(UTC)


class SystemClock:
    """The real wall clock, in UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """A clock that only moves when told to."""

    def __init__(self, at: datetime) -> None:
        self._now = ensure_utc(at)

    def now(self) -> datetime:
        return self._now

    def set(self, at: datetime) -> None:
        """Jump to ``at`` (forwards or backwards)."""
        self._now = ensure_utc(at)

    def advance(self, delta: timedelta) -> datetime:
        """Move forward by ``delta`` and return the new time."""
        if delta < timedelta(0):
            msg = f"cannot advance by a negative amount ({delta})"
            raise ValueError(msg)
        self._now += delta
        return self._now
