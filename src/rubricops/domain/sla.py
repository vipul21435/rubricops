"""Review SLAs: when an assignment is due, and which open ones are late.

:class:`SlaPolicy` holds a default turnaround (``RUBRICOPS_REVIEW_SLA_HOURS``) and
optional per-rubric overrides, and turns an assignment time into a ``due_at``.
:func:`find_overdue` asks an injected :class:`~rubricops.domain.clock.Clock` for the
current time, keeps the open assignments whose ``due_at`` has passed, and returns
them as an escalation list sorted by lateness (most late first, then by submission
and reviewer id so the order is total). An assignment due exactly now is not yet
late. Completed or released assignments are never overdue.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import MappingProxyType

from rubricops.domain.clock import Clock, ensure_utc
from rubricops.domain.pipeline import Stage


def _positive(what: str, value: timedelta) -> timedelta:
    if value <= timedelta(0):
        msg = f"{what} must be positive, got {value}"
        raise ValueError(msg)
    return value


@dataclass(frozen=True, slots=True)
class SlaPolicy:
    """A default turnaround plus per-rubric overrides keyed by rubric id."""

    default: timedelta
    overrides: Mapping[str, timedelta] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _positive("default SLA", self.default)
        for rubric_id, sla in self.overrides.items():
            _positive(f"SLA for {rubric_id}", sla)
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    @classmethod
    def from_hours(
        cls, default_hours: float, overrides: Mapping[str, float] | None = None
    ) -> SlaPolicy:
        return cls(
            timedelta(hours=default_hours),
            {rubric_id: timedelta(hours=h) for rubric_id, h in (overrides or {}).items()},
        )

    def sla_for(self, rubric_id: str | None) -> timedelta:
        if rubric_id is None:
            return self.default
        return self.overrides.get(rubric_id, self.default)

    def due_at(self, assigned_at: datetime, rubric_id: str | None = None) -> datetime:
        return ensure_utc(assigned_at) + self.sla_for(rubric_id)


@dataclass(frozen=True, slots=True)
class OpenAssignment:
    """An assignment as the SLA check sees it (mirrors the ``assignments`` table)."""

    submission_id: int
    reviewer_id: int
    assigned_at: datetime
    due_at: datetime
    stage: Stage = Stage.PRIMARY
    completed_at: datetime | None = None
    released_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("assigned_at", "due_at", "completed_at", "released_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, ensure_utc(value))
        if self.due_at < self.assigned_at:
            msg = f"submission {self.submission_id}: due_at is before assigned_at"
            raise ValueError(msg)

    @property
    def is_open(self) -> bool:
        return self.completed_at is None and self.released_at is None


@dataclass(frozen=True, slots=True)
class Overdue:
    assignment: OpenAssignment
    lateness: timedelta


def find_overdue(assignments: Iterable[OpenAssignment], clock: Clock) -> list[Overdue]:
    """Open assignments past their ``due_at`` at ``clock.now()``, most late first."""
    now = ensure_utc(clock.now())
    late = [Overdue(a, now - a.due_at) for a in assignments if a.is_open and now > a.due_at]
    late.sort(key=lambda o: (-o.lateness, o.assignment.submission_id, o.assignment.reviewer_id))
    return late


def format_lateness(delta: timedelta) -> str:
    """``1d 02h 05m`` style, rounded down to the minute."""
    minutes = int(delta.total_seconds() // 60)
    days, rest = divmod(minutes, 24 * 60)
    hours, mins = divmod(rest, 60)
    return f"{days}d {hours:02d}h {mins:02d}m" if days else f"{hours:02d}h {mins:02d}m"
