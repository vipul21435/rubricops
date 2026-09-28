from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from rubricops.domain.clock import FrozenClock
from rubricops.domain.pipeline import Stage
from rubricops.domain.sla import OpenAssignment, SlaPolicy, find_overdue, format_lateness

T0 = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


def _open(sid: int, due_hours: float, reviewer: int = 1, **kwargs: object) -> OpenAssignment:
    return OpenAssignment(sid, reviewer, T0, T0 + timedelta(hours=due_hours), **kwargs)  # type: ignore[arg-type]


def test_due_at_uses_the_default_or_a_per_rubric_override() -> None:
    policy = SlaPolicy.from_hours(24, {"code-explanation": 4})
    assert policy.due_at(T0) == T0 + timedelta(hours=24)
    assert policy.due_at(T0, "action-items") == T0 + timedelta(hours=24)
    assert policy.due_at(T0, "code-explanation") == T0 + timedelta(hours=4)
    ist = timezone(timedelta(hours=5, minutes=30))
    assert policy.due_at(T0.astimezone(ist)).tzinfo is UTC


def test_sla_policy_validates_and_is_immutable() -> None:
    with pytest.raises(ValueError, match="default SLA must be positive"):
        SlaPolicy(timedelta(0))
    with pytest.raises(ValueError, match="SLA for x must be positive"):
        SlaPolicy.from_hours(1, {"x": -1})
    overrides = {"a": timedelta(hours=1)}
    policy = SlaPolicy(timedelta(hours=2), overrides)
    overrides["a"] = timedelta(hours=9)
    assert policy.sla_for("a") == timedelta(hours=1)
    with pytest.raises(TypeError):
        policy.overrides["b"] = timedelta(hours=1)  # type: ignore[index]


def test_naive_and_inverted_times_are_rejected() -> None:
    with pytest.raises(ValueError, match="naive"):
        OpenAssignment(1, 1, datetime(2026, 9, 1), T0)  # noqa: DTZ001 - the point of the test
    with pytest.raises(ValueError, match="due_at is before assigned_at"):
        OpenAssignment(1, 1, T0, T0 - timedelta(seconds=1))


def test_due_exactly_now_is_not_late_one_second_after_is() -> None:
    clock = FrozenClock(T0 + timedelta(hours=24))
    item = _open(1, 24)
    assert find_overdue([item], clock) == []
    clock.advance(timedelta(seconds=1))
    [late] = find_overdue([item], clock)
    assert late.lateness == timedelta(seconds=1)


def test_completed_and_released_assignments_are_never_overdue() -> None:
    clock = FrozenClock(T0 + timedelta(days=10))
    done = _open(1, 1, completed_at=T0 + timedelta(minutes=30))
    released = _open(2, 1, released_at=T0 + timedelta(minutes=30))
    assert not done.is_open
    assert not released.is_open
    assert find_overdue([done, released], clock) == []


def test_escalation_list_is_sorted_by_lateness_then_ids() -> None:
    clock = FrozenClock(T0 + timedelta(hours=30))
    items = [
        _open(5, 24),  # 6h late
        _open(3, 4, stage=Stage.QA),  # 26h late
        _open(4, 24, reviewer=2),  # 6h late, tie broken by submission id
        _open(9, 48),  # not due yet
    ]
    report = find_overdue(items, clock)
    assert [(o.assignment.submission_id, o.lateness) for o in report] == [
        (3, timedelta(hours=26)),
        (4, timedelta(hours=6)),
        (5, timedelta(hours=6)),
    ]


def test_moving_the_clock_changes_the_report() -> None:
    clock = FrozenClock(T0)
    items = [_open(1, 1), _open(2, 2)]
    assert find_overdue(items, clock) == []
    clock.advance(timedelta(hours=1, minutes=30))
    assert [o.assignment.submission_id for o in find_overdue(items, clock)] == [1]
    clock.advance(timedelta(hours=1))
    assert [o.assignment.submission_id for o in find_overdue(items, clock)] == [1, 2]


@pytest.mark.parametrize(
    ("delta", "text"),
    [
        (timedelta(seconds=59), "00h 00m"),
        (timedelta(hours=6, minutes=5), "06h 05m"),
        (timedelta(days=1, hours=2, minutes=5, seconds=30), "1d 02h 05m"),
    ],
)
def test_format_lateness(delta: timedelta, text: str) -> None:
    assert format_lateness(delta) == text
