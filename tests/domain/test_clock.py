from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from rubricops.domain.clock import FrozenClock, SteppingClock, SystemClock, ensure_utc

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def test_system_clock_is_aware_utc() -> None:
    now = SystemClock().now()
    assert now.tzinfo is UTC


def test_frozen_clock_stands_still_until_moved() -> None:
    clock = FrozenClock(T0)
    assert clock.now() == clock.now() == T0
    assert clock.advance(timedelta(hours=3)) == T0 + timedelta(hours=3)
    clock.set(T0 - timedelta(days=1))
    assert clock.now() == T0 - timedelta(days=1)


def test_frozen_clock_converts_offsets_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    clock = FrozenClock(datetime(2026, 9, 1, 17, 30, tzinfo=ist))
    assert clock.now() == T0
    assert clock.now().tzinfo is UTC


def test_frozen_clock_refuses_to_go_backwards_via_advance() -> None:
    with pytest.raises(ValueError, match="negative"):
        FrozenClock(T0).advance(timedelta(seconds=-1))


def test_naive_datetimes_are_rejected() -> None:
    naive = datetime(2026, 9, 1, 12, 0)  # noqa: DTZ001 - naive on purpose
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_utc(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        FrozenClock(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        FrozenClock(T0).set(naive)


def test_stepping_clock_moves_on_every_read() -> None:
    start = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    clock = SteppingClock(start, timedelta(seconds=30))
    assert [clock.now(), clock.now()] == [start, start + timedelta(seconds=30)]
    with pytest.raises(ValueError, match="positive"):
        SteppingClock(start, timedelta(0))
