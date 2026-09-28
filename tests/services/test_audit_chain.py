"""The audit log is tamper-evident: every kind of edit is caught at the right link."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rubricops.db.models import GENESIS_HASH
from rubricops.domain.pipeline import Action
from rubricops.services.audit import (
    ChainReport,
    _canonical_of,
    canonical_event,
    link_hash,
    verify_audit_chain,
)
from tests.services.conftest import World
from tests.services.test_pipeline_service import HIGH, _to_reviewed


@pytest.fixture
def chained(world: World) -> World:
    """A finalized review: 6 setup events, then 1 submit and 4 transitions (11 events)."""
    sid = _to_reviewed(world, HIGH)
    version = world.service.get_submission(sid).version
    world.service.apply(sid, Action.FINALIZE, actor_id=world.lead1, expected_version=version)
    return world


def _unguard(session: Session) -> None:
    """What an attacker with DDL rights would do first."""
    session.execute(text("DROP TRIGGER audit_events_no_update"))
    session.execute(text("DROP TRIGGER audit_events_no_delete"))


def test_intact_chain_verifies(chained: World) -> None:
    with chained.sessions() as session:
        report = verify_audit_chain(session)
    assert report.ok
    assert report.checked == report.head_seq == 11
    assert report.summary() == f"ok: 11 events, head seq 11 hash {report.head_hash[:12]}"


def test_empty_log_verifies() -> None:
    from rubricops.db.engine import make_engine, make_session_factory  # noqa: PLC0415
    from rubricops.db.models import Base  # noqa: PLC0415

    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    with make_session_factory(engine)() as session:
        assert verify_audit_chain(session) == ChainReport(
            ok=True, checked=0, head_seq=0, head_hash=GENESIS_HASH
        )


def test_triggers_stop_the_naive_edit(chained: World) -> None:
    with chained.sessions() as session, pytest.raises(IntegrityError, match="append-only"):
        session.execute(text("UPDATE audit_events SET data = '{}' WHERE seq = 8"))


def test_edited_payload_breaks_its_own_link(chained: World) -> None:
    with chained.sessions() as session:
        _unguard(session)
        row = session.execute(text("SELECT data FROM audit_events WHERE seq = 10")).scalar_one()
        data = json.loads(row)
        data["score"] = 0.99
        session.execute(
            text("UPDATE audit_events SET data = :d WHERE seq = 10"), {"d": json.dumps(data)}
        )
        report = verify_audit_chain(session)
    assert not report.ok
    assert report.broken_at == 10
    assert report.reason == "hash does not match the event's content"
    assert report.checked == 9
    assert report.summary() == (
        "BROKEN at seq 10: hash does not match the event's content (9 events checked)"
    )


def test_rehashing_the_edited_event_breaks_the_next_link(chained: World) -> None:
    with chained.sessions() as session:
        _unguard(session)
        session.execute(text("UPDATE audit_events SET actor_id = 4 WHERE seq = 9"))
        session.expire_all()
        from rubricops.db.models import AuditEvent  # noqa: PLC0415

        forged = session.get(AuditEvent, 9)
        assert forged is not None
        new_hash = link_hash(forged.prev_hash, _canonical_of(forged))
        session.execute(text("UPDATE audit_events SET hash = :h WHERE seq = 9"), {"h": new_hash})
        report = verify_audit_chain(session)
    assert report.broken_at == 10
    assert report.reason == "prev_hash does not match the previous event's hash"


def test_deleted_event_is_a_gap(chained: World) -> None:
    with chained.sessions() as session:
        _unguard(session)
        session.execute(text("DELETE FROM audit_events WHERE seq = 5"))
        report = verify_audit_chain(session)
    assert report.broken_at == 5
    assert report.reason == "expected seq 5, found 6 (events missing)"


def test_truncation_is_caught_only_with_an_anchor(chained: World) -> None:
    with chained.sessions() as session:
        anchor = verify_audit_chain(session)
        _unguard(session)
        session.execute(text("DELETE FROM audit_events WHERE seq >= 10"))
        # Internally the shorter chain is still consistent ...
        assert verify_audit_chain(session).ok
        # ... but not with the head recorded earlier.
        report = verify_audit_chain(session, anchor=(anchor.head_seq, anchor.head_hash))
    assert not report.ok
    assert report.broken_at == 11
    assert report.reason == "anchored event seq 11 is missing (log truncated)"


def test_anchor_hash_mismatch(chained: World) -> None:
    with chained.sessions() as session:
        good = verify_audit_chain(session, anchor=(3, _hash_of(session, 3)))
        assert good.ok
        bad = verify_audit_chain(session, anchor=(3, "f" * 64))
    assert bad.broken_at == 3
    assert bad.reason == "hash differs from the anchored hash"


def _hash_of(session: Session, seq: int) -> str:
    value = session.execute(
        text("SELECT hash FROM audit_events WHERE seq = :s"), {"s": seq}
    ).scalar_one()
    assert isinstance(value, str)
    return value


def test_canonical_form_is_stable_and_ascii() -> None:
    at = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    first = canonical_event(
        seq=1,
        occurred_at=at,
        actor_id=None,
        action="submit",
        entity="submission",
        entity_id=1,
        from_status=None,
        to_status="queued",
        data={"b": 1, "a": "café"},
    )
    assert first.isascii()
    assert first.index('"a"') < first.index('"b"')
    assert '"occurred_at":"2026-09-01T09:00:00.000000+00:00"' in first
    assert link_hash(GENESIS_HASH, first) == link_hash(GENESIS_HASH, first)
    assert len(link_hash(GENESIS_HASH, first)) == 64
