"""The tamper-evident audit log: a single sha256 hash chain over every event.

Each event's ``hash`` is ``sha256(prev_hash + canonical_json(event))`` where the
canonical form covers every column except the hashes themselves (``seq``, time,
actor, action, entity, statuses and the data payload) with sorted keys and ASCII
escapes. The first event links to :data:`GENESIS_HASH`.

Appending reads the current head inside the caller's transaction and writes
``seq = head.seq + 1``; the UNIQUE constraints on ``seq``, ``prev_hash`` and ``hash``
turn a concurrent append from the same head into a failed commit rather than a fork.

:func:`verify_audit_chain` re-derives every hash and reports the first link that
does not hold. Editing any field of any event, deleting or reordering events, or
splicing in a forged one breaks the chain at that point. Truncating the newest
events leaves a shorter chain that is still internally valid, which is why the
report includes the head hash: record it elsewhere and pass it back as an
``anchor`` to detect truncation too.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rubricops.db.models import GENESIS_HASH, AuditEvent
from rubricops.domain.clock import Clock, ensure_utc


def canonical_event(
    *,
    seq: int,
    occurred_at: datetime,
    actor_id: int | None,
    action: str,
    entity: str,
    entity_id: int,
    from_status: str | None,
    to_status: str | None,
    data: Mapping[str, Any],
) -> str:
    """The exact text that is hashed for one event."""
    body = {
        "seq": seq,
        "occurred_at": ensure_utc(occurred_at).isoformat(timespec="microseconds"),
        "actor_id": actor_id,
        "action": action,
        "entity": entity,
        "entity_id": entity_id,
        "from_status": from_status,
        "to_status": to_status,
        "data": dict(data),
    }
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def link_hash(prev_hash: str, canonical: str) -> str:
    """``sha256(prev_hash + canonical)`` as lowercase hex."""
    return hashlib.sha256((prev_hash + canonical).encode("ascii")).hexdigest()


def _canonical_of(event: AuditEvent) -> str:
    return canonical_event(
        seq=event.seq,
        occurred_at=event.occurred_at,
        actor_id=event.actor_id,
        action=event.action,
        entity=event.entity,
        entity_id=event.entity_id,
        from_status=event.from_status,
        to_status=event.to_status,
        data=event.data,
    )


def append_event(
    session: Session,
    clock: Clock,
    *,
    actor_id: int | None,
    action: str,
    entity: str,
    entity_id: int,
    from_status: str | None = None,
    to_status: str | None = None,
    data: Mapping[str, Any] | None = None,
) -> AuditEvent:
    """Append one event to the chain in ``session``'s transaction and return it."""
    head = session.scalars(select(AuditEvent).order_by(AuditEvent.seq.desc()).limit(1)).first()
    seq = 1 if head is None else head.seq + 1
    prev_hash = GENESIS_HASH if head is None else head.hash
    payload = dict(data or {})
    occurred_at = ensure_utc(clock.now())
    canonical = canonical_event(
        seq=seq,
        occurred_at=occurred_at,
        actor_id=actor_id,
        action=action,
        entity=entity,
        entity_id=entity_id,
        from_status=from_status,
        to_status=to_status,
        data=payload,
    )
    event = AuditEvent(
        seq=seq,
        occurred_at=occurred_at,
        actor_id=actor_id,
        action=action,
        entity=entity,
        entity_id=entity_id,
        from_status=from_status,
        to_status=to_status,
        data=payload,
        prev_hash=prev_hash,
        hash=link_hash(prev_hash, canonical),
    )
    session.add(event)
    session.flush()
    return event


@dataclass(frozen=True, slots=True)
class ChainReport:
    """The result of walking the chain from the genesis link."""

    ok: bool
    checked: int
    head_seq: int
    head_hash: str
    broken_at: int | None = None
    reason: str | None = None

    def summary(self) -> str:
        if self.ok:
            return f"ok: {self.checked} events, head seq {self.head_seq} hash {self.head_hash[:12]}"
        return f"BROKEN at seq {self.broken_at}: {self.reason} ({self.checked} events checked)"


def verify_audit_chain(session: Session, *, anchor: tuple[int, str] | None = None) -> ChainReport:
    """Re-derive every link and report the first one that does not hold.

    ``anchor`` is a ``(seq, hash)`` pair recorded earlier (for example a previous
    report's head). The chain must still contain that event with that hash, which
    catches truncation of the newest events.
    """
    expected_seq = 1
    prev_hash = GENESIS_HASH
    anchor_seen = anchor is None
    checked = 0
    # populate_existing: verify what is stored, not objects cached in this session.
    rows = select(AuditEvent).order_by(AuditEvent.seq).execution_options(populate_existing=True)
    for event in session.scalars(rows):
        problem: str | None = None
        if event.seq != expected_seq:
            problem = f"expected seq {expected_seq}, found {event.seq} (events missing)"
        elif event.prev_hash != prev_hash:
            problem = "prev_hash does not match the previous event's hash"
        elif link_hash(prev_hash, _canonical_of(event)) != event.hash:
            problem = "hash does not match the event's content"
        elif anchor is not None and event.seq == anchor[0]:
            if event.hash != anchor[1]:
                problem = "hash differs from the anchored hash"
            anchor_seen = True
        if problem is not None:
            return ChainReport(
                ok=False,
                checked=checked,
                head_seq=expected_seq - 1,
                head_hash=prev_hash,
                broken_at=expected_seq,
                reason=problem,
            )
        checked += 1
        expected_seq += 1
        prev_hash = event.hash
    if not anchor_seen:
        assert anchor is not None  # noqa: S101 - anchor_seen starts True when there is none
        return ChainReport(
            ok=False,
            checked=checked,
            head_seq=checked,
            head_hash=prev_hash,
            broken_at=anchor[0],
            reason=f"anchored event seq {anchor[0]} is missing (log truncated)",
        )
    return ChainReport(ok=True, checked=checked, head_seq=checked, head_hash=prev_hash)
