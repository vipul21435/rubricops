"""Content hashing and append-only version history for rubrics.

A rubric version is identified by the sha256 of its canonical JSON. Canonical means:
keys sorted, no insignificant whitespace, ASCII-only escapes, and the model's own
normalisation applied first (trimmed text, guide levels and anchors in scale order,
numbers parsed to their typed values). Two YAML files that differ only in key order,
indentation or trailing spaces therefore hash identically, while any change a
reviewer could notice produces a new hash.

:class:`RubricRegistry` is the in-memory history: publishing appends version n+1
with a changelog message, nothing is ever edited or removed, and reviews refer to a
``(rubric_id, version)`` pair whose content can always be recovered.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from rubricops.domain.clock import Clock, SystemClock, ensure_utc
from rubricops.domain.diff import RubricDiff, diff_rubrics
from rubricops.domain.rubric import NonEmptyText, Rubric

ContentHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def canonical_json(rubric: Rubric) -> str:
    """Serialise ``rubric`` to its canonical JSON text (the bytes that get hashed)."""
    return json.dumps(
        rubric.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_hash(rubric: Rubric) -> str:
    """Return the lowercase hex sha256 of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(rubric).encode("ascii")).hexdigest()


class RubricVersion(BaseModel):
    """One immutable, published version of a rubric."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    rubric_id: str
    version: int = Field(ge=1)
    content_hash: ContentHash
    message: NonEmptyText
    created_at: AwareDatetime
    rubric: Rubric

    @field_validator("created_at")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _check_integrity(self) -> RubricVersion:
        if self.rubric.id != self.rubric_id:
            msg = f"rubric_id {self.rubric_id!r} does not match document id {self.rubric.id!r}"
            raise ValueError(msg)
        actual = content_hash(self.rubric)
        if actual != self.content_hash:
            msg = f"content_hash {self.content_hash} does not match the document ({actual})"
            raise ValueError(msg)
        return self

    @property
    def label(self) -> str:
        """Short display form, for example ``code-explanation@v2 (1a2b3c4d5e6f)``."""
        return f"{self.rubric_id}@v{self.version} ({self.content_hash[:12]})"


class RubricRegistryError(Exception):
    """Base class for registry errors."""


class UnknownRubricError(RubricRegistryError, LookupError):
    """No rubric with this id has been published."""


class UnknownVersionError(RubricRegistryError, LookupError):
    """The rubric exists but not at this version."""


class UnchangedRubricError(RubricRegistryError, ValueError):
    """Publishing would create a version identical to the current head."""


class RubricRegistry:
    """Append-only version history for any number of rubrics."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or SystemClock()
        self._history: dict[str, list[RubricVersion]] = {}

    def publish(self, rubric: Rubric, message: str) -> RubricVersion:
        """Append ``rubric`` as the next version of ``rubric.id`` and return it.

        Re-publishing the current head is refused (:class:`UnchangedRubricError`), so
        every version records a real change. Reverting to an older document is a
        change like any other: it becomes a new version that shares the old hash.
        """
        digest = content_hash(rubric)
        history = self._history.get(rubric.id, [])
        if history and history[-1].content_hash == digest:
            msg = f"{rubric.id} is unchanged since {history[-1].label}; nothing to publish"
            raise UnchangedRubricError(msg)
        created = RubricVersion(
            rubric_id=rubric.id,
            version=len(history) + 1,
            content_hash=digest,
            message=message,
            created_at=self._clock.now(),
            rubric=rubric,
        )
        self._history.setdefault(rubric.id, []).append(created)
        return created

    def rubric_ids(self) -> tuple[str, ...]:
        """Every published rubric id, in first-publish order."""
        return tuple(self._history)

    def history(self, rubric_id: str) -> tuple[RubricVersion, ...]:
        """All versions of ``rubric_id``, oldest first."""
        return tuple(self._versions(rubric_id))

    def latest(self, rubric_id: str) -> RubricVersion:
        """The current head version of ``rubric_id``."""
        return self._versions(rubric_id)[-1]

    def get(self, rubric_id: str, version: int) -> RubricVersion:
        """A specific version of ``rubric_id`` (1-based)."""
        versions = self._versions(rubric_id)
        if not 1 <= version <= len(versions):
            msg = f"{rubric_id} has versions 1..{len(versions)}, not {version}"
            raise UnknownVersionError(msg)
        return versions[version - 1]

    def find_by_hash(self, digest: str) -> tuple[RubricVersion, ...]:
        """Every version whose content hashes to ``digest`` (more than one after a revert)."""
        return tuple(
            version
            for versions in self._history.values()
            for version in versions
            if version.content_hash == digest
        )

    def diff(self, rubric_id: str, old_version: int, new_version: int) -> RubricDiff:
        """Structural diff from ``old_version`` to ``new_version`` of ``rubric_id``."""
        return diff_rubrics(
            self.get(rubric_id, old_version).rubric, self.get(rubric_id, new_version).rubric
        )

    def __contains__(self, rubric_id: object) -> bool:
        return rubric_id in self._history

    def __len__(self) -> int:
        """Total number of published versions across all rubrics."""
        return sum(len(versions) for versions in self._history.values())

    def _versions(self, rubric_id: str) -> list[RubricVersion]:
        try:
            return self._history[rubric_id]
        except KeyError:
            msg = f"no rubric {rubric_id!r} has been published"
            raise UnknownRubricError(msg) from None
