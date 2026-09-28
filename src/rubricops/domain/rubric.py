"""Rubric documents: weighted criteria on integer scales with scoring guides and anchors.

A :class:`Rubric` is an immutable, validated value. Everything a reviewer needs to
score consistently lives in it: each :class:`Criterion` declares an integer
:class:`Scale`, a scoring guide with exactly one :class:`ScaleLevel` descriptor per
scale point, and at least one :class:`AnchorExample` per level so that "a 3 on
clarity" means the same thing to every reviewer.

Validation is strict on purpose. A rubric that is ambiguous (a scale point with no
descriptor), unanchored (a level with no example) or unweighable (weights that do
not sum to 1) produces noisy labels, and noisy labels are expensive to discover
after thousands of reviews have been collected against it.

Weights are compared in exact decimal arithmetic via :func:`exact_decimal`, so
``0.7 + 0.2 + 0.1`` sums to exactly 1 the way the author wrote it rather
than to whatever binary floating point makes of it.
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

#: Largest number of points a single criterion scale may have (for example 0..10).
MAX_SCALE_POINTS = 11

#: Absolute tolerance, in exact arithmetic, for the sum of criterion weights.
WEIGHT_SUM_TOLERANCE = Fraction(1, 10**9)

CriterionId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,47}$")]
RubricId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=4000)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=200)]


def exact_decimal(value: float) -> Fraction:
    """Return the exact rational value of the shortest decimal that round-trips ``value``.

    ``Fraction(0.1)`` is the binary approximation 3602879701896397/36028797018963968;
    ``exact_decimal(0.1)`` is 1/10, which is what a rubric author typed in YAML.
    """
    if not math.isfinite(value):
        msg = f"expected a finite number, got {value!r}"
        raise ValueError(msg)
    return Fraction(repr(float(value)))


class _Frozen(BaseModel):
    """Base for rubric value objects: immutable, strict about unknown keys, trimmed text."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Scale(_Frozen):
    """An inclusive integer scale ``low..high``, for example 1..5."""

    low: int
    high: int

    @model_validator(mode="after")
    def _check_bounds(self) -> Scale:
        if self.low >= self.high:
            msg = f"scale low ({self.low}) must be below high ({self.high})"
            raise ValueError(msg)
        if self.high - self.low + 1 > MAX_SCALE_POINTS:
            msg = (
                f"scale {self.low}..{self.high} has {self.high - self.low + 1} points; "
                f"at most {MAX_SCALE_POINTS} are allowed"
            )
            raise ValueError(msg)
        return self

    @property
    def points(self) -> range:
        """Every valid score on this scale, lowest first."""
        return range(self.low, self.high + 1)

    def __contains__(self, score: object) -> bool:
        return isinstance(score, int) and not isinstance(score, bool) and score in self.points

    def __str__(self) -> str:
        return f"{self.low}..{self.high}"


class ScaleLevel(_Frozen):
    """The scoring-guide entry for one scale point: a short label and a descriptor."""

    score: int
    label: ShortText
    descriptor: NonEmptyText


class AnchorExample(_Frozen):
    """A worked example showing what a response at ``score`` looks like, and why."""

    score: int
    text: NonEmptyText
    rationale: NonEmptyText | None = None


class Criterion(_Frozen):
    """One weighted dimension of a rubric, scored on an integer scale.

    ``gating`` is an optional minimum: a review fails when this criterion scores
    below it, however high the weighted total is (for example "a factually wrong
    answer fails regardless of how well written it is").
    """

    id: CriterionId
    title: ShortText
    weight: float = Field(gt=0, allow_inf_nan=False)
    scale: Scale
    gating: int | None = None
    guide: tuple[ScaleLevel, ...] = Field(min_length=1)
    anchors: tuple[AnchorExample, ...] = Field(min_length=1)

    @field_validator("guide")
    @classmethod
    def _order_guide(cls, guide: tuple[ScaleLevel, ...]) -> tuple[ScaleLevel, ...]:
        # Canonical order, so the content hash ignores the order levels were written in.
        return tuple(sorted(guide, key=lambda level: level.score))

    @field_validator("anchors")
    @classmethod
    def _order_anchors(cls, anchors: tuple[AnchorExample, ...]) -> tuple[AnchorExample, ...]:
        # Stable sort: the author's order among anchors for the same level is kept.
        return tuple(sorted(anchors, key=lambda anchor: anchor.score))

    @model_validator(mode="after")
    def _check_guide_and_anchors(self) -> Criterion:
        points = set(self.scale.points)
        counts = Counter(level.score for level in self.guide)
        duplicated = sorted(score for score, n in counts.items() if n > 1)
        if duplicated:
            msg = f"criterion {self.id!r}: guide has more than one descriptor for {duplicated}"
            raise ValueError(msg)
        outside = sorted(set(counts) - points)
        if outside:
            msg = f"criterion {self.id!r}: guide levels {outside} are outside scale {self.scale}"
            raise ValueError(msg)
        missing = sorted(points - set(counts))
        if missing:
            msg = f"criterion {self.id!r}: guide is missing scale points {missing}"
            raise ValueError(msg)

        anchor_scores = {anchor.score for anchor in self.anchors}
        stray = sorted(anchor_scores - points)
        if stray:
            msg = f"criterion {self.id!r}: anchors at {stray} are outside scale {self.scale}"
            raise ValueError(msg)
        unanchored = sorted(points - anchor_scores)
        if unanchored:
            msg = f"criterion {self.id!r}: no anchor example for scale points {unanchored}"
            raise ValueError(msg)

        if self.gating is not None:
            if self.gating not in points:
                msg = (
                    f"criterion {self.id!r}: gating minimum {self.gating} is outside "
                    f"scale {self.scale}"
                )
                raise ValueError(msg)
            if self.gating == self.scale.low:
                msg = (
                    f"criterion {self.id!r}: gating minimum {self.gating} is the bottom of "
                    "the scale and can never fail; remove it or raise it"
                )
                raise ValueError(msg)

        return self

    def level(self, score: int) -> ScaleLevel:
        """Return the guide entry for ``score``."""
        for level in self.guide:
            if level.score == score:
                return level
        msg = f"{score} is not on scale {self.scale} of criterion {self.id!r}"
        raise KeyError(msg)

    def anchors_for(self, score: int) -> tuple[AnchorExample, ...]:
        """Return the anchor examples for ``score`` in authored order."""
        return tuple(anchor for anchor in self.anchors if anchor.score == score)

    def normalise(self, score: int) -> Fraction:
        """Map ``score`` onto [0, 1] exactly: low -> 0, high -> 1, linear in between."""
        if score not in self.scale:
            msg = f"{score!r} is not on scale {self.scale} of criterion {self.id!r}"
            raise ValueError(msg)
        return Fraction(score - self.scale.low, self.scale.high - self.scale.low)


class Rubric(_Frozen):
    """A complete rubric document.

    ``pass_threshold`` applies to the normalised weighted score in [0, 1]. Criteria
    are kept in authored order, which is the order a review form presents them.
    """

    id: RubricId
    title: ShortText
    description: NonEmptyText | None = None
    pass_threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    criteria: tuple[Criterion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_criteria(self) -> Rubric:
        counts = Counter(criterion.id for criterion in self.criteria)
        duplicated = sorted(cid for cid, n in counts.items() if n > 1)
        if duplicated:
            msg = f"duplicate criterion ids: {duplicated}"
            raise ValueError(msg)
        total = sum((exact_decimal(c.weight) for c in self.criteria), Fraction(0))
        if abs(total - 1) > WEIGHT_SUM_TOLERANCE:
            msg = (
                f"criterion weights must sum to 1, got {float(total):g}; "
                "weights are fractions of the total score (0.4 means 40%)"
            )
            raise ValueError(msg)
        return self

    @property
    def criterion_ids(self) -> tuple[str, ...]:
        """Criterion ids in authored order."""
        return tuple(criterion.id for criterion in self.criteria)

    def criterion(self, criterion_id: str) -> Criterion:
        """Return the criterion called ``criterion_id``."""
        for criterion in self.criteria:
            if criterion.id == criterion_id:
                return criterion
        msg = f"rubric {self.id!r} has no criterion {criterion_id!r}"
        raise KeyError(msg)
