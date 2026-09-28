"""Score a review against a specific rubric version.

A review is a map from criterion id to an integer score. :func:`score_review`
checks it against the rubric (every criterion scored exactly once, every score an
integer on that criterion's scale) and returns a :class:`ScoreResult`:

- ``score``: the weighted mean of each criterion's normalised score
  ``(s - low) / (high - low)``, so it always lies in [0, 1];
- ``criteria``: each criterion's normalised score, weight share and contribution;
- ``gating_failures``: criteria scored below their ``gating`` minimum;
- ``passed``: the score meets ``pass_threshold`` and no gate failed.

The arithmetic is exact. Weights and the threshold are read as the decimals the
author wrote (:func:`~rubricops.domain.rubric.exact_decimal`) and combined as
fractions, so a score that equals the threshold on paper passes; floats appear only
in the reported values.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from pydantic import BaseModel, ConfigDict

from rubricops.domain.rubric import Rubric, Scale, exact_decimal
from rubricops.domain.versioning import RubricVersion, content_hash


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CriterionScore(_Frozen):
    """How one criterion contributed to the total."""

    criterion_id: str
    score: int
    scale: Scale
    normalised: float
    weight: float
    contribution: float


class GatingFailure(_Frozen):
    """A criterion that scored below its gating minimum."""

    criterion_id: str
    score: int
    minimum: int


class ScoreResult(_Frozen):
    """The outcome of scoring one review against one rubric version."""

    rubric_id: str
    rubric_version: int | None
    content_hash: str
    score: float
    pass_threshold: float
    meets_threshold: bool
    gating_failures: tuple[GatingFailure, ...]
    passed: bool
    criteria: tuple[CriterionScore, ...]


class InvalidScoresError(ValueError):
    """The score map does not fit the rubric; ``issues`` lists every problem found."""

    def __init__(self, rubric_id: str, issues: list[str]) -> None:
        self.rubric_id = rubric_id
        self.issues = tuple(issues)
        super().__init__(f"invalid scores for rubric {rubric_id!r}: " + "; ".join(issues))


def validate_scores(rubric: Rubric, scores: Mapping[str, object]) -> dict[str, int]:
    """Check ``scores`` against ``rubric`` and return it as a plain ``{id: int}`` dict.

    Every problem is collected before raising, so a reviewer fixing a form sees all
    of them at once: unknown criteria, missing criteria, non-integer values (``True``
    and ``3.0`` included) and scores off a criterion's scale.
    """
    issues: list[str] = []
    known = set(rubric.criterion_ids)
    unknown = sorted(str(key) for key in scores if key not in known)
    if unknown:
        issues.append(f"unknown criteria {unknown}")
    missing = [cid for cid in rubric.criterion_ids if cid not in scores]
    if missing:
        issues.append(f"missing scores for {missing}")

    clean: dict[str, int] = {}
    for criterion in rubric.criteria:
        if criterion.id not in scores:
            continue
        value = scores[criterion.id]
        if not isinstance(value, int) or isinstance(value, bool):
            issues.append(f"{criterion.id}: expected an integer, got {value!r}")
        elif value not in criterion.scale:
            issues.append(f"{criterion.id}: {value} is outside scale {criterion.scale}")
        else:
            clean[criterion.id] = value
    if issues:
        raise InvalidScoresError(rubric.id, issues)
    return clean


def score_review(
    rubric: Rubric,
    scores: Mapping[str, object],
    *,
    version: int | None = None,
) -> ScoreResult:
    """Score ``scores`` against ``rubric``; ``version`` is recorded when known."""
    clean = validate_scores(rubric, scores)
    weights = {c.id: exact_decimal(c.weight) for c in rubric.criteria}
    total_weight = sum(weights.values(), Fraction(0))

    parts: list[CriterionScore] = []
    gating_failures: list[GatingFailure] = []
    total = Fraction(0)
    for criterion in rubric.criteria:
        value = clean[criterion.id]
        normalised = criterion.normalise(value)
        share = weights[criterion.id] / total_weight
        contribution = share * normalised
        total += contribution
        parts.append(
            CriterionScore(
                criterion_id=criterion.id,
                score=value,
                scale=criterion.scale,
                normalised=float(normalised),
                weight=float(share),
                contribution=float(contribution),
            )
        )
        if criterion.gating is not None and value < criterion.gating:
            gating_failures.append(
                GatingFailure(criterion_id=criterion.id, score=value, minimum=criterion.gating)
            )

    meets = total >= exact_decimal(rubric.pass_threshold)
    return ScoreResult(
        rubric_id=rubric.id,
        rubric_version=version,
        content_hash=content_hash(rubric),
        score=float(total),
        pass_threshold=rubric.pass_threshold,
        meets_threshold=meets,
        gating_failures=tuple(gating_failures),
        passed=meets and not gating_failures,
        criteria=tuple(parts),
    )


def score_version(rubric_version: RubricVersion, scores: Mapping[str, object]) -> ScoreResult:
    """Score against a published version, pinning its number and hash in the result."""
    return score_review(rubric_version.rubric, scores, version=rubric_version.version)
