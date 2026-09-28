"""Property-based checks of the scoring engine over randomly generated rubrics."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rubricops.domain.rubric import Rubric
from rubricops.domain.scoring import score_review
from tests.factories import criterion_dict, rubric_dict

PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,
)


@st.composite
def rubrics(draw: st.DrawFn) -> Rubric:
    """Valid rubrics with 1-6 criteria, arbitrary integer scales, gates and thresholds."""
    n = draw(st.integers(min_value=1, max_value=6))
    raw_weights = draw(st.lists(st.integers(1, 50), min_size=n, max_size=n))
    total = sum(raw_weights)
    criteria = []
    for i, raw in enumerate(raw_weights):
        low = draw(st.integers(-3, 3))
        high = low + draw(st.integers(1, 10))
        gating = draw(st.none() | st.integers(low + 1, high))
        criteria.append(
            criterion_dict(f"c{i}", weight=raw / total, low=low, high=high, gating=gating)
        )
    threshold = draw(st.floats(0.0, 1.0, allow_nan=False))
    return Rubric.model_validate(rubric_dict(*criteria, threshold=threshold))


@st.composite
def rubric_and_scores(draw: st.DrawFn) -> tuple[Rubric, dict[str, int]]:
    rubric = draw(rubrics())
    scores = {
        c.id: draw(st.integers(c.scale.low, c.scale.high), label=c.id) for c in rubric.criteria
    }
    return rubric, scores


@PROPERTY_SETTINGS
@given(rubric_and_scores())
def test_score_is_in_unit_interval_and_is_the_sum_of_contributions(
    case: tuple[Rubric, dict[str, int]],
) -> None:
    rubric, scores = case
    result = score_review(rubric, scores)
    assert 0.0 <= result.score <= 1.0
    assert math.fsum(c.contribution for c in result.criteria) == pytest.approx(result.score)
    assert math.fsum(c.weight for c in result.criteria) == pytest.approx(1.0)
    assert result.passed == (result.meets_threshold and not result.gating_failures)


@PROPERTY_SETTINGS
@given(rubric_and_scores(), st.data())
def test_score_is_monotone_in_each_criterion(
    case: tuple[Rubric, dict[str, int]], data: st.DataObject
) -> None:
    rubric, scores = case
    raisable = [c for c in rubric.criteria if scores[c.id] < c.scale.high]
    if not raisable:
        return
    criterion = data.draw(st.sampled_from(raisable), label="raised")
    step = data.draw(st.integers(1, criterion.scale.high - scores[criterion.id]), label="step")
    before = score_review(rubric, scores)
    after = score_review(rubric, {**scores, criterion.id: scores[criterion.id] + step})
    assert after.score > before.score
    assert len(after.gating_failures) <= len(before.gating_failures)
    if before.passed:
        assert after.passed


@PROPERTY_SETTINGS
@given(rubrics())
def test_extremes_score_zero_and_one(rubric: Rubric) -> None:
    lowest = score_review(rubric, {c.id: c.scale.low for c in rubric.criteria})
    highest = score_review(rubric, {c.id: c.scale.high for c in rubric.criteria})
    assert lowest.score == 0.0
    assert highest.score == 1.0
    assert highest.passed
