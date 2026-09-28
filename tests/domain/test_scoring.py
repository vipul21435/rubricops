from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rubricops.domain.clock import FrozenClock
from rubricops.domain.rubric import Rubric
from rubricops.domain.scoring import (
    GatingFailure,
    InvalidScoresError,
    ScoreResult,
    score_review,
    score_version,
    validate_scores,
)
from rubricops.domain.versioning import RubricRegistry, content_hash
from tests.factories import criterion_dict, make_rubric


def _rubric(threshold: float = 0.6, gating: int | None = 2) -> Rubric:
    # accuracy: weight 0.6 on 1..4; clarity: weight 0.4 on 0..2
    return make_rubric(
        criterion_dict("accuracy", weight=0.6, gating=gating),
        criterion_dict("clarity", weight=0.4, low=0, high=2),
        threshold=threshold,
    )


def test_weighted_normalised_score_by_hand() -> None:
    # accuracy 4 on 1..4 -> 1.0; clarity 1 on 0..2 -> 0.5; 0.6 * 1.0 + 0.4 * 0.5 = 0.8
    result = score_review(_rubric(), {"accuracy": 4, "clarity": 1})
    assert result.score == pytest.approx(0.8)
    assert [c.normalised for c in result.criteria] == [1.0, 0.5]
    assert [c.weight for c in result.criteria] == pytest.approx([0.6, 0.4])
    assert [c.contribution for c in result.criteria] == pytest.approx([0.6, 0.2])
    assert result.meets_threshold
    assert result.passed
    assert result.gating_failures == ()
    assert result.rubric_version is None
    assert result.content_hash == content_hash(_rubric())


def test_score_exactly_at_threshold_passes() -> None:
    # 0.3 * 2/3 + 0.7 * 1/2 is exactly 0.55 on paper; naive float arithmetic says less.
    assert 0.3 * (2 / 3) + 0.7 * 0.5 < 0.55
    rubric = make_rubric(
        criterion_dict("accuracy", weight=0.3),
        criterion_dict("clarity", weight=0.7, low=0, high=2),
        threshold=0.55,
    )
    result = score_review(rubric, {"accuracy": 3, "clarity": 1})
    assert result.score == 0.55
    assert result.meets_threshold
    assert result.passed
    assert not score_review(rubric, {"accuracy": 2, "clarity": 1}).passed


def test_gate_fails_even_when_the_weighted_score_passes() -> None:
    # accuracy 2 on 1..4 -> 1/3, below the gate of 3; clarity 2 -> 1.0; total 0.6.
    result = score_review(_rubric(gating=3), {"accuracy": 2, "clarity": 2})
    assert result.meets_threshold
    assert result.gating_failures == (GatingFailure(criterion_id="accuracy", score=2, minimum=3),)
    assert not result.passed


def test_below_threshold_without_gate_failure() -> None:
    result = score_review(_rubric(), {"accuracy": 2, "clarity": 1})
    assert result.score == pytest.approx(0.6 / 3 + 0.2)
    assert not result.meets_threshold
    assert result.gating_failures == ()
    assert not result.passed


def test_bounds_of_the_scale_map_to_zero_and_one() -> None:
    rubric = _rubric(threshold=1.0, gating=None)
    assert score_review(rubric, {"accuracy": 1, "clarity": 0}).score == 0.0
    top = score_review(rubric, {"accuracy": 4, "clarity": 2})
    assert top.score == 1.0
    assert top.passed


def test_threshold_zero_passes_everything_that_clears_the_gates() -> None:
    rubric = _rubric(threshold=0.0, gating=None)
    assert score_review(rubric, {"accuracy": 1, "clarity": 0}).passed


def test_score_version_pins_version_and_hash() -> None:
    registry = RubricRegistry(FrozenClock(datetime(2026, 9, 1, tzinfo=UTC)))
    registry.publish(_rubric(), "initial")
    v2 = registry.publish(_rubric(threshold=0.9), "raise the bar")
    old = score_version(registry.get("sample-rubric", 1), {"accuracy": 4, "clarity": 1})
    new = score_version(v2, {"accuracy": 4, "clarity": 1})
    assert (old.rubric_version, new.rubric_version) == (1, 2)
    assert new.content_hash == v2.content_hash
    assert old.passed
    assert not new.passed
    assert old.score == new.score


def test_result_round_trips_through_json() -> None:
    result = score_review(_rubric(), {"accuracy": 3, "clarity": 2})
    assert ScoreResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("scores", "issue"),
    [
        ({"accuracy": 3}, "missing scores for ['clarity']"),
        ({"accuracy": 3, "clarity": 1, "tone": 2}, "unknown criteria ['tone']"),
        ({"accuracy": 5, "clarity": 1}, "accuracy: 5 is outside scale 1..4"),
        ({"accuracy": 0, "clarity": 1}, "accuracy: 0 is outside scale 1..4"),
        ({"accuracy": 3, "clarity": -1}, "clarity: -1 is outside scale 0..2"),
        ({"accuracy": 3.0, "clarity": 1}, "accuracy: expected an integer, got 3.0"),
        ({"accuracy": "3", "clarity": 1}, "accuracy: expected an integer, got '3'"),
        ({"accuracy": 3, "clarity": True}, "clarity: expected an integer, got True"),
        ({"accuracy": None, "clarity": 1}, "accuracy: expected an integer, got None"),
    ],
)
def test_invalid_scores_are_rejected(scores: dict[str, Any], issue: str) -> None:
    with pytest.raises(InvalidScoresError) as excinfo:
        score_review(_rubric(), scores)
    assert excinfo.value.issues == (issue,)
    assert excinfo.value.rubric_id == "sample-rubric"


def test_every_issue_is_reported_at_once() -> None:
    with pytest.raises(InvalidScoresError) as excinfo:
        validate_scores(_rubric(), {"accuracy": 9, "extra": 1, "zzz": 2})
    assert excinfo.value.issues == (
        "unknown criteria ['extra', 'zzz']",
        "missing scores for ['clarity']",
        "accuracy: 9 is outside scale 1..4",
    )
    assert isinstance(excinfo.value, ValueError)
    assert "invalid scores for rubric 'sample-rubric'" in str(excinfo.value)


def test_validate_scores_returns_a_clean_copy() -> None:
    scores = {"clarity": 2, "accuracy": 1}
    clean = validate_scores(_rubric(), scores)
    assert clean == scores
    assert clean is not scores
