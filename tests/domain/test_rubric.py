from __future__ import annotations

import copy
import math
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from rubricops.domain.rubric import (
    MAX_SCALE_POINTS,
    AnchorExample,
    Criterion,
    Rubric,
    Scale,
    ScaleLevel,
    exact_decimal,
)
from tests.factories import criterion_dict, make_rubric, rubric_dict


def _reject(data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Rubric.model_validate(data)


# --- valid documents -------------------------------------------------------------


def test_valid_rubric_round_trips() -> None:
    rubric = make_rubric()
    assert rubric.criterion_ids == ("accuracy", "clarity")
    assert rubric.pass_threshold == pytest.approx(0.6)
    assert Rubric.model_validate(rubric.model_dump()) == rubric


def test_weights_that_sum_to_one_as_written_are_accepted() -> None:
    # 0.7 + 0.2 + 0.1 is not 1.0 in binary floating point, but it is as written.
    assert 0.7 + 0.2 + 0.1 != 1.0
    rubric = make_rubric(
        criterion_dict("a", weight=0.7),
        criterion_dict("b", weight=0.2),
        criterion_dict("c", weight=0.1),
    )
    assert len(rubric.criteria) == 3
    thirds = make_rubric(
        criterion_dict("a", weight=0.3334),
        criterion_dict("b", weight=0.3333),
        criterion_dict("c", weight=0.3333),
    )
    assert len(thirds.criteria) == 3


def test_text_is_stripped() -> None:
    data = rubric_dict()
    data["title"] = "  Padded title \n"
    assert Rubric.model_validate(data).title == "Padded title"


def test_guide_is_stored_in_scale_order_and_anchors_keep_authored_order() -> None:
    data = criterion_dict("accuracy", low=1, high=3)
    data["guide"].reverse()
    data["anchors"] = [
        {"score": 3, "text": "top"},
        {"score": 1, "text": "first bottom"},
        {"score": 2, "text": "middle"},
        {"score": 1, "text": "second bottom"},
    ]
    criterion = Criterion.model_validate(data)
    assert [level.score for level in criterion.guide] == [1, 2, 3]
    assert [a.text for a in criterion.anchors] == ["first bottom", "second bottom", "middle", "top"]
    assert [a.text for a in criterion.anchors_for(1)] == ["first bottom", "second bottom"]


def test_models_are_immutable() -> None:
    rubric = make_rubric()
    with pytest.raises(ValidationError, match="frozen"):
        rubric.pass_threshold = 0.1  # type: ignore[misc]


def test_lookups() -> None:
    rubric = make_rubric()
    accuracy = rubric.criterion("accuracy")
    assert accuracy.level(3).label == "level 3"
    assert accuracy.anchors_for(2)[0].text == "example of accuracy at 2"
    with pytest.raises(KeyError, match="no criterion 'nope'"):
        rubric.criterion("nope")
    with pytest.raises(KeyError, match="not on scale"):
        accuracy.level(9)


def test_normalise_maps_scale_onto_unit_interval_exactly() -> None:
    criterion = Criterion.model_validate(criterion_dict(low=1, high=4))
    assert [criterion.normalise(s) for s in (1, 2, 3, 4)] == [
        Fraction(0),
        Fraction(1, 3),
        Fraction(2, 3),
        Fraction(1),
    ]


@pytest.mark.parametrize("score", [0, 5, True, 2.0, "2"])
def test_normalise_rejects_off_scale_and_non_int(score: object) -> None:
    criterion = Criterion.model_validate(criterion_dict(low=1, high=4))
    with pytest.raises(ValueError, match="not on scale"):
        criterion.normalise(score)  # type: ignore[arg-type]


def test_scale_helpers() -> None:
    scale = Scale(low=0, high=2)
    assert list(scale.points) == [0, 1, 2]
    assert str(scale) == "0..2"
    assert 1 in scale
    assert True not in scale
    assert 3 not in scale


def test_exact_decimal() -> None:
    assert exact_decimal(0.1) == Fraction(1, 10)
    assert exact_decimal(0.3334) == Fraction(3334, 10000)
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="finite"):
            exact_decimal(bad)


# --- scale, level and anchor errors ---------------------------------------------


@pytest.mark.parametrize(("low", "high"), [(3, 3), (5, 1)])
def test_scale_rejects_empty_or_inverted_bounds(low: int, high: int) -> None:
    with pytest.raises(ValidationError, match="must be below high"):
        Scale(low=low, high=high)


def test_scale_rejects_too_many_points() -> None:
    Scale(low=0, high=MAX_SCALE_POINTS - 1)
    with pytest.raises(ValidationError, match="at most 11 are allowed"):
        Scale(low=0, high=MAX_SCALE_POINTS)


@pytest.mark.parametrize("field", ["label", "descriptor"])
def test_scale_level_rejects_blank_text(field: str) -> None:
    data = {"score": 1, "label": "ok", "descriptor": "ok", field: "   "}
    with pytest.raises(ValidationError, match="at least 1 character"):
        ScaleLevel.model_validate(data)


def test_anchor_rejects_blank_text() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        AnchorExample(score=1, text="")


def test_unknown_keys_are_rejected() -> None:
    data = rubric_dict()
    data["criteria"][0]["weigth"] = 0.5
    _reject(data, "Extra inputs are not permitted")


# --- criterion errors ------------------------------------------------------------


@pytest.mark.parametrize("cid", ["Accuracy", "1st", "has space", "has-dash", "", "x" * 49])
def test_criterion_rejects_bad_ids(cid: str) -> None:
    with pytest.raises(ValidationError, match="should match pattern"):
        Criterion.model_validate(criterion_dict(cid))


@pytest.mark.parametrize("weight", [0, -0.2])
def test_criterion_rejects_non_positive_weight(weight: float) -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Criterion.model_validate(criterion_dict(weight=weight))


@pytest.mark.parametrize("weight", [math.inf, math.nan])
def test_criterion_rejects_non_finite_weight(weight: float) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        Criterion.model_validate(criterion_dict(weight=weight))


@pytest.mark.parametrize("field", ["guide", "anchors"])
def test_criterion_rejects_empty_guide_or_anchors(field: str) -> None:
    data = criterion_dict()
    data[field] = []
    with pytest.raises(ValidationError, match="at least 1 item"):
        Criterion.model_validate(data)


def test_guide_rejects_duplicate_descriptor() -> None:
    data = criterion_dict(low=1, high=3)
    data["guide"].append({"score": 2, "label": "again", "descriptor": "second level 2"})
    with pytest.raises(ValidationError, match=r"more than one descriptor for \[2\]"):
        Criterion.model_validate(data)


def test_guide_rejects_level_outside_scale() -> None:
    data = criterion_dict(low=1, high=3)
    data["guide"].append({"score": 7, "label": "extra", "descriptor": "off the scale"})
    with pytest.raises(ValidationError, match=r"guide levels \[7\] are outside scale 1\.\.3"):
        Criterion.model_validate(data)


def test_guide_rejects_missing_scale_point() -> None:
    data = criterion_dict(low=1, high=4)
    data["guide"] = [level for level in data["guide"] if level["score"] not in (2, 3)]
    with pytest.raises(ValidationError, match=r"guide is missing scale points \[2, 3\]"):
        Criterion.model_validate(data)


def test_anchor_outside_scale_is_rejected() -> None:
    data = criterion_dict(low=1, high=4)
    data["anchors"].append({"score": 0, "text": "below the scale"})
    with pytest.raises(ValidationError, match=r"anchors at \[0\] are outside scale 1\.\.4"):
        Criterion.model_validate(data)


def test_level_without_anchor_is_rejected() -> None:
    data = criterion_dict(low=1, high=4)
    data["anchors"] = [a for a in data["anchors"] if a["score"] != 4]
    with pytest.raises(ValidationError, match=r"no anchor example for scale points \[4\]"):
        Criterion.model_validate(data)


@pytest.mark.parametrize("gating", [0, 5])
def test_gating_outside_scale_is_rejected(gating: int) -> None:
    with pytest.raises(ValidationError, match=r"gating minimum \d is outside scale"):
        Criterion.model_validate(criterion_dict(low=1, high=4, gating=gating))


def test_gating_at_bottom_of_scale_is_rejected() -> None:
    with pytest.raises(ValidationError, match="can never fail"):
        Criterion.model_validate(criterion_dict(low=1, high=4, gating=1))


# --- rubric errors ---------------------------------------------------------------


@pytest.mark.parametrize("rid", ["Upper", "-leading-dash", "", "has space"])
def test_rubric_rejects_bad_ids(rid: str) -> None:
    data = rubric_dict()
    data["id"] = rid
    _reject(data, "should match pattern")


@pytest.mark.parametrize(
    ("threshold", "match"),
    [
        (-0.01, "greater than or equal to 0"),
        (1.01, "less than or equal to 1"),
        (math.nan, "finite"),
    ],
)
def test_rubric_rejects_threshold_outside_unit_interval(threshold: float, match: str) -> None:
    _reject(rubric_dict(threshold=threshold), match)


def test_rubric_accepts_threshold_bounds() -> None:
    assert make_rubric(threshold=0).pass_threshold == 0
    assert make_rubric(threshold=1).pass_threshold == 1


def test_rubric_requires_a_criterion() -> None:
    data = rubric_dict()
    data["criteria"] = []
    _reject(data, "at least 1 item")


def test_rubric_rejects_duplicate_criterion_ids() -> None:
    data = rubric_dict(
        criterion_dict("accuracy", weight=0.5),
        criterion_dict("accuracy", weight=0.25),
        criterion_dict("clarity", weight=0.25),
    )
    _reject(data, r"duplicate criterion ids: \['accuracy'\]")


@pytest.mark.parametrize(("w1", "w2", "total"), [(0.5, 0.4, "0.9"), (0.7, 0.4, "1.1")])
def test_rubric_rejects_weights_that_do_not_normalise(w1: float, w2: float, total: str) -> None:
    data = rubric_dict(criterion_dict("a", weight=w1), criterion_dict("b", weight=w2))
    _reject(data, f"weights must sum to 1, got {total}")


def test_single_invalid_criterion_fails_the_whole_rubric() -> None:
    data = rubric_dict()
    broken = copy.deepcopy(data)
    broken["criteria"][1]["guide"].pop()
    _reject(broken, "guide is missing scale points")
    Rubric.model_validate(data)
