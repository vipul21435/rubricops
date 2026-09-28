from __future__ import annotations

import copy
from typing import Any

from rubricops.domain.diff import (
    GatingChange,
    RubricDiff,
    ScaleChange,
    ThresholdChange,
    WeightChange,
    diff_rubrics,
)
from rubricops.domain.rubric import Rubric, Scale
from tests.factories import criterion_dict, rubric_dict


def _base() -> dict[str, Any]:
    return rubric_dict(
        criterion_dict("accuracy", weight=0.5, gating=3),
        criterion_dict("clarity", weight=0.3),
        criterion_dict("style", weight=0.2, low=0, high=2),
    )


def _diff(old: dict[str, Any], new: dict[str, Any]) -> RubricDiff:
    return diff_rubrics(Rubric.model_validate(old), Rubric.model_validate(new))


def test_identical_documents_have_an_empty_diff() -> None:
    diff = _diff(_base(), _base())
    assert diff.is_empty
    assert not diff.affects_scoring
    assert diff.summary_lines() == ["no changes"]


def test_added_and_removed_criteria() -> None:
    new = rubric_dict(
        criterion_dict("accuracy", weight=0.5, gating=3),
        criterion_dict("clarity", weight=0.3),
        criterion_dict("safety", weight=0.2, low=0, high=1, gating=1),
    )
    diff = _diff(_base(), new)
    assert [ref.id for ref in diff.added] == ["safety"]
    assert [ref.id for ref in diff.removed] == ["style"]
    assert diff.added[0].gating == 1
    assert diff.affects_scoring
    assert diff.summary_lines() == [
        "+ added criterion safety: weight 0.2, scale 0..1, gating 1",
        "- removed criterion style: weight 0.2, scale 0..2",
    ]


def test_reweighting() -> None:
    new = _base()
    new["criteria"][0]["weight"] = 0.6
    new["criteria"][2]["weight"] = 0.1
    diff = _diff(_base(), new)
    assert diff.reweighted == (
        WeightChange(criterion_id="accuracy", old=0.5, new=0.6),
        WeightChange(criterion_id="style", old=0.2, new=0.1),
    )
    assert diff.affects_scoring
    assert "~ reweighted accuracy: 0.5 -> 0.6" in diff.summary_lines()


def test_rescaling_reports_the_scale_not_the_implied_guide_rewrite() -> None:
    new = _base()
    new["criteria"][1] = criterion_dict("clarity", weight=0.3, low=1, high=5)
    diff = _diff(_base(), new)
    assert diff.rescaled == (
        ScaleChange(criterion_id="clarity", old=Scale(low=1, high=4), new=Scale(low=1, high=5)),
    )
    assert diff.guide_changed == ()
    assert diff.summary_lines() == ["~ rescaled clarity: 1..4 -> 1..5"]


def test_gating_added_changed_and_removed() -> None:
    new = _base()
    new["criteria"][0]["gating"] = 4
    new["criteria"][1]["gating"] = 2
    new["criteria"][2]["gating"] = None
    old = _base()
    old["criteria"][2]["gating"] = 1
    diff = _diff(old, new)
    assert diff.gating_changed == (
        GatingChange(criterion_id="accuracy", old=3, new=4),
        GatingChange(criterion_id="clarity", old=None, new=2),
        GatingChange(criterion_id="style", old=1, new=None),
    )
    assert diff.affects_scoring
    assert "~ gating clarity: none -> 2" in diff.summary_lines()
    assert "~ gating style: 1 -> none" in diff.summary_lines()


def test_threshold_change() -> None:
    new = _base()
    new["pass_threshold"] = 0.75
    diff = _diff(_base(), new)
    assert diff.threshold == ThresholdChange(old=0.6, new=0.75)
    assert diff.affects_scoring
    assert diff.summary_lines() == ["~ pass_threshold: 0.6 -> 0.75"]


def test_guide_and_anchor_edits_do_not_affect_scoring() -> None:
    new = _base()
    new["criteria"][0]["guide"][0]["descriptor"] = "rewritten descriptor"
    new["criteria"][2]["anchors"].append({"score": 1, "text": "a second middle example"})
    diff = _diff(_base(), new)
    assert diff.guide_changed == ("accuracy", "style")
    assert not diff.affects_scoring
    assert not diff.is_empty
    assert diff.summary_lines() == ["~ guide or anchors edited: accuracy, style"]


def test_titles_order_and_metadata() -> None:
    new = copy.deepcopy(_base())
    new["criteria"][1]["title"] = "Clarity of the explanation"
    new["criteria"].reverse()
    new["title"] = "Renamed rubric"
    new["description"] = None
    diff = _diff(_base(), new)
    assert diff.retitled == ("clarity",)
    assert diff.reordered
    assert diff.metadata_changed == ("title", "description")
    assert not diff.affects_scoring
    assert diff.summary_lines() == [
        "~ retitled: clarity",
        "~ criteria reordered",
        "~ rubric metadata changed: title, description",
    ]


def test_adding_a_criterion_is_not_reported_as_reordering() -> None:
    new = rubric_dict(
        criterion_dict("intro", weight=0.1),
        criterion_dict("accuracy", weight=0.4, gating=3),
        criterion_dict("clarity", weight=0.3),
        criterion_dict("style", weight=0.2, low=0, high=2),
    )
    diff = _diff(_base(), new)
    assert not diff.reordered
    assert [ref.id for ref in diff.added] == ["intro"]


def test_diff_is_directional() -> None:
    new = _base()
    new["pass_threshold"] = 0.8
    forward = _diff(_base(), new)
    backward = _diff(new, _base())
    assert forward.threshold == ThresholdChange(old=0.6, new=0.8)
    assert backward.threshold == ThresholdChange(old=0.8, new=0.6)
