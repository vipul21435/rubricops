from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from rubricops.cli import app
from rubricops.domain.scoring import ScoreResult
from tests.factories import rubric_dict

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CODE_V1 = EXAMPLES / "rubrics" / "code-explanation.v1.yaml"
CODE_V2 = EXAMPLES / "rubrics" / "code-explanation.yaml"
ACTION_ITEMS = EXAMPLES / "rubrics" / "action-items.yaml"
REVIEW = EXAMPLES / "reviews" / "code-explanation-review.yaml"


def _dump(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_rubric_group_shows_help() -> None:
    result = runner.invoke(app, ["rubric"])
    assert "validate" in result.output
    assert "diff" in result.output
    assert "score" in result.output


# --- validate --------------------------------------------------------------------


def test_validate_reports_each_example() -> None:
    result = runner.invoke(app, ["rubric", "validate", str(ACTION_ITEMS), str(CODE_V2)])
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith(f"ok    {ACTION_ITEMS}  id=action-items criteria=4")
    assert "pass_threshold=0.75 sha256=" in lines[0]
    assert "id=code-explanation criteria=4 pass_threshold=0.7" in lines[1]


def test_validate_fails_on_any_bad_file_but_checks_them_all(tmp_path: Path) -> None:
    data = rubric_dict()
    data["criteria"][0]["anchors"].append({"score": 9, "text": "off the scale"})
    bad = _dump(tmp_path / "bad.yaml", data)
    result = runner.invoke(app, ["rubric", "validate", str(bad), str(ACTION_ITEMS)])
    assert result.exit_code == 1
    assert f"error: {bad}" in result.stderr
    assert "criteria.0: criterion 'accuracy': anchors at [9] are outside scale 1..4" in (
        result.stderr
    )
    assert "ok    " in result.stdout


# --- diff ------------------------------------------------------------------------


def test_diff_of_the_example_versions() -> None:
    result = runner.invoke(app, ["rubric", "diff", str(CODE_V1), str(CODE_V2)])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[1:] == [
        "  + added criterion safe_advice: weight 0.15, scale 0..2, gating 1",
        "  ~ reweighted correctness: 0.5 -> 0.4",
        "  ~ reweighted completeness: 0.3 -> 0.25",
        "  ~ rescaled clarity: 1..3 -> 1..4",
        "  ~ gating correctness: none -> 3",
        "  ~ pass_threshold: 0.6 -> 0.7",
        "  ~ guide or anchors edited: completeness",
        "scoring affected: yes",
    ]


def test_diff_json() -> None:
    result = runner.invoke(app, ["rubric", "diff", "--json", str(CODE_V1), str(CODE_V2)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["affects_scoring"] is True
    assert payload["old"]["id"] == payload["new"]["id"] == "code-explanation"
    assert payload["old"]["sha256"] != payload["new"]["sha256"]
    assert [c["id"] for c in payload["changes"]["added"]] == ["safe_advice"]
    assert payload["changes"]["threshold"] == {"old": 0.6, "new": 0.7}


def test_fail_on_scoring_change_gate(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["rubric", "diff", "--fail-on-scoring-change", str(CODE_V1), str(CODE_V2)]
    )
    assert result.exit_code == 1

    base = rubric_dict()
    reworded = rubric_dict()
    reworded["criteria"][0]["guide"][0]["descriptor"] = "Clearer wording for level one."
    old, new = _dump(tmp_path / "old.yaml", base), _dump(tmp_path / "new.yaml", reworded)
    result = runner.invoke(app, ["rubric", "diff", "--fail-on-scoring-change", str(old), str(new)])
    assert result.exit_code == 0
    assert "scoring affected: no" in result.stdout


def test_diff_of_identical_files(tmp_path: Path) -> None:
    same = _dump(tmp_path / "same.yaml", rubric_dict())
    result = runner.invoke(app, ["rubric", "diff", str(same), str(same)])
    assert result.exit_code == 0
    assert "  no changes" in result.stdout


def test_diff_with_an_invalid_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rubric", "diff", str(tmp_path / "gone.yaml"), str(CODE_V2)])
    assert result.exit_code == 1
    assert "cannot read file" in result.stderr


# --- score -----------------------------------------------------------------------


def test_score_from_file() -> None:
    result = runner.invoke(app, ["rubric", "score", str(CODE_V2), "--scores", str(REVIEW)])
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0].startswith("rubric code-explanation  sha256 ")
    assert lines[1].split() == ["criterion", "score", "scale", "weight", "contribution"]
    assert lines[2].split() == ["correctness", "3", "1..4", "0.400", "0.2667"]
    assert lines[5].split() == ["safe_advice", "1", "0..2", "0.150", "0.0750"]
    assert lines[-1] == "score 0.7083  threshold 0.7  -> PASS"


def test_score_options_override_the_file_and_gates_are_reported() -> None:
    result = runner.invoke(
        app, ["rubric", "score", str(CODE_V2), "-f", str(REVIEW), "-s", "correctness=2"]
    )
    assert result.exit_code == 0, result.output
    assert "gate failed: correctness scored 2, needs at least 3" in result.stdout
    assert result.stdout.splitlines()[-1] == "score 0.5750  threshold 0.7  -> FAIL"


def test_score_json_is_a_score_result() -> None:
    args = ["rubric", "score", str(ACTION_ITEMS), "--json"]
    for pair in ("field_accuracy=3", "recall=2", "grounding=2", "schema_validity=1"):
        args += ["--score", pair]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    parsed = ScoreResult.model_validate_json(result.stdout)
    assert parsed.passed
    assert parsed.score == pytest.approx(0.45 + 0.25 * 2 / 3 + 0.2 + 0.1)


def test_score_reports_every_problem() -> None:
    scores = ["-s", "field_accuracy=4", "-s", "recall=x", "-s", "tone=1"]
    result = runner.invoke(app, ["rubric", "score", str(ACTION_ITEMS), *scores])
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "error: scores do not fit rubric action-items",
        "  unknown criteria ['tone']",
        "  missing scores for ['grounding', 'schema_validity']",
        "  field_accuracy: 4 is outside scale 0..3",
        "  recall: expected an integer, got 'x'",
    ]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "give scores with --scores FILE or --score"),
        (["-s", "no-equals-sign"], "expected CRITERION=SCORE"),
        (["-s", "=3"], "expected CRITERION=SCORE"),
    ],
)
def test_score_usage_errors(extra: list[str], message: str) -> None:
    result = runner.invoke(app, ["rubric", "score", str(CODE_V2), *extra])
    assert result.exit_code == 2
    assert message in " ".join(result.output.split())


def test_score_with_unreadable_inputs(tmp_path: Path) -> None:
    missing = runner.invoke(
        app, ["rubric", "score", str(CODE_V2), "--scores", str(tmp_path / "none.yaml")]
    )
    assert missing.exit_code == 1
    assert "cannot read file" in missing.stderr
    bad_rubric = runner.invoke(app, ["rubric", "score", str(tmp_path / "none.yaml"), "-s", "a=1"])
    assert bad_rubric.exit_code == 1
