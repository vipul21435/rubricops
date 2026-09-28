from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rubricops.domain.rubric import Rubric
from rubricops.loaders import DocumentError, load_document, load_rubric, load_scores
from tests.factories import rubric_dict

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_yaml_and_json_rubrics_load_to_the_same_document(tmp_path: Path) -> None:
    data = rubric_dict()
    as_yaml = _write(tmp_path / "r.yaml", yaml.safe_dump(data, sort_keys=False))
    as_json = _write(tmp_path / "r.JSON", json.dumps(data))
    assert load_rubric(as_yaml) == load_rubric(as_json) == Rubric.model_validate(data)


EXAMPLE_RUBRICS = sorted((EXAMPLES / "rubrics").glob("*.yaml"))


def test_example_rubrics_are_present() -> None:
    assert [p.name for p in EXAMPLE_RUBRICS] == [
        "action-items.yaml",
        "code-explanation.v1.yaml",
        "code-explanation.yaml",
    ]


@pytest.mark.parametrize("path", EXAMPLE_RUBRICS, ids=lambda p: p.name)
def test_every_example_rubric_is_valid(path: Path) -> None:
    assert load_rubric(path).criteria


def test_yaml_aliases_and_merge_keys_are_supported(tmp_path: Path) -> None:
    shared = _write(
        tmp_path / "shared.yaml",
        "a: &level {score: 1, text: shared}\nb: *level\nc: {<<: *level, text: merged}\n",
    )
    assert load_document(shared) == {
        "a": {"score": 1, "text": "shared"},
        "b": {"score": 1, "text": "shared"},
        "c": {"score": 1, "text": "merged"},
    }


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "dup.yaml", "id: a\nweight: 0.5\nweight: 0.7\n")
    with pytest.raises(DocumentError, match="found duplicate key 'weight'"):
        load_document(path)
    nested = _write(tmp_path / "nested.yaml", "a: &x {k: 1}\nb: {<<: *x, j: 2, j: 3}\n")
    with pytest.raises(DocumentError, match="found duplicate key 'j'"):
        load_document(nested)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "dup.json", '{"weight": 0.5, "weight": 0.7}')
    with pytest.raises(DocumentError, match="duplicate key 'weight'"):
        load_document(path)


@pytest.mark.parametrize(
    ("name", "text", "match"),
    [
        ("r.toml", "id = 1", "unsupported file type '.toml'"),
        ("r.yaml", "id: [unclosed", "cannot parse"),
        ("r.json", "{not json", "cannot parse"),
    ],
)
def test_unreadable_documents(tmp_path: Path, name: str, text: str, match: str) -> None:
    with pytest.raises(DocumentError, match=match):
        load_document(_write(tmp_path / name, text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="cannot read file") as excinfo:
        load_document(tmp_path / "absent.yaml")
    assert excinfo.value.path == tmp_path / "absent.yaml"


@pytest.mark.parametrize("loader", [load_rubric, load_scores])
def test_top_level_must_be_a_mapping(tmp_path: Path, loader: object) -> None:
    path = _write(tmp_path / "list.yaml", "- 1\n- 2\n")
    with pytest.raises(DocumentError, match="mapping, got list"):
        loader(path)  # type: ignore[operator]


def test_validation_errors_are_flattened_with_locations(tmp_path: Path) -> None:
    data = rubric_dict()
    data["criteria"][1]["guide"].pop()
    data["pass_threshold"] = 3
    path = _write(tmp_path / "bad.yaml", yaml.safe_dump(data))
    with pytest.raises(DocumentError) as excinfo:
        load_rubric(path)
    assert excinfo.value.issues == (
        "pass_threshold: Input should be less than or equal to 1",
        "criteria.1: criterion 'clarity': guide is missing scale points [2]",
    )


def test_document_level_errors_have_a_placeholder_location(tmp_path: Path) -> None:
    data = rubric_dict()
    data["criteria"][0]["weight"] = 0.1
    path = _write(tmp_path / "weights.yaml", yaml.safe_dump(data))
    with pytest.raises(DocumentError) as excinfo:
        load_rubric(path)
    assert excinfo.value.issues == (
        "<document>: criterion weights must sum to 1, got 0.5; "
        "weights are fractions of the total score (0.4 means 40%)",
    )


def test_load_scores_keeps_values_for_the_scorer_to_validate(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.yaml", "accuracy: 3\nclarity: two\n1: 2\n")
    assert load_scores(path) == {"accuracy": 3, "clarity": "two", "1": 2}
