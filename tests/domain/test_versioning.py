from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from rubricops.domain.clock import FrozenClock
from rubricops.domain.rubric import Rubric
from rubricops.domain.versioning import (
    RubricRegistry,
    RubricVersion,
    UnchangedRubricError,
    UnknownRubricError,
    UnknownVersionError,
    canonical_json,
    content_hash,
)
from tests.factories import criterion_dict, make_rubric, rubric_dict

T0 = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)

# A literal document and its pinned hash: an accidental change to canonicalisation
# (which would orphan every stored version) fails loudly instead of re-hashing history.
GOLDEN_DOC: dict[str, Any] = {
    "id": "golden",
    "title": "Golden rubric",
    "pass_threshold": 0.5,
    "criteria": [
        {
            "id": "correct",
            "title": "Correct",
            "weight": 1,
            "scale": {"low": 0, "high": 1},
            "guide": [
                {"score": 0, "label": "No", "descriptor": "The answer is wrong."},
                {"score": 1, "label": "Yes", "descriptor": "The answer is right."},
            ],
            "anchors": [{"score": 0, "text": "2 + 2 = 5"}, {"score": 1, "text": "2 + 2 = 4"}],
        }
    ],
}
GOLDEN_CANONICAL = (
    '{"criteria":[{"anchors":[{"rationale":null,"score":0,"text":"2 + 2 = 5"},'
    '{"rationale":null,"score":1,"text":"2 + 2 = 4"}],"gating":null,'
    '"guide":[{"descriptor":"The answer is wrong.","label":"No","score":0},'
    '{"descriptor":"The answer is right.","label":"Yes","score":1}],"id":"correct",'
    '"scale":{"high":1,"low":0},"title":"Correct","weight":1.0}],"description":null,'
    '"id":"golden","pass_threshold":0.5,"title":"Golden rubric"}'
)
GOLDEN_HASH = "101f953321f06861a8b676318aff304d9d94bc16719f5bf499dedfc801d78675"


def _reverse_keys(value: Any) -> Any:
    """Rebuild every dict with its keys in reverse insertion order."""
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


# --- canonical JSON and content hash ---------------------------------------------


def test_canonical_form_and_hash_are_pinned() -> None:
    rubric = Rubric.model_validate(GOLDEN_DOC)
    assert canonical_json(rubric) == GOLDEN_CANONICAL
    assert content_hash(rubric) == GOLDEN_HASH
    assert hashlib.sha256(GOLDEN_CANONICAL.encode()).hexdigest() == GOLDEN_HASH


def test_canonical_json_is_compact_sorted_and_ascii() -> None:
    data = rubric_dict()
    e_acute = "\N{LATIN SMALL LETTER E WITH ACUTE}"
    data["description"] = f"Grades r{e_acute}sum{e_acute}"
    text = canonical_json(Rubric.model_validate(data))
    assert text.isascii()
    assert '"description":"Grades r\\u00e9sum\\u00e9"' in text
    assert ": " not in text
    assert ", " not in text
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed)
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")) == text


def test_hash_ignores_key_order() -> None:
    data = rubric_dict()
    reordered = _reverse_keys(data)
    assert list(reordered) != list(data)
    assert canonical_json(Rubric.model_validate(reordered)) == canonical_json(
        Rubric.model_validate(data)
    )
    assert content_hash(Rubric.model_validate(reordered)) == content_hash(
        Rubric.model_validate(data)
    )


def test_hash_ignores_guide_and_cross_level_anchor_order() -> None:
    data = rubric_dict()
    shuffled = copy.deepcopy(data)
    for criterion in shuffled["criteria"]:
        criterion["guide"].reverse()
        criterion["anchors"].reverse()
    assert content_hash(Rubric.model_validate(shuffled)) == content_hash(
        Rubric.model_validate(data)
    )


def test_hash_ignores_surrounding_whitespace_and_number_spelling() -> None:
    data = rubric_dict(criterion_dict("only", weight=1.0), threshold=1.0)
    respelled = copy.deepcopy(data)
    respelled["title"] = f"  {data['title']}\n"
    respelled["criteria"][0]["weight"] = 1
    respelled["pass_threshold"] = 1
    respelled["criteria"][0]["scale"] = {"low": "1", "high": 4.0}
    assert content_hash(Rubric.model_validate(respelled)) == content_hash(
        Rubric.model_validate(data)
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.update(title="Other title"), id="title"),
        pytest.param(lambda d: d.update(pass_threshold=0.61), id="threshold"),
        pytest.param(lambda d: d["criteria"].reverse(), id="criterion-order"),
        pytest.param(lambda d: d["criteria"][0]["anchors"][0].update(text="new"), id="anchor"),
        pytest.param(lambda d: d["criteria"][0].update(gating=3), id="gating"),
        pytest.param(
            lambda d: (d["criteria"][0].update(weight=0.5), d["criteria"][1].update(weight=0.5)),
            id="weights",
        ),
        pytest.param(
            lambda d: d["criteria"][0]["anchors"].insert(
                0, {"score": 1, "text": "another bottom example"}
            ),
            id="anchor-within-level",
        ),
    ],
)
def test_hash_changes_with_content(mutate: Any) -> None:
    data = rubric_dict()
    changed = copy.deepcopy(data)
    mutate(changed)
    assert content_hash(Rubric.model_validate(changed)) != content_hash(Rubric.model_validate(data))


# --- RubricVersion integrity -----------------------------------------------------


def _version_kwargs(**overrides: Any) -> dict[str, Any]:
    rubric = make_rubric()
    kwargs: dict[str, Any] = {
        "rubric_id": rubric.id,
        "version": 1,
        "content_hash": content_hash(rubric),
        "message": "initial",
        "created_at": T0,
        "rubric": rubric,
    }
    kwargs.update(overrides)
    return kwargs


def test_version_accepts_matching_hash_and_normalises_time_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    version = RubricVersion(**_version_kwargs(created_at=T0.astimezone(ist)))
    assert version.created_at == T0
    assert version.created_at.utcoffset() == timedelta(0)
    assert version.label == f"sample-rubric@v1 ({version.content_hash[:12]})"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"content_hash": "a" * 64}, "does not match the document"),
        ({"content_hash": "not-hex"}, "should match pattern"),
        ({"rubric_id": "other"}, "does not match document id"),
        ({"version": 0}, "greater than or equal to 1"),
        ({"message": "   "}, "at least 1 character"),
        ({"created_at": datetime(2026, 9, 1)}, "timezone"),  # noqa: DTZ001 - naive on purpose
    ],
)
def test_version_rejects_inconsistent_fields(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        RubricVersion(**_version_kwargs(**overrides))


# --- registry --------------------------------------------------------------------


def _v2_document() -> Rubric:
    return make_rubric(
        criterion_dict("accuracy", weight=0.5, gating=3),
        criterion_dict("clarity", weight=0.3, low=0, high=2),
        criterion_dict("safety", weight=0.2, low=0, high=1),
        threshold=0.7,
    )


def test_publish_appends_numbered_versions_with_messages_and_times() -> None:
    clock = FrozenClock(T0)
    registry = RubricRegistry(clock)
    v1 = registry.publish(make_rubric(), "initial rubric")
    clock.advance(timedelta(days=2))
    v2 = registry.publish(_v2_document(), "add a safety criterion")

    assert (v1.version, v2.version) == (1, 2)
    assert v1.created_at == T0
    assert v2.created_at == T0 + timedelta(days=2)
    assert v2.message == "add a safety criterion"
    assert registry.history("sample-rubric") == (v1, v2)
    assert registry.latest("sample-rubric") == v2
    assert registry.get("sample-rubric", 1).rubric == make_rubric()
    assert registry.rubric_ids() == ("sample-rubric",)
    assert "sample-rubric" in registry
    assert "missing" not in registry
    assert len(registry) == 2


def test_history_is_a_snapshot_that_callers_cannot_mutate() -> None:
    registry = RubricRegistry(FrozenClock(T0))
    registry.publish(make_rubric(), "initial")
    history = registry.history("sample-rubric")
    assert isinstance(history, tuple)
    registry.publish(_v2_document(), "second")
    assert len(history) == 1
    assert len(registry.history("sample-rubric")) == 2


def test_publishing_the_head_again_is_refused() -> None:
    registry = RubricRegistry(FrozenClock(T0))
    registry.publish(make_rubric(), "initial")
    reordered = Rubric.model_validate(_reverse_keys(rubric_dict()))
    with pytest.raises(UnchangedRubricError, match=r"unchanged since sample-rubric@v1"):
        registry.publish(reordered, "same content, keys reordered")
    assert len(registry) == 1


def test_revert_creates_a_new_version_sharing_the_old_hash() -> None:
    registry = RubricRegistry(FrozenClock(T0))
    v1 = registry.publish(make_rubric(), "initial")
    registry.publish(_v2_document(), "tighten")
    v3 = registry.publish(make_rubric(), "revert to v1")
    assert v3.version == 3
    assert v3.content_hash == v1.content_hash
    assert registry.find_by_hash(v1.content_hash) == (v1, v3)
    assert registry.find_by_hash("f" * 64) == ()


def test_blank_changelog_message_is_rejected() -> None:
    registry = RubricRegistry(FrozenClock(T0))
    with pytest.raises(ValidationError, match="at least 1 character"):
        registry.publish(make_rubric(), "  ")
    assert len(registry) == 0


def test_rubrics_keep_independent_histories() -> None:
    registry = RubricRegistry(FrozenClock(T0))
    registry.publish(make_rubric(), "a")
    other = Rubric.model_validate({**rubric_dict(), "id": "other-rubric"})
    assert registry.publish(other, "b").version == 1
    assert registry.rubric_ids() == ("sample-rubric", "other-rubric")


def test_unknown_rubric_and_version_errors() -> None:
    registry = RubricRegistry(FrozenClock(T0))
    with pytest.raises(UnknownRubricError, match="no rubric 'nope'"):
        registry.latest("nope")
    registry.publish(make_rubric(), "initial")
    for bad in (0, 2):
        with pytest.raises(UnknownVersionError, match=f"versions 1..1, not {bad}"):
            registry.get("sample-rubric", bad)
    assert issubclass(UnknownRubricError, LookupError)


def test_registry_diff_between_versions() -> None:
    registry = RubricRegistry(FrozenClock(T0))
    registry.publish(make_rubric(), "initial")
    registry.publish(_v2_document(), "add safety")
    diff = registry.diff("sample-rubric", 1, 2)
    assert [ref.id for ref in diff.added] == ["safety"]
    assert diff.threshold is not None
    assert registry.diff("sample-rubric", 2, 2).is_empty


def test_default_clock_is_the_system_clock() -> None:
    before = datetime.now(UTC)
    version = RubricRegistry().publish(make_rubric(), "initial")
    assert before <= version.created_at <= datetime.now(UTC)
