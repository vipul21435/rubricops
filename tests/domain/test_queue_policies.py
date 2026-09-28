from __future__ import annotations

from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rubricops.domain.pipeline import Stage
from rubricops.domain.queue import (
    POLICY_NAMES,
    AssignmentPolicy,
    Exclusion,
    LoadBalanced,
    NoEligibleReviewer,
    QueueItem,
    Reviewer,
    RoundRobin,
    SkillTagMatch,
    assign_batch,
    exclusion,
    make_policy,
)

AUTHOR = 100


def _reviewers(*ids: int, **kwargs: object) -> list[Reviewer]:
    return [Reviewer(i, f"r{i}", **kwargs) for i in ids]  # type: ignore[arg-type]


def _item(sid: int = 1, **kwargs: object) -> QueueItem:
    return QueueItem(sid, AUTHOR, **kwargs)  # type: ignore[arg-type]


# -- shared eligibility rules -------------------------------------------------


def test_exclusion_rules_in_priority_order() -> None:
    qa = _item(stage=Stage.QA, primary_reviewer_id=2)
    assert exclusion(qa, Reviewer(AUTHOR, "a")) is Exclusion.AUTHOR
    assert exclusion(qa, Reviewer(2, "p")) is Exclusion.PRIMARY_REVIEWER
    assert exclusion(qa, Reviewer(3, "c", open_assignments=2, capacity=2)) is Exclusion.AT_CAPACITY
    assert exclusion(qa, Reviewer(4, "ok", open_assignments=1, capacity=2)) is None
    # The primary reviewer is only excluded from the QA stage.
    assert exclusion(_item(primary_reviewer_id=2), Reviewer(2, "p")) is None


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_every_policy_excludes_author_and_primary_reviewer(name: str) -> None:
    reviewers = [*_reviewers(1, 2, 3), Reviewer(AUTHOR, "author")]
    item = _item(stage=Stage.QA, primary_reviewer_id=1)
    for _ in range(6):
        chosen = make_policy(name).choose(item, reviewers)
        assert chosen.id not in {AUTHOR, 1}


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_every_policy_raises_a_typed_refusal_with_reasons(name: str) -> None:
    reviewers = [
        Reviewer(AUTHOR, "author"),
        Reviewer(1, "primary"),
        Reviewer(2, "full", open_assignments=1, capacity=1),
    ]
    item = _item(7, stage=Stage.QA, primary_reviewer_id=1)
    with pytest.raises(NoEligibleReviewer) as info:
        make_policy(name).choose(item, reviewers)
    assert dict(info.value.excluded) == {
        AUTHOR: Exclusion.AUTHOR,
        1: Exclusion.PRIMARY_REVIEWER,
        2: Exclusion.AT_CAPACITY,
    }
    assert info.value.policy == name
    assert "submission 7" in str(info.value)
    assert f"{AUTHOR}:author" in str(info.value)


def test_refusal_with_no_candidates_says_so() -> None:
    with pytest.raises(NoEligibleReviewer, match="no candidates"):
        LoadBalanced().choose(_item(), [])


def test_duplicate_reviewers_are_rejected() -> None:
    with pytest.raises(ValueError, match="listed twice"):
        LoadBalanced().choose(_item(), _reviewers(1, 1))
    with pytest.raises(ValueError, match="unique"):
        assign_batch(LoadBalanced(), [_item()], _reviewers(1, 1))


def test_value_objects_validate() -> None:
    with pytest.raises(ValueError, match="negative open load"):
        Reviewer(1, "r", open_assignments=-1)
    with pytest.raises(ValueError, match="capacity must be at least 1"):
        Reviewer(1, "r", capacity=0)
    with pytest.raises(ValueError, match="needs its primary_reviewer_id"):
        _item(stage=Stage.QA)
    with pytest.raises(ValueError, match="adjudication is done by a lead"):
        _item(stage=Stage.ADJUDICATION)


def test_make_policy() -> None:
    assert isinstance(make_policy("round-robin", cursor=3), RoundRobin)
    assert isinstance(make_policy("skill-match"), SkillTagMatch)
    assert isinstance(make_policy("load-balanced"), LoadBalanced)
    with pytest.raises(ValueError, match="only round-robin takes a cursor"):
        make_policy("load-balanced", cursor=1)
    with pytest.raises(ValueError, match="unknown policy 'random'"):
        make_policy("random")


# -- round robin --------------------------------------------------------------


def test_round_robin_walks_ids_from_the_cursor_and_wraps() -> None:
    policy = RoundRobin()
    reviewers = _reviewers(30, 10, 20)
    picks = [policy.choose(_item(i), reviewers).id for i in range(5)]
    assert picks == [10, 20, 30, 10, 20]
    assert policy.cursor == 20


def test_round_robin_cursor_survives_a_reviewer_leaving() -> None:
    policy = RoundRobin(cursor=20)
    # Reviewer 20 left; the next id after the cursor is still 30.
    assert policy.choose(_item(), _reviewers(10, 30)).id == 30


def test_round_robin_skips_ineligible_without_losing_its_place() -> None:
    policy = RoundRobin(cursor=1)
    reviewers = [*_reviewers(1, 3), Reviewer(2, "full", open_assignments=1, capacity=1)]
    assert policy.choose(_item(), reviewers).id == 3
    assert policy.choose(_item(), reviewers).id == 1


@given(n_reviewers=st.integers(1, 8), n_items=st.integers(0, 60))
def test_round_robin_is_fair(n_reviewers: int, n_items: int) -> None:
    reviewers = _reviewers(*range(1, n_reviewers + 1))
    result = assign_batch(RoundRobin(), [_item(i) for i in range(n_items)], reviewers)
    counts = Counter(d.reviewer.id for d in result.decisions if d.reviewer)
    per_reviewer = [counts.get(r.id, 0) for r in reviewers]
    assert sum(per_reviewer) == n_items
    assert max(per_reviewer) - min(per_reviewer) <= 1


# -- skill tags ---------------------------------------------------------------


def test_skill_match_requires_a_superset_of_tags() -> None:
    reviewers = [
        Reviewer(1, "py", frozenset({"python"})),
        Reviewer(2, "py-sql", frozenset({"python", "sql"}), open_assignments=3),
        Reviewer(3, "sql", frozenset({"sql"})),
    ]
    item = _item(skill_tags=frozenset({"python", "sql"}))
    assert SkillTagMatch().choose(item, reviewers).id == 2
    with pytest.raises(NoEligibleReviewer) as info:
        SkillTagMatch().choose(_item(skill_tags=frozenset({"rust"})), reviewers)
    assert set(info.value.excluded.values()) == {Exclusion.MISSING_SKILLS}


def test_skill_match_breaks_ties_by_load_then_id() -> None:
    tags = frozenset({"python"})
    reviewers = [
        Reviewer(5, "a", tags, open_assignments=2),
        Reviewer(4, "b", tags, open_assignments=1),
        Reviewer(3, "c", tags, open_assignments=1),
    ]
    assert SkillTagMatch().choose(_item(skill_tags=tags), reviewers).id == 3


def test_untagged_items_match_any_reviewer() -> None:
    assert SkillTagMatch().choose(_item(), _reviewers(9)).id == 9


# -- load balancing -----------------------------------------------------------


def test_load_balanced_prefers_the_lightest_load_then_id() -> None:
    reviewers = [
        Reviewer(1, "a", open_assignments=4),
        Reviewer(2, "b", open_assignments=2),
        Reviewer(3, "c", open_assignments=2),
    ]
    assert LoadBalanced().choose(_item(), reviewers).id == 2


def test_capacity_caps_hold_across_a_batch() -> None:
    reviewers = [Reviewer(1, "a", capacity=2), Reviewer(2, "b", capacity=1)]
    result = assign_batch(LoadBalanced(), [_item(i) for i in range(5)], reviewers)
    assert result.assigned == 3
    assert result.loads == {1: 2, 2: 1}
    refused = [d for d in result.decisions if d.reviewer is None]
    assert len(refused) == 2
    assert all(d.refusal is not None for d in refused)


@given(loads=st.lists(st.integers(0, 5), min_size=1, max_size=6), n_items=st.integers(0, 40))
def test_load_balanced_levels_loads(loads: list[int], n_items: int) -> None:
    reviewers = [Reviewer(i, f"r{i}", open_assignments=load) for i, load in enumerate(loads)]
    result = assign_batch(LoadBalanced(), [_item(1000 + i) for i in range(n_items)], reviewers)
    final = list(result.loads.values())
    assert sum(final) == sum(loads) + n_items
    # Once the batch has lifted every reviewer to the old maximum, loads stay within one.
    if n_items >= sum(max(loads) - load for load in loads):
        assert max(final) - min(final) <= 1
    # Nobody is ever given work while a lighter eligible reviewer is waiting.
    assert max(final) <= max(*loads, min(final) + 1)


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_batch_never_breaks_independence(name: str) -> None:
    policy: AssignmentPolicy = make_policy(name)
    reviewers = _reviewers(1, 2, 3, AUTHOR)
    items = [_item(i, stage=Stage.QA, primary_reviewer_id=1 + i % 3) for i in range(12)]
    result = assign_batch(policy, items, reviewers)
    assert result.assigned == 12
    for decision in result.decisions:
        assert decision.reviewer is not None
        assert decision.reviewer.id not in {AUTHOR, decision.item.primary_reviewer_id}
