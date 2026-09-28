"""Fleiss' kappa: a published example, hand-computed cases, degenerate data, properties."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rubricops.stats.agreement import cohen_kappa, fleiss_kappa
from rubricops.stats.data import AgreementInputError, ReliabilityData

PROPERTY_SETTINGS = settings(max_examples=200, deadline=None)

# Wikipedia, "Fleiss' kappa", section "Worked example" (retrieved 2026-09):
# 10 subjects, 14 raters each, 5 categories; one row of category counts per subject.
WIKIPEDIA_COUNTS = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4, 2],
    [0, 0, 3, 5, 6],
    [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0],
    [3, 2, 6, 3, 0],
    [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0],
    [0, 2, 2, 3, 7],
]


def test_published_example_fourteen_raters_ten_subjects() -> None:
    """Wikipedia, "Fleiss' kappa", worked example: P-bar = 0.378, P-bar_e = 0.213,
    kappa = 0.210 (all rounded to three decimals in the source).
    """
    data = ReliabilityData.from_counts(WIKIPEDIA_COUNTS, categories=[1, 2, 3, 4, 5])
    result = fleiss_kappa(data)
    assert result.value == pytest.approx(0.210, abs=5e-4)
    assert 1 - result.observed_disagreement == pytest.approx(0.378, abs=5e-4)
    assert 1 - result.expected_disagreement == pytest.approx(0.213, abs=5e-4)
    assert (result.n_units, result.n_raters, result.n_ratings) == (10, 14, 140)
    assert result.metric == "fleiss"


def test_hand_computed_three_units_three_raters() -> None:
    """Units [a,a,a], [a,a,b], [b,b,b]; m = 3.

    Per-unit agreement P_u = (sum n_uj^2 - m) / (m (m - 1)) = 1, 1/3, 1, so
    P-bar = 7/9. Pooled p = (5/9, 4/9), P-bar_e = 41/81.
    kappa = (7/9 - 41/81) / (1 - 41/81) = 22/40 = 0.55.
    """
    result = fleiss_kappa([["a", "a", "a"], ["a", "a", "b"], ["b", "b", "b"]])
    assert result.value == pytest.approx(0.55)
    assert result.observed_disagreement == pytest.approx(2 / 9)
    assert result.expected_disagreement == pytest.approx(40 / 81)


def test_two_raters_give_scotts_pi() -> None:
    """Scott (1955): pi uses the pooled marginals of both raters.

    For the 50-proposal example in test_cohen: p_o = 0.7; pooled "yes" is
    (25 + 30) / 100 = 0.55, so p_e = 0.55^2 + 0.45^2 = 0.505 and
    pi = (0.7 - 0.505) / 0.495 = 0.3939...
    """
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    data = ReliabilityData.from_columns(a, b)
    assert fleiss_kappa(data).value == pytest.approx(0.195 / 0.495)
    assert cohen_kappa(data).value == pytest.approx(0.4)


def test_missing_cells_are_fine_when_every_unit_has_m_ratings() -> None:
    """Fleiss' design lets each unit be rated by a different subset of raters."""
    sparse = fleiss_kappa([["a", "a", None], [None, "b", "b"], ["a", None, "b"]])
    dense = fleiss_kappa([["a", "a"], ["b", "b"], ["a", "b"]])
    assert sparse.value == pytest.approx(dense.value)
    assert sparse.n_units == 3
    assert sparse.n_ratings == 6


def test_units_without_ratings_are_ignored() -> None:
    result = fleiss_kappa([["a", "b"], [None, None], ["b", "b"]])
    assert result.n_units == 2
    assert result.value == pytest.approx(fleiss_kappa([["a", "b"], ["b", "b"]]).value)


def test_uneven_rating_counts_are_rejected() -> None:
    with pytest.raises(AgreementInputError, match=r"found \[2, 3\].*Krippendorff"):
        fleiss_kappa([["a", "a", "b"], ["a", None, "b"]])


def test_one_rating_per_unit_is_rejected() -> None:
    with pytest.raises(AgreementInputError, match="at least two ratings per unit"):
        fleiss_kappa([["a", None], [None, "b"]])


def test_single_category_is_nan_with_a_reason() -> None:
    result = fleiss_kappa([["pass"] * 4, ["pass"] * 4])
    assert math.isnan(result.value)
    assert result.reason is not None
    assert "every usable rating is 'pass'" in result.reason
    assert result.observed_disagreement == 0.0
    assert result.expected_disagreement == 0.0


def test_no_rated_unit_is_nan_with_a_reason() -> None:
    result = fleiss_kappa([[None, None]])
    assert math.isnan(result.value)
    assert result.reason == "no unit has any ratings"
    assert result.n_units == 0


def test_perfect_agreement_over_two_categories_is_exactly_one() -> None:
    assert fleiss_kappa([["a"] * 5, ["b"] * 5, ["a"] * 5]).value == 1.0


def test_from_counts_round_trips_and_validates() -> None:
    data = ReliabilityData.from_counts([[2, 0, 1], [0, 3, 0]], categories=["x", "y", "z"])
    assert data.unit_counts().tolist() == [[2, 0, 1], [0, 3, 0]]
    assert data.categories == ("x", "y", "z")
    padded = ReliabilityData.from_counts(np.array([[1, 1], [2, 1]]))
    assert padded.n_raters == 3
    assert padded.categories == (0, 1)
    assert ReliabilityData.from_counts(np.zeros((0, 2), dtype=np.int64)).n_units == 0


@pytest.mark.parametrize(
    ("counts", "categories", "message"),
    [
        ([1, 2], None, "units x categories table"),
        ([[1.5, 2.0]], None, "units x categories table"),
        ([[1, -1]], None, "units x categories table"),
        ([[1, 1]], ["a"], "1 category names for 2 count columns"),
    ],
)
def test_from_counts_rejects_bad_tables(
    counts: Any, categories: list[str] | None, message: str
) -> None:
    with pytest.raises(AgreementInputError, match=message):
        ReliabilityData.from_counts(counts, categories=categories)


# --- properties --------------------------------------------------------------------


@st.composite
def count_tables(draw: st.DrawFn) -> list[list[int]]:
    """n units x k categories, m ratings per unit, at least two categories used."""
    k = draw(st.integers(2, 5))
    m = draw(st.integers(2, 7))
    n = draw(st.integers(2, 15))
    rows = []
    for _ in range(n):
        cuts = sorted(draw(st.lists(st.integers(0, m), min_size=k - 1, max_size=k - 1)))
        edges = [0, *cuts, m]
        rows.append([edges[i + 1] - edges[i] for i in range(k)])
    rows[0] = [m] + [0] * (k - 1)
    rows[1] = [0, m] + [0] * (k - 2)
    return rows


@PROPERTY_SETTINGS
@given(count_tables(), st.permutations(range(5)))
def test_invariant_under_relabelling_and_column_order(
    counts: list[list[int]], permutation: list[int]
) -> None:
    k = len(counts[0])
    order = [p for p in permutation if p < k]
    original = fleiss_kappa(ReliabilityData.from_counts(counts))
    shuffled = [[row[j] for j in order] for row in counts]
    relabelled = fleiss_kappa(
        ReliabilityData.from_counts(shuffled, categories=[f"c{j}" for j in order])
    )
    assert relabelled.value == pytest.approx(original.value, abs=1e-12)


@PROPERTY_SETTINGS
@given(count_tables())
def test_unanimous_units_give_kappa_one(counts: list[list[int]]) -> None:
    m = sum(counts[0])
    unanimous = [[m if j == int(np.argmax(row)) else 0 for j in range(len(row))] for row in counts]
    assert fleiss_kappa(ReliabilityData.from_counts(unanimous)).value == 1.0


@PROPERTY_SETTINGS
@given(count_tables())
def test_kappa_is_at_most_one_and_at_least_the_minimum(counts: list[list[int]]) -> None:
    """Fleiss (1971): kappa lies in [-1 / (m - 1), 1]."""
    m = sum(counts[0])
    value = fleiss_kappa(ReliabilityData.from_counts(counts)).value
    assert -1 / (m - 1) - 1e-12 <= value <= 1.0 + 1e-12
