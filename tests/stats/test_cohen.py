"""Cohen's kappa: a published example, hand-computed cases, degenerate data, properties."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rubricops.stats.agreement import KappaWeights, as_reliability_data, cohen_kappa
from rubricops.stats.data import AgreementInputError, ReliabilityData

PROPERTY_SETTINGS = settings(max_examples=200, deadline=None)
ALL_WEIGHTS: tuple[KappaWeights, ...] = ("none", "linear", "quadratic")
SCALED_WEIGHTS: tuple[KappaWeights, ...] = ("linear", "quadratic")


def test_published_example_two_readers_fifty_proposals() -> None:
    """Wikipedia, "Cohen's kappa", section "Simple example" (retrieved 2026-09).

    Two readers each say Yes or No to 50 grant proposals: both Yes 20, A Yes and
    B No 5, A No and B Yes 10, both No 15. p_o = 0.7, p_e = 0.5, kappa = 0.4.
    """
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    result = cohen_kappa(ReliabilityData.from_columns(a, b))
    assert result.value == pytest.approx(0.4)
    assert 1 - result.observed_disagreement == pytest.approx(0.7)
    assert 1 - result.expected_disagreement == pytest.approx(0.5)
    assert (result.n_units, result.n_raters, result.n_ratings) == (50, 2, 100)
    assert result.defined
    assert result.metric == "cohen"


# Hand-computed: a = [1, 1, 2, 3], b = [1, 2, 2, 1] on the scale {1, 2, 3}.
# Marginals pa = [1/2, 1/4, 1/4], pb = [1/2, 1/2, 0].
HAND_A = [1, 1, 2, 3]
HAND_B = [1, 2, 2, 1]


@pytest.mark.parametrize(
    ("weights", "observed", "expected", "kappa"),
    [
        # 2 of 4 pairs agree: D_o = 1/2; chance agreement 1/4 + 1/8 = 3/8, D_e = 5/8.
        ("none", 1 / 2, 5 / 8, 1 / 5),
        # w = |i - j| / 2: D_o = (0 + 1/2 + 0 + 1) / 4 = 3/8;
        # D_e = 1/2 * 1/4 + 1/4 * 1/4 + 1/4 * 3/4 = 3/8, so kappa = 0.
        ("linear", 3 / 8, 3 / 8, 0.0),
        # w = ((i - j) / 2)^2: D_o = (0 + 1/4 + 0 + 1) / 4 = 5/16;
        # D_e = 1/2 * 1/8 + 1/4 * 1/8 + 1/4 * 5/8 = 1/4, so kappa = 1 - 5/4.
        ("quadratic", 5 / 16, 1 / 4, -1 / 4),
    ],
)
def test_hand_computed_small_case(
    weights: KappaWeights, observed: float, expected: float, kappa: float
) -> None:
    """Worked by hand; the arithmetic is in the comments on each parameter set."""
    result = cohen_kappa(ReliabilityData.from_columns(HAND_A, HAND_B), weights=weights)
    assert result.observed_disagreement == pytest.approx(observed)
    assert result.expected_disagreement == pytest.approx(expected)
    assert result.value == pytest.approx(kappa)
    assert result.metric == ("cohen" if weights == "none" else f"cohen-{weights}")


def test_quadratic_kappa_equals_lins_ccc_on_the_hand_case() -> None:
    """Lin's CCC = 2 cov / (var_a + var_b + (mean_a - mean_b)^2) with population moments.

    For HAND_A/HAND_B: cov = -1/8, var_a = 11/16, var_b = 1/4, mean gap 1/4, so the
    CCC is (-1/4) / 1 = -1/4, the quadratic kappa above.
    """
    assert _lins_ccc(HAND_A, HAND_B) == pytest.approx(-0.25)


def test_weights_use_rating_values_not_ranks() -> None:
    """Unused scale points between the observed ones change nothing: weights are by value."""
    a, b = [1, 1, 4, 4, 1], [1, 4, 4, 1, 1]
    for weights in SCALED_WEIGHTS:
        inferred = cohen_kappa(ReliabilityData.from_columns(a, b), weights=weights)
        full_scale = cohen_kappa(
            ReliabilityData.from_columns(a, b, categories=[1, 2, 3, 4]), weights=weights
        )
        assert inferred.value == pytest.approx(full_scale.value)


def test_a_near_miss_costs_less_than_a_far_miss_when_weighted() -> None:
    near = cohen_kappa([[1, 1], [2, 2], [3, 3], [4, 3]], weights="linear", categories=[1, 2, 3, 4])
    far = cohen_kappa([[1, 1], [2, 2], [3, 3], [4, 1]], weights="linear", categories=[1, 2, 3, 4])
    unweighted_near = cohen_kappa([[1, 1], [2, 2], [3, 3], [4, 3]], categories=[1, 2, 3, 4])
    unweighted_far = cohen_kappa([[1, 1], [2, 2], [3, 3], [4, 1]], categories=[1, 2, 3, 4])
    assert near.value > far.value
    assert unweighted_near.value == pytest.approx(unweighted_far.value)


def test_units_missing_either_rating_are_dropped() -> None:
    result = cohen_kappa([["a", "a"], ["b", "b"], ["a", None], [None, "b"], ["b", "a"]])
    assert result.n_units == 3
    assert result.n_ratings == 6
    assert result.value == pytest.approx(cohen_kappa([["a", "a"], ["b", "b"], ["b", "a"]]).value)


def test_perfect_agreement_over_two_categories_is_exactly_one() -> None:
    result = cohen_kappa([["pass", "pass"], ["fail", "fail"], ["pass", "pass"]])
    assert result.value == 1.0
    assert result.observed_disagreement == 0.0


def test_opposite_raters_give_minus_one() -> None:
    result = cohen_kappa([[0, 1], [1, 0], [0, 1], [1, 0]])
    assert result.value == pytest.approx(-1.0)


@pytest.mark.parametrize("weights", ["none", "linear", "quadratic"])
def test_single_category_is_nan_with_a_reason(weights: KappaWeights) -> None:
    """Both reviewers pass everything: perfect agreement, but 0/0 after chance correction."""
    result = cohen_kappa([[5, 5], [5, 5], [5, None]], weights=weights)
    assert math.isnan(result.value)
    assert not result.defined
    assert result.reason is not None
    assert "every usable rating is 5" in result.reason
    assert "single category" in result.reason
    assert result.expected_disagreement == 0.0
    assert result.n_units == 2


def test_no_complete_pair_is_nan_with_a_reason() -> None:
    result = cohen_kappa([["a", None], [None, "b"]])
    assert math.isnan(result.value)
    assert result.reason == "no unit has a rating from both raters"
    assert math.isnan(result.observed_disagreement)
    assert math.isnan(result.expected_disagreement)
    assert (result.n_units, result.n_ratings) == (0, 0)


@pytest.mark.parametrize("rows", [[[1, 2, 3]], [[1]], []])
def test_needs_exactly_two_raters(rows: list[list[int]]) -> None:
    with pytest.raises(AgreementInputError, match="exactly two raters"):
        cohen_kappa(rows)


def test_weighted_kappa_needs_numeric_ratings() -> None:
    with pytest.raises(AgreementInputError, match="needs numeric ratings"):
        cohen_kappa([["low", "high"], ["high", "high"]], weights="linear")


def test_categories_cannot_be_given_twice() -> None:
    data = ReliabilityData.from_rows([[1, 2]])
    assert as_reliability_data(data) is data
    with pytest.raises(AgreementInputError, match="fixed when ReliabilityData is built"):
        cohen_kappa(data, categories=[1, 2])


# --- properties --------------------------------------------------------------------


def _lins_ccc(a: list[int], b: list[int]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    return 2 * cov / (float(x.var()) + float(y.var()) + float(x.mean() - y.mean()) ** 2)


@st.composite
def rater_pairs(draw: st.DrawFn, *, min_categories: int = 2) -> tuple[list[int], list[int]]:
    """Two raters on a 0..k-1 scale who each use at least ``min_categories`` values."""
    k = draw(st.integers(min_categories, 6))
    n = draw(st.integers(k, 30))
    a = draw(st.lists(st.integers(0, k - 1), min_size=n, max_size=n))
    b = draw(st.lists(st.integers(0, k - 1), min_size=n, max_size=n))
    a[:k] = list(range(k))  # every category occurs, so chance disagreement is positive
    return a, b


@PROPERTY_SETTINGS
@given(rater_pairs(), st.sampled_from(ALL_WEIGHTS))
def test_identical_raters_give_kappa_one(
    pair: tuple[list[int], list[int]], weights: KappaWeights
) -> None:
    a, _ = pair
    result = cohen_kappa(ReliabilityData.from_columns(a, a), weights=weights)
    assert result.value == 1.0


@PROPERTY_SETTINGS
@given(rater_pairs(), st.permutations(range(6)))
def test_unweighted_kappa_is_invariant_under_relabelling(
    pair: tuple[list[int], list[int]], permutation: list[int]
) -> None:
    a, b = pair
    names = [f"label-{p}" for p in permutation]
    original = cohen_kappa(ReliabilityData.from_columns(a, b))
    relabelled = cohen_kappa(
        ReliabilityData.from_columns([names[x] for x in a], [names[x] for x in b])
    )
    assert relabelled.value == pytest.approx(original.value, abs=1e-12)


@PROPERTY_SETTINGS
@given(rater_pairs(), st.sampled_from(ALL_WEIGHTS))
def test_kappa_is_symmetric_and_bounded(
    pair: tuple[list[int], list[int]], weights: KappaWeights
) -> None:
    a, b = pair
    ab = cohen_kappa(ReliabilityData.from_columns(a, b), weights=weights)
    ba = cohen_kappa(ReliabilityData.from_columns(b, a), weights=weights)
    assert ab.value == pytest.approx(ba.value, abs=1e-12)
    assert ab.value <= 1.0 + 1e-12
    assert ab.value >= -1.0 - 1e-12


@PROPERTY_SETTINGS
@given(rater_pairs())
def test_quadratic_kappa_equals_lins_concordance_correlation(
    pair: tuple[list[int], list[int]],
) -> None:
    """Robieson (1999); see also Warrens (2012), "Some paradoxical results for the
    quadratically weighted kappa", Psychometrika 77(2): value-weighted quadratic
    kappa is Lin's concordance correlation coefficient with population moments.
    """
    a, b = pair
    kappa = cohen_kappa(ReliabilityData.from_columns(a, b), weights="quadratic").value
    assert kappa == pytest.approx(_lins_ccc(a, b), abs=1e-9)


@PROPERTY_SETTINGS
@given(rater_pairs(), st.integers(1, 7), st.integers(-10, 10))
def test_weighted_kappa_is_invariant_under_affine_rescaling(
    pair: tuple[list[int], list[int]], scale: int, shift: int
) -> None:
    a, b = pair
    for weights in SCALED_WEIGHTS:
        original = cohen_kappa(ReliabilityData.from_columns(a, b), weights=weights)
        moved = cohen_kappa(
            ReliabilityData.from_columns(
                [scale * x + shift for x in a], [scale * x + shift for x in b]
            ),
            weights=weights,
        )
        assert moved.value == pytest.approx(original.value, abs=1e-9)
