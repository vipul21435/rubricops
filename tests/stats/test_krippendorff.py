"""Krippendorff's alpha: published examples, hand-computed cases, missing data, properties."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rubricops.stats.agreement import (
    AlphaLevel,
    coincidence_matrix,
    fleiss_kappa,
    krippendorff_alpha,
)
from rubricops.stats.data import AgreementInputError, ReliabilityData

PROPERTY_SETTINGS = settings(max_examples=200, deadline=None)
LEVELS: tuple[AlphaLevel, ...] = ("nominal", "interval")

# Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. Annenberg School
# for Communication, University of Pennsylvania. Section C: four observers, twelve
# units, values 1-5, missing data. One list per observer (A, B, C, D); None = missing.
_ = None
KRIPPENDORFF_2011 = ReliabilityData.from_columns(
    [1, 2, 3, 3, 2, 1, 4, 1, 2, _, _, _],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, _, 3],
    [_, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, _],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, _],
)


def test_published_example_nominal() -> None:
    """Krippendorff (2011), section C: nominal alpha = 0.743.

    Unit 12 has one value and is dropped, leaving n = 40 pairable values. From the
    coincidence matrix: the off-diagonal sum is 8 and sum(n_c^2) = 81 + 169 + 100 + 25
    + 9 = 384, so alpha = 1 - (n - 1) * 8 / (n^2 - 384) = 1 - 312 / 1216 = 0.74342.
    """
    result = krippendorff_alpha(KRIPPENDORFF_2011, level="nominal")
    assert round(result.value, 3) == 0.743
    assert result.value == pytest.approx(1 - 312 / 1216)
    assert (result.n_units, result.n_raters, result.n_ratings) == (11, 4, 40)
    assert result.metric == "alpha-nominal"


def test_published_example_interval() -> None:
    """Krippendorff (2011), section C, same data: interval alpha = 0.849."""
    result = krippendorff_alpha(KRIPPENDORFF_2011, level="interval")
    assert round(result.value, 3) == 0.849
    assert result.metric == "alpha-interval"
    assert result.n_ratings == 40


def test_published_coincidence_matrix() -> None:
    """Krippendorff (2011), section C: the coincidence matrix and its margins.

    Rows and columns are the values 1-5. For example unit 6 (values 1, 2, 3, 4) adds
    1/3 to each off-diagonal cell among 1-4, and unit 8 (1, 1, 2, 1) adds 2 to o_11
    and 1 to o_12 and o_21.
    """
    o = coincidence_matrix(KRIPPENDORFF_2011)
    expected = np.array(
        [
            [7, 4 / 3, 1 / 3, 1 / 3, 0],
            [4 / 3, 10, 4 / 3, 1 / 3, 0],
            [1 / 3, 4 / 3, 8, 1 / 3, 0],
            [1 / 3, 1 / 3, 1 / 3, 4, 0],
            [0, 0, 0, 0, 3],
        ]
    )
    np.testing.assert_allclose(o, expected)
    np.testing.assert_allclose(o.sum(axis=1), [9, 13, 10, 5, 3])


def test_published_example_binary_two_observers() -> None:
    """Krippendorff (2011), section A: two observers, ten units, binary data.

    n_0 = 14, n_1 = 6, o_01 = 4, so alpha = 1 - (n - 1) * o_01 / (n_0 * n_1)
    = 1 - 19 * 4 / 84 = 0.095.
    """
    result = krippendorff_alpha(
        ReliabilityData.from_columns([0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    )
    assert round(result.value, 3) == 0.095
    assert result.value == pytest.approx(1 - 76 / 84)


def test_hand_computed_interval_case() -> None:
    """Units (1, 2), (3, 3) and (2, None); the last one is not pairable.

    n = 4 pairable values. Coincidences: o_12 = o_21 = 1, o_33 = 2, so n_1 = n_2 = 1
    and n_3 = 2. D_o = (1 * 1 + 1 * 1) / 4 = 1/2. Over unordered pairs of values,
    n_c * n_k * (c - k)^2 is 1 for (1, 2), 8 for (1, 3) and 2 for (2, 3); doubled for
    order that is 22, so D_e = 22 / (4 * 3) = 11/6 and alpha = 1 - 3/11 = 8/11.
    """
    result = krippendorff_alpha([[1, 2], [3, 3], [2, None]], level="interval")
    assert result.observed_disagreement == pytest.approx(1 / 2)
    assert result.expected_disagreement == pytest.approx(11 / 6)
    assert result.value == pytest.approx(8 / 11)
    assert result.n_units == 2


def test_hand_computed_nominal_case_with_varying_raters() -> None:
    """Units (a, a, b) and (b, b), plus (a) which is dropped.

    Unit 1 (m = 3): o_aa += 2 * 1 / 2 = 1, o_ab = o_ba += 2 / 2 = 1. Unit 2 (m = 2):
    o_bb += 2. n_a = 2, n_b = 3, n = 5. D_o = 2 / 5; D_e = 2 * 2 * 3 / (5 * 4) = 3/5.
    alpha = 1 - (2/5) / (3/5) = 1/3.
    """
    result = krippendorff_alpha([["a", "a", "b"], ["b", "b", None], ["a", None, None]])
    assert result.value == pytest.approx(1 / 3)
    assert (result.n_units, result.n_ratings) == (2, 5)


def test_units_with_one_rating_are_dropped_not_counted() -> None:
    base = [[1, 2, 2], [3, 3, None], [1, 1, 1]]
    padded = [*base, [5, None, None], [None, None, None]]
    for level in LEVELS:
        assert krippendorff_alpha(padded, level=level).value == pytest.approx(
            krippendorff_alpha(base, level=level).value
        )


@pytest.mark.parametrize("level", LEVELS)
def test_single_value_is_nan_with_a_reason(level: AlphaLevel) -> None:
    result = krippendorff_alpha([[4, 4, None], [4, 4, 4], [2, None, None]], level=level)
    assert math.isnan(result.value)
    assert result.reason is not None
    assert "every usable rating is 4" in result.reason
    assert result.expected_disagreement == 0.0


def test_no_pairable_unit_is_nan_with_a_reason() -> None:
    result = krippendorff_alpha([[1, None], [None, 2]])
    assert math.isnan(result.value)
    assert result.reason == "no unit has two or more ratings, so no rating can be paired"
    assert result.n_units == 0
    assert coincidence_matrix(ReliabilityData.from_rows([[1, None]])).tolist() == [[0.0]]


def test_interval_level_needs_numeric_ratings() -> None:
    with pytest.raises(AgreementInputError, match="needs numeric ratings"):
        krippendorff_alpha([["low", "high"]], level="interval")


def test_interval_credits_near_misses_that_nominal_does_not() -> None:
    near = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 4]]
    far = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 1]]
    assert (
        krippendorff_alpha(near, level="interval").value
        > krippendorff_alpha(far, level="interval").value
    )
    assert krippendorff_alpha(near).value == pytest.approx(krippendorff_alpha(far).value)


# --- properties --------------------------------------------------------------------


@st.composite
def sparse_tables(draw: st.DrawFn) -> list[list[int | None]]:
    """Units x raters on 0..k-1 with missing cells; two units guarantee >= 2 values."""
    k = draw(st.integers(2, 5))
    raters = draw(st.integers(2, 5))
    n = draw(st.integers(2, 15))
    cell = st.none() | st.integers(0, k - 1)
    rows = [draw(st.lists(cell, min_size=raters, max_size=raters)) for _ in range(n)]
    rows[0] = [0, 0] + [None] * (raters - 2)
    rows[1] = [k - 1] * raters
    return rows


@st.composite
def complete_tables(draw: st.DrawFn) -> list[list[int]]:
    k = draw(st.integers(2, 5))
    raters = draw(st.integers(2, 5))
    n = draw(st.integers(2, 15))
    rows = [
        draw(st.lists(st.integers(0, k - 1), min_size=raters, max_size=raters)) for _ in range(n)
    ]
    rows[0] = [0] * raters
    rows[1] = [k - 1] * raters
    return rows


@PROPERTY_SETTINGS
@given(complete_tables())
def test_nominal_alpha_is_fleiss_kappa_with_a_small_sample_correction(
    rows: list[list[int]],
) -> None:
    """With complete data, alpha = 1 - (1 - kappa_Fleiss) * (n - 1) / n, n = units * m.

    Follows from the coincidence-matrix form: D_o = 1 - P-bar and
    D_e = n (1 - P-bar_e) / (n - 1).
    """
    n = len(rows) * len(rows[0])
    kappa = fleiss_kappa(rows).value
    alpha = krippendorff_alpha(rows).value
    assert alpha == pytest.approx(1 - (1 - kappa) * (n - 1) / n, abs=1e-12)


@PROPERTY_SETTINGS
@given(sparse_tables(), st.permutations(range(5)))
def test_nominal_alpha_is_invariant_under_relabelling(
    rows: list[list[int | None]], permutation: list[int]
) -> None:
    names = [f"label-{p}" for p in permutation]
    relabelled = [[None if v is None else names[v] for v in row] for row in rows]
    assert krippendorff_alpha(relabelled).value == pytest.approx(
        krippendorff_alpha(rows).value, abs=1e-12
    )


@PROPERTY_SETTINGS
@given(sparse_tables(), st.sampled_from([-3, -1, 2, 5]), st.integers(-10, 10))
def test_interval_alpha_is_invariant_under_affine_rescaling(
    rows: list[list[int | None]], scale: int, shift: int
) -> None:
    moved = [[None if v is None else scale * v + shift for v in row] for row in rows]
    assert krippendorff_alpha(moved, level="interval").value == pytest.approx(
        krippendorff_alpha(rows, level="interval").value, abs=1e-9
    )


@PROPERTY_SETTINGS
@given(sparse_tables(), st.randoms(use_true_random=False))
def test_alpha_ignores_which_rater_gave_a_rating(
    rows: list[list[int | None]], rng: random.Random
) -> None:
    shuffled = [row[:] for row in rows]
    for row in shuffled:
        rng.shuffle(row)
    for level in LEVELS:
        assert krippendorff_alpha(shuffled, level=level).value == pytest.approx(
            krippendorff_alpha(rows, level=level).value, abs=1e-12
        )


@PROPERTY_SETTINGS
@given(sparse_tables())
def test_unanimous_units_give_alpha_one(rows: list[list[int | None]]) -> None:
    unanimous = [
        [None if v is None else next(x for x in row if x is not None) for v in row] for row in rows
    ]
    for level in LEVELS:
        assert krippendorff_alpha(unanimous, level=level).value == 1.0


@PROPERTY_SETTINGS
@given(sparse_tables())
def test_binary_interval_equals_nominal(rows: list[list[int | None]]) -> None:
    """With two values 0 and 1, (c - k)^2 = 1 exactly when c != k."""
    binary = [[None if v is None else min(v, 1) for v in row] for row in rows]
    assert krippendorff_alpha(binary, level="interval").value == pytest.approx(
        krippendorff_alpha(binary).value, abs=1e-12
    )
