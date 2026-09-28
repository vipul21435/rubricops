"""Gwet's AC1/AC2 agreement coefficients, against published and hand-computed values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rubricops.stats.agreement import cohen_kappa, gwet_agreement
from rubricops.stats.data import AgreementInputError, ReliabilityData

if TYPE_CHECKING:
    from rubricops.stats.agreement import KappaWeights

# The published 3x3 contingency table from Gwet's irrCAC package (cont3x3abstractors):
# 100 abstracts rated Ectopic/AIU/NIU by two abstractors, 89 agreeing. irrCAC reports
# AC1 = 0.8493305 and percent agreement 0.89.
ABSTRACTORS: list[list[str]] = (
    [["Ectopic", "Ectopic"]] * 13
    + [["AIU", "AIU"]] * 20
    + [["AIU", "NIU"]] * 7
    + [["NIU", "AIU"]] * 4
    + [["NIU", "NIU"]] * 56
)


def test_published_example_matches_irrcac() -> None:
    result = gwet_agreement(ReliabilityData.from_rows(ABSTRACTORS))
    assert result.metric == "ac1"
    assert result.value == pytest.approx(0.8493305)
    # percent agreement is 89/100, so observed disagreement is 0.11.
    assert result.observed_disagreement == pytest.approx(0.11)
    assert result.n_units == 100
    assert result.defined


def test_hand_computed_small_case() -> None:
    # Three units, two raters: u1 agrees (1,1), u2 agrees (2,2), u3 disagrees (1,2).
    # p_a = 2/3. Category proportions 1: 3/6=0.5, 2: 3/6=0.5, so
    # p_e = 2*(0.5*0.5 + 0.5*0.5)/(2*1) = 0.5, AC1 = (2/3 - 0.5)/(1 - 0.5) = 1/3.
    d = ReliabilityData.from_rows([[1, 1], [2, 2], [1, 2]])
    result = gwet_agreement(d)
    assert result.value == pytest.approx(1 / 3)
    assert result.defined


def test_perfect_agreement_over_two_categories_is_exactly_one() -> None:
    d = ReliabilityData.from_rows([[1, 1], [2, 2]])
    assert gwet_agreement(d).value == 1.0
    assert gwet_agreement(d, weights="linear").value == 1.0
    assert gwet_agreement(d, weights="quadratic").value == 1.0


def test_kappa_paradox_ac1_stays_sane() -> None:
    """The Feinstein-Cicchetti imbalance case: raters agree 90% of the time but
    kappa collapses toward zero because the marginal distribution is 95:5. AC1's
    chance term uses the overall proportions, so it reports the agreement that is
    actually there instead of the paradoxically near-zero kappa."""
    rows = [["A", "A"]] * 90 + [["A", "B"]] * 5 + [["B", "A"]] * 5
    d = ReliabilityData.from_rows(rows)
    kappa = cohen_kappa(d)
    ac1 = gwet_agreement(d)
    assert kappa.observed_disagreement == pytest.approx(0.10)  # p_a = 0.90
    assert kappa.value == pytest.approx(-0.0526, abs=1e-3)  # the paradox
    assert ac1.value == pytest.approx(0.89, abs=2e-2)
    assert ac1.value > 0.5  # reads as agreement, unlike kappa


def test_weighted_ac2_rewards_near_misses_more_than_far_misses() -> None:
    # Two units disagree by 1 point, two by 3 points on a 1..4 scale.
    rows = [[1, 2], [2, 3], [1, 4], [1, 4]]  # two |Δ|=1, two |Δ|=3, span 3
    d = ReliabilityData.from_rows(rows)
    nominal = gwet_agreement(d)
    linear = gwet_agreement(d, weights="linear")
    quadratic = gwet_agreement(d, weights="quadratic")
    # Quadratic penalises |Δ|=3 far more than linear, so its AC2 is lower.
    assert quadratic.value < linear.value < nominal.value
    assert quadratic.defined


def test_weighted_ac2_matches_a_hand_double_rating_case() -> None:
    # Two raters, two units on a declared 1..3 scale. Unit1 is a 1-point miss (1 vs 2),
    # unit2 agrees on 1. Quadratic weights (span 2): w[1<->2] = 1-(1/2)^2 = 0.75.
    #   pa: unit1 = 0.75, unit2 = 1.00 -> mean 0.875
    #   p_k = [0.75, 0.25, 0]; w_sum (3x3 quadratic matrix) = 6.0
    #   pe = 6.0 * (0.75*0.25 + 0.25*0.75) / (3*2) = 6*0.375/6 = 0.375
    #   AC2 = (0.875 - 0.375) / (1 - 0.375) = 0.8
    d = ReliabilityData.from_rows([[1, 2], [1, 1]], categories=[1, 2, 3])
    result = gwet_agreement(d, weights="quadratic")
    assert result.defined
    assert result.value == pytest.approx(0.8)


def test_weighted_ac2_hand_case_without_weights_differs() -> None:
    # Same data but unweighted: unit1 is a full disagreement, pa = 0.5.
    d = ReliabilityData.from_rows([[1, 2], [1, 1]], categories=[1, 2, 3])
    ac2 = gwet_agreement(d, weights="quadratic")
    ac1 = gwet_agreement(d)
    assert ac1.observed_disagreement == pytest.approx(0.5)
    assert ac2.value > ac1.value  # the near miss is worth more than nothing


def test_weighted_ac2_needs_numeric_ratings() -> None:
    d = ReliabilityData.from_rows([["low", "low"], ["mid", "high"]])
    with pytest.raises(AgreementInputError):
        gwet_agreement(d, weights="linear")


def test_weighted_ac2_rejects_a_zero_span() -> None:
    # Distinct category codes whose numeric values coincide have a zero span, which
    # gives no scale for weighting. Only reachable via direct construction.
    d = ReliabilityData(codes=np.array([[0, 1]]), categories=(1, 2), _values=np.array([2.0, 2.0]))
    with pytest.raises(AgreementInputError, match="non-zero numeric span"):
        gwet_agreement(d, weights="linear")


def test_units_with_one_rating_still_shape_chance_not_observed() -> None:
    # A fully-agreeing pair plus a single-rater unit of the rare category. The rare
    # unit has no partner, so it cannot add observed-agreement pairs, but it still
    # balances the category proportions and so raises the chance term p_e.
    d = ReliabilityData.from_rows([[1, 1], [2, None]])
    result = gwet_agreement(d)
    assert result.n_units == 1  # only the pairable unit enters observed agreement
    assert result.n_ratings == 2
    assert result.defined
    # The single-rater unit moved p_e from ~0 (single category) to 0.5 -> expected
    # disagreement 0.5. Observed agreement is still perfect (1.0 - 0.0).
    assert result.expected_disagreement == pytest.approx(0.5)
    assert result.observed_disagreement == pytest.approx(0.0)


def test_single_category_gives_one_not_nan() -> None:
    """Gwet's AC1 deliberately does NOT collapse on a unanimous single category the
    way kappa does (0/0 -> NaN). Both raters agreeing on one value is, per Gwet's
    chance model, perfect agreement with pe ~ 0, so AC1 reports 1.0. This is the
    same case the kappa-paradox test above builds toward: imbalance should not be
    mistaken for disagreement."""
    d = ReliabilityData.from_rows([[1, 1], [1, 1]])
    result = gwet_agreement(d)
    assert result.defined
    assert result.value == pytest.approx(1.0, abs=1e-9)


def test_no_pairable_unit_is_nan_with_a_reason() -> None:
    d = ReliabilityData.from_rows([[1, None], [None, 2]])
    result = gwet_agreement(d)
    assert not result.defined
    assert "two or more ratings" in (result.reason or "")


def test_all_metric_names_flow_through_weights() -> None:
    d = ReliabilityData.from_rows([[1, 1], [2, 3]])
    assert gwet_agreement(d).metric == "ac1"
    assert gwet_agreement(d, weights="linear").metric == "ac2-linear"
    assert gwet_agreement(d, weights="quadratic").metric == "ac2-quadratic"


@pytest.mark.parametrize("weights", ["none", "linear", "quadratic"])
def test_agreement_is_an_increasing_function_of_agreement(weights: KappaWeights) -> None:
    agreed = ReliabilityData.from_rows([[1, 1], [2, 2], [3, 3]])
    mixed = ReliabilityData.from_rows([[1, 3], [2, 1], [3, 2]])
    assert (
        gwet_agreement(agreed, weights=weights).value > gwet_agreement(mixed, weights=weights).value
    )


def test_bounded_between_negative_one_and_one() -> None:
    rng = np.random.default_rng(7)
    rows = rng.integers(1, 5, size=(40, 3))
    d = ReliabilityData.from_rows(rows.tolist())
    assert -1.0 <= gwet_agreement(d).value <= 1.0
