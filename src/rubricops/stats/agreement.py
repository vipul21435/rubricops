"""Chance-corrected inter-rater agreement coefficients, computed with numpy.

Every coefficient here has the form ``1 - D_o / D_e``: observed disagreement over
the disagreement expected by chance. :class:`AgreementResult` reports both terms so a
reader can see *why* a value is low (reviewers disagree) or unstable (little
disagreement was possible in the first place).

Input is a units x raters table (see :class:`~rubricops.stats.data.ReliabilityData`);
``None`` or NaN marks a missing rating.

Degenerate data returns NaN with a ``reason`` instead of raising:

- no usable units (for example no unit rated by both raters), and
- only one category among the usable ratings. Expected disagreement is then zero,
  so the coefficient is 0/0. This covers "perfect agreement on a single category":
  two reviewers who both pass every item agree perfectly, but a kappa cannot tell
  that apart from chance.

Perfect agreement over two or more categories is well defined and gives exactly 1.
Malformed input (wrong number of raters, text ratings for a weighted metric) raises
:class:`~rubricops.stats.data.AgreementInputError`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from rubricops.stats.data import (
    MISSING,
    AgreementInputError,
    FloatArray,
    IntArray,
    ReliabilityData,
)

KappaWeights = Literal["none", "linear", "quadratic"]
RatingsLike = ReliabilityData | Sequence[Sequence[object]] | npt.NDArray[np.generic]


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """One agreement coefficient and the terms it was computed from.

    ``value`` is ``1 - observed_disagreement / expected_disagreement``, or NaN with a
    ``reason`` when that ratio is undefined. ``n_units`` and ``n_ratings`` count only
    the units and ratings that entered the computation (after dropping incomplete
    pairs or units with fewer than two ratings).
    """

    metric: str
    value: float
    observed_disagreement: float
    expected_disagreement: float
    n_units: int
    n_raters: int
    n_ratings: int
    reason: str | None = None

    @property
    def defined(self) -> bool:
        return self.reason is None


def as_reliability_data(
    ratings: RatingsLike, *, categories: Sequence[object] | None = None
) -> ReliabilityData:
    """Accept prepared :class:`ReliabilityData` or encode a units x raters table."""
    if isinstance(ratings, ReliabilityData):
        if categories is not None:
            msg = "categories are fixed when ReliabilityData is built; pass them there"
            raise AgreementInputError(msg)
        return ratings
    return ReliabilityData.from_rows(ratings, categories=categories)


def _result(
    metric: str,
    data: ReliabilityData,
    used_codes: IntArray,
    *,
    observed: float,
    expected: float,
    n_units: int,
    empty_reason: str,
) -> AgreementResult:
    """Build the result, turning a zero expected disagreement into NaN with a reason."""
    reason = None
    value = math.nan
    if n_units == 0:
        reason = empty_reason
    elif expected <= 0.0:
        only = data.categories[int(used_codes.flat[0])]
        reason = (
            f"every usable rating is {only!r}: with a single category the expected "
            "disagreement is 0, so the coefficient is 0/0"
        )
    else:
        value = 1.0 - observed / expected
    return AgreementResult(
        metric=metric,
        value=value,
        observed_disagreement=observed if n_units else math.nan,
        expected_disagreement=expected if n_units else math.nan,
        n_units=n_units,
        n_raters=data.n_raters,
        n_ratings=int(used_codes.size),
        reason=reason,
    )


def _nominal_weights(k: int) -> FloatArray:
    return np.ones((k, k), dtype=np.float64) - np.eye(k, dtype=np.float64)


def _kappa_weights(data: ReliabilityData, weights: KappaWeights) -> FloatArray:
    """Disagreement weights ``w[i, j]`` in [0, 1] with a zero diagonal."""
    if weights == "none":
        return _nominal_weights(data.n_categories)
    values = data.scale_values()
    distance = np.abs(values[:, None] - values[None, :])
    span = float(values.max() - values.min())
    if span > 0.0:
        distance /= span
    return distance if weights == "linear" else distance**2


def cohen_kappa(
    ratings: RatingsLike,
    *,
    weights: KappaWeights = "none",
    categories: Sequence[object] | None = None,
) -> AgreementResult:
    """Cohen's kappa for exactly two raters, optionally linear or quadratic weighted.

    ``ratings`` has one row per unit and two columns (``ReliabilityData.from_columns
    (a, b)`` builds it from two lists). Units missing either rating are dropped.

    Weighted kappa needs numeric ratings. Weights use the rating *values*, so a 1-4
    disagreement costs three times a 1-2 one even if 2 and 3 are never used:
    ``|x - y| / span`` (linear) or its square (quadratic), where ``span`` is the
    range of ``categories``. Quadratic-weighted kappa then equals Lin's concordance
    correlation coefficient of the two raters' scores.
    """
    data = as_reliability_data(ratings, categories=categories)
    if data.n_raters != 2:
        msg = f"Cohen's kappa compares exactly two raters; got {data.n_raters}"
        raise AgreementInputError(msg)
    metric = "cohen" if weights == "none" else f"cohen-{weights}"
    w = _kappa_weights(data, weights)
    pairs = data.codes[(data.codes != MISSING).all(axis=1)]
    n = int(pairs.shape[0])
    k = data.n_categories
    observed = expected = 0.0
    if n:
        joint = np.bincount(pairs[:, 0] * k + pairs[:, 1], minlength=k * k).reshape(k, k) / n
        observed = float((w * joint).sum())
        expected = float(joint.sum(axis=1) @ w @ joint.sum(axis=0))
    return _result(
        metric,
        data,
        pairs,
        observed=observed,
        expected=expected,
        n_units=n,
        empty_reason="no unit has a rating from both raters",
    )
