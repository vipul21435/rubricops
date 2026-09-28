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
AlphaLevel = Literal["nominal", "interval"]
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


def fleiss_kappa(
    ratings: RatingsLike, *, categories: Sequence[object] | None = None
) -> AgreementResult:
    """Fleiss' kappa for units that each carry the same number ``m >= 2`` of ratings.

    Which raters gave the ratings does not matter (Fleiss' design lets every unit
    have a different set of raters), so missing cells are fine as long as each unit
    ends up with ``m`` ratings. Units with no ratings at all are ignored. Any other
    unevenness raises :class:`AgreementInputError`; Krippendorff's alpha is the
    statistic for varying numbers of ratings.

    With two raters per unit this is Scott's pi.
    """
    data = as_reliability_data(ratings, categories=categories)
    counts = data.unit_counts()
    per_unit = counts.sum(axis=1)
    rated = per_unit > 0
    sizes = sorted({int(m) for m in per_unit[rated]})
    if len(sizes) > 1:
        msg = (
            f"Fleiss' kappa needs the same number of ratings on every unit; found {sizes}. "
            "Use Krippendorff's alpha for missing or uneven ratings"
        )
        raise AgreementInputError(msg)
    if sizes == [1]:
        msg = "Fleiss' kappa needs at least two ratings per unit; every unit has one"
        raise AgreementInputError(msg)
    counts = counts[rated]
    used = data.codes[rated]
    used = used[used != MISSING]
    n = int(counts.shape[0])
    observed = expected = 0.0
    if n:
        m = sizes[0]
        w = _nominal_weights(data.n_categories)
        # Per unit: the share of ordered rating pairs that disagree.
        per_unit_disagreement = np.einsum("uj,jk,uk->u", counts, w, counts) / (m * (m - 1))
        observed = float(per_unit_disagreement.mean())
        p = counts.sum(axis=0) / (n * m)
        expected = float(p @ w @ p)
    return _result(
        "fleiss",
        data,
        used,
        observed=observed,
        expected=expected,
        n_units=n,
        empty_reason="no unit has any ratings",
    )


def coincidence_matrix(data: ReliabilityData) -> FloatArray:
    """Krippendorff's coincidence matrix ``o[c, k]`` over the pairable units.

    Each unit with ``m_u >= 2`` ratings contributes every ordered pair of its ratings
    from different raters, weighted ``1 / (m_u - 1)``, so every pairable rating adds
    exactly 1 to its row. Units with fewer than two ratings contribute nothing.
    """
    counts = data.unit_counts()
    per_unit = counts.sum(axis=1)
    pairable = per_unit >= 2
    c = counts[pairable].astype(np.float64)
    scaled = c / (per_unit[pairable] - 1)[:, None].astype(np.float64)
    coincidences: FloatArray = scaled.T @ c - np.diag(scaled.sum(axis=0))
    return coincidences


def krippendorff_alpha(
    ratings: RatingsLike,
    *,
    level: AlphaLevel = "nominal",
    categories: Sequence[object] | None = None,
) -> AgreementResult:
    """Krippendorff's alpha at the nominal or interval level of measurement.

    Works with any number of raters, missing ratings and a different number of
    ratings per unit. Units with fewer than two ratings are not pairable and are
    dropped (``n_units`` counts the rest). With ``o`` the coincidence matrix,
    ``n_c`` its row sums and ``n`` the number of pairable ratings::

        D_o = sum(o[c, k] * delta2[c, k]) / n
        D_e = sum(n_c * n_k * delta2[c, k]) / (n * (n - 1))

    where ``delta2`` is 1 for different categories (nominal) or ``(c - k)^2``
    (interval, which needs numeric ratings). Interval disagreements are therefore in
    squared rating units; alpha itself is unit-free.
    """
    data = as_reliability_data(ratings, categories=categories)
    if level == "interval":
        values = data.scale_values()
        delta2 = (values[:, None] - values[None, :]) ** 2
    else:
        delta2 = _nominal_weights(data.n_categories)
    per_unit = (data.codes != MISSING).sum(axis=1)
    used = data.codes[per_unit >= 2]
    used = used[used != MISSING]
    n = int(used.size)
    observed = expected = 0.0
    if n:
        o = coincidence_matrix(data)
        n_c = o.sum(axis=1)
        observed = float((o * delta2).sum() / n)
        expected = float(n_c @ delta2 @ n_c / (n * (n - 1)))
    return _result(
        f"alpha-{level}",
        data,
        used,
        observed=observed,
        expected=expected,
        n_units=int((per_unit >= 2).sum()),
        empty_reason="no unit has two or more ratings, so no rating can be paired",
    )
