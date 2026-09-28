"""Percentile bootstrap confidence intervals for agreement statistics.

The resampled unit is the *unit* (the rated item), not the individual rating: a
unit's ratings are not independent of each other, but units are (to the extent the
sample of items is). Each resample draws ``n_units`` rows with replacement from the
full table, using a seeded :class:`numpy.random.Generator`, so the same seed always
gives the same interval.

A resample can be degenerate even when the full data is not, for example when it
happens to draw only units that were all rated "pass". Such resamples have no
defined statistic; they are skipped and counted in ``CIResult.n_degenerate`` rather
than silently biasing the interval. A large share of degenerate resamples is itself
a warning that the estimate rests on very few informative units.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rubricops.stats.agreement import AgreementResult
from rubricops.stats.data import ReliabilityData

Statistic = Callable[[ReliabilityData], AgreementResult]


@dataclass(frozen=True, slots=True)
class CIResult:
    """A point estimate with its percentile bootstrap interval.

    ``low`` and ``high`` are NaN, with a ``reason``, when the point estimate is
    undefined (no resampling is done then, so ``n_resamples`` is 0) or when every
    resample was degenerate.
    """

    estimate: float
    low: float
    high: float
    confidence: float
    n_resamples: int
    n_degenerate: int
    reason: str | None = None

    @property
    def n_valid(self) -> int:
        """Resamples that produced a defined statistic and entered the percentiles."""
        return self.n_resamples - self.n_degenerate

    @property
    def defined(self) -> bool:
        return self.reason is None

    @property
    def contains_estimate(self) -> bool:
        """Whether ``low <= estimate <= high``.

        A percentile interval is not guaranteed to contain the estimate: with very few
        units or a low confidence level the bootstrap distribution can sit to one side
        of it. False here means the interval should be read with care.
        """
        return self.defined and self.low <= self.estimate <= self.high


def bootstrap_ci(
    data: ReliabilityData,
    statistic: Statistic,
    *,
    seed: int | np.random.Generator,
    confidence: float = 0.95,
    n_resamples: int = 2000,
) -> CIResult:
    """Resample units with replacement and report the percentile interval.

    ``statistic`` is any function of :class:`ReliabilityData`, typically one of
    :data:`rubricops.stats.agreement.METRICS`. ``seed`` is required so every
    interval is reproducible; pass an ``int`` or a ``numpy.random.Generator``.
    """
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must be strictly between 0 and 1, got {confidence}"
        raise ValueError(msg)
    if n_resamples < 1:
        msg = f"n_resamples must be at least 1, got {n_resamples}"
        raise ValueError(msg)
    point = statistic(data)
    if not point.defined:
        return CIResult(
            estimate=math.nan,
            low=math.nan,
            high=math.nan,
            confidence=confidence,
            n_resamples=0,
            n_degenerate=0,
            reason=f"point estimate undefined: {point.reason}",
        )
    rng = np.random.default_rng(seed)
    n = data.n_units
    estimates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        estimates[i] = statistic(data.take(rng.integers(0, n, size=n))).value
    valid = estimates[~np.isnan(estimates)]
    n_degenerate = n_resamples - int(valid.size)
    if valid.size == 0:
        return CIResult(
            estimate=point.value,
            low=math.nan,
            high=math.nan,
            confidence=confidence,
            n_resamples=n_resamples,
            n_degenerate=n_degenerate,
            reason=f"all {n_resamples} resamples were degenerate",
        )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(valid, [tail, 1.0 - tail])
    return CIResult(
        estimate=point.value,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        n_degenerate=n_degenerate,
    )
