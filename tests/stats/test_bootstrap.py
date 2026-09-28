"""Bootstrap confidence intervals and the metric registry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rubricops.stats.agreement import METRICS, AgreementResult, cohen_kappa
from rubricops.stats.bootstrap import bootstrap_ci
from rubricops.stats.data import ReliabilityData


def _two_raters(n_units: int, *, agree: float, seed: int) -> ReliabilityData:
    """Two raters on a 1..4 scale; rater B copies A with probability ``agree``."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 5, size=n_units).tolist()
    b = [x if rng.random() < agree else int(rng.integers(1, 5)) for x in a]
    return ReliabilityData.from_columns(a, b)


DATA = _two_raters(60, agree=0.7, seed=7)


def test_metric_registry_names_match_reported_metric() -> None:
    for name, statistic in METRICS.items():
        assert statistic(DATA).metric == name


def test_metric_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        METRICS["new"] = METRICS["cohen"]  # type: ignore[index]


def test_same_seed_gives_the_same_interval() -> None:
    first = bootstrap_ci(DATA, METRICS["cohen"], seed=11, n_resamples=300)
    second = bootstrap_ci(DATA, METRICS["cohen"], seed=11, n_resamples=300)
    assert first == second


def test_a_generator_is_accepted_as_the_seed() -> None:
    from_int = bootstrap_ci(DATA, METRICS["cohen"], seed=5, n_resamples=200)
    from_rng = bootstrap_ci(DATA, METRICS["cohen"], seed=np.random.default_rng(5), n_resamples=200)
    assert from_int == from_rng


def test_different_seeds_give_different_intervals() -> None:
    first = bootstrap_ci(DATA, METRICS["cohen"], seed=1, n_resamples=300)
    second = bootstrap_ci(DATA, METRICS["cohen"], seed=2, n_resamples=300)
    assert (first.low, first.high) != (second.low, second.high)
    assert first.estimate == second.estimate


@pytest.mark.parametrize("name", sorted(METRICS))
def test_interval_brackets_the_estimate_on_moderate_data(name: str) -> None:
    result = bootstrap_ci(DATA, METRICS[name], seed=3, n_resamples=400)
    assert result.defined
    assert result.contains_estimate
    assert -1.0 <= result.low <= result.estimate <= result.high <= 1.0
    assert result.estimate == METRICS[name](DATA).value
    assert result.n_resamples == 400
    assert result.n_degenerate == 0
    assert result.n_valid == 400
    assert result.confidence == 0.95


def test_a_higher_confidence_level_gives_a_wider_interval() -> None:
    narrow = bootstrap_ci(DATA, METRICS["cohen"], seed=9, confidence=0.5, n_resamples=500)
    wide = bootstrap_ci(DATA, METRICS["cohen"], seed=9, confidence=0.99, n_resamples=500)
    assert wide.low < narrow.low
    assert narrow.high < wide.high


def test_more_agreement_gives_a_higher_interval() -> None:
    low = bootstrap_ci(_two_raters(80, agree=0.2, seed=1), METRICS["cohen"], seed=4)
    high = bootstrap_ci(_two_raters(80, agree=0.9, seed=1), METRICS["cohen"], seed=4)
    assert low.high < high.low


def test_identical_units_give_a_zero_width_interval() -> None:
    data = ReliabilityData.from_rows([[1, 2]] * 10)
    result = bootstrap_ci(data, METRICS["cohen"], seed=0, n_resamples=50)
    assert result.low == result.high == result.estimate


def test_degenerate_resamples_are_skipped_and_counted() -> None:
    """Two units, both raters agree: kappa = 1. A resample that draws the same unit
    twice sees one category and has no kappa; about half of all resamples do."""
    data = ReliabilityData.from_rows([["pass", "pass"], ["fail", "fail"]])
    result = bootstrap_ci(data, METRICS["cohen"], seed=0, n_resamples=1000)
    assert result.defined
    assert result.estimate == 1.0
    assert result.low == result.high == 1.0
    assert 400 < result.n_degenerate < 600
    assert result.n_valid + result.n_degenerate == 1000


def test_undefined_point_estimate_skips_resampling() -> None:
    data = ReliabilityData.from_rows([["pass", "pass"], ["pass", "pass"]])
    result = bootstrap_ci(data, METRICS["cohen"], seed=0)
    assert not result.defined
    assert result.reason is not None
    assert result.reason.startswith("point estimate undefined: every usable rating is 'pass'")
    assert math.isnan(result.estimate)
    assert math.isnan(result.low)
    assert math.isnan(result.high)
    assert result.n_resamples == 0
    assert result.n_valid == 0
    assert not result.contains_estimate


def test_all_degenerate_resamples_give_no_interval() -> None:
    def only_on_full_data(data: ReliabilityData) -> AgreementResult:
        result = cohen_kappa(data)
        if data is DATA:
            return result
        return AgreementResult(
            metric="test",
            value=math.nan,
            observed_disagreement=math.nan,
            expected_disagreement=math.nan,
            n_units=0,
            n_raters=2,
            n_ratings=0,
            reason="resample",
        )

    result = bootstrap_ci(DATA, only_on_full_data, seed=0, n_resamples=20)
    assert result.estimate == cohen_kappa(DATA).value
    assert math.isnan(result.low)
    assert math.isnan(result.high)
    assert (result.n_resamples, result.n_degenerate, result.n_valid) == (20, 20, 0)
    assert result.reason == "all 20 resamples were degenerate"
    assert not result.defined


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_confidence_must_be_strictly_between_zero_and_one(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be strictly between 0 and 1"):
        bootstrap_ci(DATA, METRICS["cohen"], seed=0, confidence=confidence)


def test_at_least_one_resample_is_required() -> None:
    with pytest.raises(ValueError, match="n_resamples must be at least 1"):
        bootstrap_ci(DATA, METRICS["cohen"], seed=0, n_resamples=0)
