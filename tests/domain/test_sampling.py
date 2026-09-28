from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rubricops.domain.sampling import (
    QaSampler,
    Reason,
    SampleCandidate,
    SamplingRules,
)

SEED = 20260929


def _candidate(sid: int = 1, **kwargs: object) -> SampleCandidate:
    defaults: dict[str, object] = {
        "reviewer_id": 7,
        "score": 0.9,
        "pass_threshold": 0.7,
        "reviewer_completed_reviews": 100,
    }
    defaults.update(kwargs)
    return SampleCandidate(sid, **defaults)  # type: ignore[arg-type]


def _no_random(**kwargs: object) -> QaSampler:
    return QaSampler(SamplingRules(rate=0.0, **kwargs), SEED)  # type: ignore[arg-type]


def test_a_safe_review_with_rate_zero_is_not_sampled() -> None:
    decision = _no_random().decide(_candidate())
    assert not decision.sampled
    assert decision.reasons == ()
    assert decision.details == ()
    assert 0.0 <= decision.draw < 1.0
    assert decision.rate == 0.0


def test_rate_one_samples_everything_at_random() -> None:
    sampler = QaSampler(SamplingRules(rate=1.0), SEED)
    decisions = sampler.decide_all(_candidate(i) for i in range(50))
    assert all(d.reasons == (Reason.RANDOM,) for d in decisions)
    assert decisions[0].details[0].startswith("draw ")


def test_new_reviewer_rule() -> None:
    sampler = _no_random(new_reviewer_min_reviews=5)
    assert sampler.decide(_candidate(reviewer_completed_reviews=4)).reasons == (
        Reason.NEW_REVIEWER,
    )
    assert not sampler.decide(_candidate(reviewer_completed_reviews=5)).sampled
    assert (
        "4 completed reviews < 5"
        in sampler.decide(_candidate(reviewer_completed_reviews=4)).details[0]
    )


def test_calibration_flag_hook() -> None:
    decision = _no_random().decide(_candidate(reviewer_flagged=True))
    assert decision.reasons == (Reason.CALIBRATION_FLAG,)


@pytest.mark.parametrize(
    ("score", "near"),
    [(0.75, True), (0.65, True), (0.7, True), (0.7501, False), (0.6499, False), (1.0, False)],
)
def test_near_threshold_margin_is_exact(score: float, near: bool) -> None:
    decision = _no_random(threshold_margin=0.05).decide(_candidate(score=score))
    assert (Reason.NEAR_THRESHOLD in decision.reasons) is near


def test_low_agreement_uses_the_spread_of_item_scores() -> None:
    sampler = _no_random(max_score_spread=0.25)
    assert not sampler.decide(_candidate(score=0.9, item_scores=[0.65])).sampled  # exactly 0.25
    decision = sampler.decide(_candidate(score=0.9, item_scores=[0.95, 0.6]))
    assert decision.reasons == (Reason.LOW_AGREEMENT,)
    assert decision.details == ("item scores span 0.3500 > 0.25",)
    assert not sampler.decide(_candidate(score=0.9)).sampled  # one score: no spread


def test_every_reason_that_applies_is_recorded_in_order() -> None:
    sampler = QaSampler(SamplingRules(rate=1.0, new_reviewer_min_reviews=10), SEED)
    decision = sampler.decide(
        _candidate(
            score=0.72,
            reviewer_completed_reviews=2,
            reviewer_flagged=True,
            item_scores=[0.2],
        )
    )
    assert decision.reasons == (
        Reason.RANDOM,
        Reason.NEW_REVIEWER,
        Reason.CALIBRATION_FLAG,
        Reason.NEAR_THRESHOLD,
        Reason.LOW_AGREEMENT,
    )
    assert len(decision.details) == 5


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="rate must be in"):
        SamplingRules(rate=1.5)
    with pytest.raises(ValueError, match="new_reviewer_min_reviews"):
        SamplingRules(rate=0.1, new_reviewer_min_reviews=-1)
    with pytest.raises(ValueError, match="score must be in"):
        _candidate(score=-0.1)
    with pytest.raises(ValueError, match="item score must be in"):
        _candidate(item_scores=[2.0])


def test_same_seed_same_decisions_regardless_of_order() -> None:
    rules = SamplingRules(rate=0.3)
    candidates = [_candidate(i) for i in range(200)]
    forward = QaSampler(rules, SEED).decide_all(candidates)
    backward = QaSampler(rules, SEED).decide_all(reversed(candidates))
    assert forward == list(reversed(backward))
    other = QaSampler(rules, SEED + 1).decide_all(candidates)
    assert [d.sampled for d in forward] != [d.sampled for d in other]


def test_a_new_round_gets_a_fresh_draw() -> None:
    sampler = _no_random()
    assert sampler.draw(1, 1) != sampler.draw(1, 2)
    assert sampler.draw(1, 1) == sampler.draw(1, 1)


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.5])
def test_random_rate_is_within_binomial_bounds(rate: float) -> None:
    n = 10_000
    sampler = QaSampler(SamplingRules(rate=rate), SEED)
    hits = sum(d.sampled for d in sampler.decide_all(_candidate(i) for i in range(n)))
    sigma = math.sqrt(n * rate * (1 - rate))
    assert abs(hits - n * rate) <= 4 * sigma


@given(
    score=st.floats(0, 1),
    threshold=st.floats(0, 1),
    completed=st.integers(0, 50),
    flagged=st.booleans(),
)
def test_risk_rules_only_add_reasons(
    score: float, threshold: float, completed: int, flagged: bool
) -> None:
    candidate = _candidate(
        score=score,
        pass_threshold=threshold,
        reviewer_completed_reviews=completed,
        reviewer_flagged=flagged,
    )
    decision = QaSampler(SamplingRules(rate=0.2), SEED).decide(candidate)
    assert decision.sampled == bool(decision.reasons)
    assert len(decision.reasons) == len(decision.details)
    assert (Reason.RANDOM in decision.reasons) == (decision.draw < 0.2)
