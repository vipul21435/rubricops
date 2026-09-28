"""Calibration against gold items: injected leniency, harshness and drift are found."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rubricops.domain.calibration import (
    Bias,
    CalibrationRules,
    GoldItem,
    GoldReview,
    calibrate,
    classify,
    detect_drift,
    signed_error_ci,
)
from rubricops.domain.rubric import Rubric
from rubricops.domain.sampling import QaSampler, Reason, SampleCandidate, SamplingRules
from rubricops.domain.scoring import InvalidScoresError
from rubricops.loaders import load_rubric

RUBRIC = load_rubric(Path(__file__).resolve().parents[2] / "examples/rubrics/code-explanation.yaml")
START = datetime(2026, 9, 1, tzinfo=UTC)
RULES = CalibrationRules(n_resamples=500, window=10)


def _clip(rubric: Rubric, cid: str, value: int) -> int:
    scale = next(c.scale for c in rubric.criteria if c.id == cid)
    return max(scale.low, min(scale.high, value))


def _gold(n: int, seed: int = 1) -> list[GoldItem]:
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data
    return [
        GoldItem(
            f"g{i:02d}",
            {c.id: rng.randint(c.scale.low + 1, c.scale.high - 1) for c in RUBRIC.criteria},
        )
        for i in range(n)
    ]


def _reviews(
    reviewer: str,
    gold: list[GoldItem],
    *,
    offset: dict[str, int] | None = None,
    noise: float = 0.0,
    seed: int = 2,
    later_offset: int = 0,
) -> list[GoldReview]:
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data
    out = []
    for i, item in enumerate(gold):
        shift = later_offset if i >= len(gold) // 2 else 0
        scores = {}
        for cid, value in item.scores.items():
            jitter = rng.choice((-1, 1)) if rng.random() < noise else 0
            raw = value + (offset or {}).get(cid, 0) + jitter + shift
            scores[cid] = _clip(RUBRIC, cid, raw)
        out.append(GoldReview(reviewer, item.item_id, scores, START + timedelta(hours=i)))
    return out


def test_a_perfect_reviewer_is_exact_and_unflagged() -> None:
    gold = _gold(20)
    report = calibrate(RUBRIC, gold, _reviews("ann", gold), RULES)
    card = report.scorecard("ann")
    assert card.exact_match_rate == 1.0
    assert card.verdict_agreement == 1.0
    assert all(c.mae == 0.0 and c.bias is Bias.NEUTRAL for c in card.criteria)
    assert card.flags == ()
    assert not report.is_flagged("ann")


def test_injected_leniency_and_harshness_are_detected_per_criterion() -> None:
    gold = _gold(30)
    reviews = [
        *_reviews("fair", gold, noise=0.3, seed=3),
        *_reviews("soft", gold, offset={"clarity": 1}, noise=0.3, seed=4),
        *_reviews("hard", gold, offset={"correctness": -1}, noise=0.3, seed=5),
    ]
    report = calibrate(RUBRIC, gold, reviews, RULES)
    bias = {
        card.reviewer: {c.criterion_id: c.bias for c in card.criteria} for card in report.scorecards
    }
    assert set(bias["fair"].values()) == {Bias.NEUTRAL}
    assert bias["soft"]["clarity"] is Bias.LENIENT
    assert bias["hard"]["correctness"] is Bias.HARSH
    assert {k for k, v in bias["soft"].items() if v is not Bias.NEUTRAL} == {"clarity"}
    assert {k for k, v in bias["hard"].items() if v is not Bias.NEUTRAL} == {"correctness"}
    assert report.flagged == {"soft", "hard"}
    assert "clarity lenient" in report.scorecard("soft").flags


def test_a_step_change_triggers_drift_and_a_steady_reviewer_does_not() -> None:
    gold = _gold(20)
    drifting = _reviews("dan", gold, later_offset=-1, seed=6)
    alert = detect_drift("dan", drifting, {g.item_id: g for g in gold}, RULES)
    assert alert is not None
    assert alert.previous_mae == 0.0
    assert alert.recent_mae == pytest.approx(1.0)
    assert "rose +1.000" in alert.describe()
    steady = _reviews("sue", gold, noise=0.2, seed=7)
    assert detect_drift("sue", steady, {g.item_id: g for g in gold}, RULES) is None
    assert detect_drift("dan", drifting[:19], {g.item_id: g for g in gold}, RULES) is None
    report = calibrate(RUBRIC, gold, drifting, RULES)
    assert "drift" in report.scorecard("dan").flags


def test_peer_agreement_is_perfect_between_identical_reviewers() -> None:
    gold = _gold(12)
    reviews = [*_reviews("a", gold, noise=0.4, seed=8), *_reviews("b", gold, noise=0.4, seed=8)]
    card = calibrate(RUBRIC, gold, reviews, RULES).scorecard("a")
    (peer,) = card.peers
    assert peer.peer == "b"
    assert peer.shared_ratings == 12 * 4
    assert peer.kappa.value == pytest.approx(1.0)
    assert peer.alpha.value == pytest.approx(1.0)


def test_flags_feed_the_qa_sampler() -> None:
    gold = _gold(30)
    report = calibrate(RUBRIC, gold, _reviews("soft", gold, offset={"clarity": 1}), RULES)
    sampler = QaSampler(SamplingRules(rate=0.0, new_reviewer_min_reviews=0), seed=1)
    candidate = SampleCandidate(
        submission_id=1,
        reviewer_id=9,
        score=0.95,
        pass_threshold=0.7,
        reviewer_completed_reviews=50,
        reviewer_flagged=report.is_flagged("soft"),
    )
    assert sampler.decide(candidate).reasons == (Reason.CALIBRATION_FLAG,)


def test_low_verdict_agreement_is_a_flag_and_unknown_items_are_listed() -> None:
    gold = _gold(10)
    harsh = _reviews("low", gold, offset={c.id: 2 for c in RUBRIC.criteria})
    stray = GoldReview("low", "missing", dict(gold[0].scores), START)
    card = calibrate(RUBRIC, gold, [*harsh, stray], RULES)
    assert card.unknown_items == ("missing",)
    flags = card.scorecard("low").flags
    assert any(flag.startswith("verdict agreement") for flag in flags)
    with pytest.raises(KeyError, match="no gold reviews by 'nobody'"):
        card.scorecard("nobody")


def test_invalid_scores_are_rejected() -> None:
    gold = [GoldItem("g", {"correctness": 9})]
    with pytest.raises(InvalidScoresError):
        calibrate(RUBRIC, gold, [], RULES)


def test_signed_error_ci_and_classification() -> None:
    assert classify(signed_error_ci([1, 1, 1, 2], RULES)) is Bias.LENIENT
    assert classify(signed_error_ci([-1, -2, -1, -1], RULES)) is Bias.HARSH
    assert classify(signed_error_ci([-1, 1, 0, 0], RULES)) is Bias.NEUTRAL
    assert classify(signed_error_ci([], RULES)) is Bias.NEUTRAL


@pytest.mark.parametrize(
    "kwargs",
    [{"window": 0}, {"drift_threshold": -1.0}, {"min_verdict_agreement": 1.5}],
)
def test_rules_are_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        CalibrationRules(**kwargs)  # type: ignore[arg-type]


def test_reviewers_with_no_shared_items_have_no_peer_agreement() -> None:
    gold = _gold(8)
    reviews = [*_reviews("a", gold[:4]), *_reviews("b", gold[4:])]
    report = calibrate(RUBRIC, gold, reviews, RULES)
    assert report.scorecard("a").peers == ()
    assert report.scorecard("b").peers == ()
