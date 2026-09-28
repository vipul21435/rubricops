"""QA sampling: which primary reviews get a second, independent audit.

A review is sampled when a seeded random draw falls under the base rate
(``RUBRICOPS_QA_SAMPLE_RATE``) *or* any risk rule fires:

- ``new_reviewer``: the reviewer has fewer than ``new_reviewer_min_reviews``
  completed reviews;
- ``calibration_flag``: calibration flagged the reviewer (the hook slice 5 fills);
- ``near_threshold``: the normalised score is within ``threshold_margin`` of the
  rubric's pass threshold, compared as exact decimals so ``0.75`` against a ``0.7``
  threshold with a ``0.05`` margin is inside;
- ``low_agreement``: the scores recorded on this item (for example from earlier
  rounds or overlapping reviewers) span more than ``max_score_spread``.

Every decision records all the reasons that applied, the random draw and the rate,
so an audit can replay it. The draw for an item comes from
``random.Random(f"{seed}:{submission_id}:{round}")``: it depends only on the seed
and the item, not on the order or size of the batch, so re-running a sweep gives
the same decisions and adding an item never changes the others. The draw is taken
even when a risk rule fires, so the random part stays an unbiased sample.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from rubricops.domain.rubric import exact_decimal


class Reason(StrEnum):
    RANDOM = "random"
    NEW_REVIEWER = "new_reviewer"
    CALIBRATION_FLAG = "calibration_flag"
    NEAR_THRESHOLD = "near_threshold"
    LOW_AGREEMENT = "low_agreement"


def _unit(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0, 1], got {value}"
        raise ValueError(msg)
    return value


@dataclass(frozen=True, slots=True)
class SamplingRules:
    rate: float
    new_reviewer_min_reviews: int = 20
    threshold_margin: float = 0.05
    max_score_spread: float = 0.25

    def __post_init__(self) -> None:
        _unit("rate", self.rate)
        _unit("threshold_margin", self.threshold_margin)
        _unit("max_score_spread", self.max_score_spread)
        if self.new_reviewer_min_reviews < 0:
            msg = f"new_reviewer_min_reviews must be >= 0, got {self.new_reviewer_min_reviews}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SampleCandidate:
    """A primary review waiting for the finalize-or-QA decision."""

    submission_id: int
    reviewer_id: int
    score: float
    pass_threshold: float
    reviewer_completed_reviews: int
    round: int = 1
    item_scores: Sequence[float] = ()
    reviewer_flagged: bool = False

    def __post_init__(self) -> None:
        _unit("score", self.score)
        _unit("pass_threshold", self.pass_threshold)
        for other in self.item_scores:
            _unit("item score", other)
        object.__setattr__(self, "item_scores", tuple(self.item_scores))


@dataclass(frozen=True, slots=True)
class SampleDecision:
    submission_id: int
    sampled: bool
    reasons: tuple[Reason, ...]
    details: tuple[str, ...]
    draw: float
    rate: float


@dataclass(frozen=True, slots=True)
class QaSampler:
    """Seeded, order-independent QA sampling with recorded reasons."""

    rules: SamplingRules
    seed: int

    def draw(self, submission_id: int, round_: int = 1) -> float:
        return random.Random(f"{self.seed}:{submission_id}:{round_}").random()  # noqa: S311

    def decide(self, candidate: SampleCandidate) -> SampleDecision:
        rules = self.rules
        draw = self.draw(candidate.submission_id, candidate.round)
        reasons: list[Reason] = []
        details: list[str] = []
        if draw < rules.rate:
            reasons.append(Reason.RANDOM)
            details.append(f"draw {draw:.4f} < rate {rules.rate:g}")
        if candidate.reviewer_completed_reviews < rules.new_reviewer_min_reviews:
            reasons.append(Reason.NEW_REVIEWER)
            details.append(
                f"{candidate.reviewer_completed_reviews} completed reviews "
                f"< {rules.new_reviewer_min_reviews}"
            )
        if candidate.reviewer_flagged:
            reasons.append(Reason.CALIBRATION_FLAG)
            details.append("reviewer flagged by calibration")
        gap = abs(exact_decimal(candidate.score) - exact_decimal(candidate.pass_threshold))
        if gap <= exact_decimal(rules.threshold_margin):
            reasons.append(Reason.NEAR_THRESHOLD)
            details.append(
                f"score {candidate.score:.4f} within {rules.threshold_margin:g} "
                f"of threshold {candidate.pass_threshold:g}"
            )
        scores = (candidate.score, *candidate.item_scores)
        if len(scores) > 1:
            spread = exact_decimal(max(scores)) - exact_decimal(min(scores))
            if spread > exact_decimal(rules.max_score_spread):
                reasons.append(Reason.LOW_AGREEMENT)
                details.append(f"item scores span {float(spread):.4f} > {rules.max_score_spread:g}")
        return SampleDecision(
            submission_id=candidate.submission_id,
            sampled=bool(reasons),
            reasons=tuple(reasons),
            details=tuple(details),
            draw=draw,
            rate=rules.rate,
        )

    def decide_all(self, candidates: Iterable[SampleCandidate]) -> list[SampleDecision]:
        return [self.decide(c) for c in candidates]
