"""Reviewer calibration against gold items: accuracy, bias, drift and flags.

A gold item carries the per-criterion scores an expert panel agreed on. A reviewer
who grades a gold item (blind, as if it were a normal item) produces a
:class:`GoldReview`; comparing the two gives, per reviewer:

- **accuracy**: the exact-match rate (every criterion equal), the mean absolute
  error per criterion on the criterion's own scale, and pass/fail agreement (both
  verdicts from :func:`rubricops.domain.scoring.score_review`);
- **bias** per criterion: the mean signed error (reviewer minus gold) with a
  seeded percentile bootstrap interval from :func:`rubricops.stats.bootstrap_ci`.
  It is ``lenient`` only when the whole interval is above 0, ``harsh`` only when it
  is below 0, and ``neutral`` otherwise, so a few noisy reviews never label anyone;
- **drift**: the mean absolute error of the most recent window of gold reviews
  against the window before it; a rise of more than ``drift_threshold`` score
  points emits a :class:`DriftAlert`;
- **peer agreement**: Cohen's kappa (quadratic weights) and Krippendorff's alpha
  (interval) with every peer, over the (item, criterion) scores both graded.

A reviewer is *flagged* when any criterion is lenient or harsh, when drift fired,
or when pass/fail agreement is below ``min_verdict_agreement``. The flags feed the
QA sampler (``QueueService(is_flagged=...)``) through :meth:`CalibrationReport.is_flagged`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np

from rubricops.domain.rubric import Rubric
from rubricops.domain.scoring import score_review, validate_scores
from rubricops.stats.agreement import AgreementResult, cohen_kappa, krippendorff_alpha
from rubricops.stats.bootstrap import CIResult, bootstrap_ci
from rubricops.stats.data import ReliabilityData


class Bias(StrEnum):
    LENIENT = "lenient"
    HARSH = "harsh"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class GoldItem:
    item_id: str
    scores: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class GoldReview:
    reviewer: str
    item_id: str
    scores: Mapping[str, int]
    reviewed_at: datetime


@dataclass(frozen=True, slots=True)
class CalibrationRules:
    confidence: float = 0.95
    n_resamples: int = 2000
    seed: int = 7
    window: int = 10
    drift_threshold: float = 0.5
    min_verdict_agreement: float = 0.8

    def __post_init__(self) -> None:
        if self.window < 1:
            msg = f"window must be at least 1, got {self.window}"
            raise ValueError(msg)
        if self.drift_threshold < 0:
            msg = f"drift_threshold must be >= 0, got {self.drift_threshold}"
            raise ValueError(msg)
        if not 0.0 <= self.min_verdict_agreement <= 1.0:
            msg = f"min_verdict_agreement must be in [0, 1], got {self.min_verdict_agreement}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CriterionBias:
    criterion_id: str
    mae: float
    signed: CIResult
    bias: Bias


@dataclass(frozen=True, slots=True)
class DriftAlert:
    reviewer: str
    previous_mae: float
    recent_mae: float
    window: int
    threshold: float

    @property
    def change(self) -> float:
        return self.recent_mae - self.previous_mae

    def describe(self) -> str:
        return (
            f"{self.reviewer}: mean absolute error rose {self.change:+.3f} "
            f"({self.previous_mae:.3f} -> {self.recent_mae:.3f}) over the last "
            f"{self.window} gold reviews, threshold {self.threshold:g}"
        )


@dataclass(frozen=True, slots=True)
class PeerAgreement:
    peer: str
    shared_ratings: int
    kappa: AgreementResult
    alpha: AgreementResult


@dataclass(frozen=True, slots=True)
class Scorecard:
    reviewer: str
    gold_reviews: int
    exact_match_rate: float
    verdict_agreement: float
    criteria: tuple[CriterionBias, ...]
    drift: DriftAlert | None
    peers: tuple[PeerAgreement, ...]
    flags: tuple[str, ...]

    @property
    def flagged(self) -> bool:
        return bool(self.flags)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    rubric_id: str
    rules: CalibrationRules
    scorecards: tuple[Scorecard, ...]
    unknown_items: tuple[str, ...] = field(default=())

    def scorecard(self, reviewer: str) -> Scorecard:
        for card in self.scorecards:
            if card.reviewer == reviewer:
                return card
        msg = f"no gold reviews by {reviewer!r}"
        raise KeyError(msg)

    @property
    def flagged(self) -> frozenset[str]:
        return frozenset(card.reviewer for card in self.scorecards if card.flagged)

    def is_flagged(self, reviewer: str) -> bool:
        """The QA sampler hook: True sends every item of this reviewer to QA."""
        return reviewer in self.flagged


def _mean_statistic(data: ReliabilityData) -> AgreementResult:
    """The mean of a one-column table of numbers, shaped for :func:`bootstrap_ci`."""
    n = data.n_units
    mean = float(data.scale_values()[data.codes[:, 0]].mean()) if n else math.nan
    return AgreementResult(
        metric="mean_signed_error",
        value=mean,
        observed_disagreement=math.nan,
        expected_disagreement=math.nan,
        n_units=n,
        n_raters=1,
        n_ratings=n,
        reason=None if n else "no gold reviews",
    )


def signed_error_ci(errors: Sequence[int], rules: CalibrationRules) -> CIResult:
    """Mean of ``errors`` with its seeded percentile bootstrap interval."""
    data = ReliabilityData.from_rows([[e] for e in errors])
    return bootstrap_ci(
        data,
        _mean_statistic,
        seed=rules.seed,
        confidence=rules.confidence,
        n_resamples=rules.n_resamples,
    )


def classify(ci: CIResult) -> Bias:
    if not ci.defined:
        return Bias.NEUTRAL
    if ci.low > 0:
        return Bias.LENIENT
    if ci.high < 0:
        return Bias.HARSH
    return Bias.NEUTRAL


def _mae(reviews: Sequence[GoldReview], gold: Mapping[str, GoldItem]) -> float:
    errors = [
        abs(review.scores[cid] - gold[review.item_id].scores[cid])
        for review in reviews
        for cid in review.scores
    ]
    return float(np.mean(errors))


def detect_drift(
    reviewer: str,
    reviews: Sequence[GoldReview],
    gold: Mapping[str, GoldItem],
    rules: CalibrationRules,
) -> DriftAlert | None:
    """Compare the last ``window`` gold reviews with the ``window`` before them."""
    ordered = sorted(reviews, key=lambda r: (r.reviewed_at, r.item_id))
    if len(ordered) < 2 * rules.window:
        return None
    recent = ordered[-rules.window :]
    previous = ordered[-2 * rules.window : -rules.window]
    before, after = _mae(previous, gold), _mae(recent, gold)
    if after - before <= rules.drift_threshold:
        return None
    return DriftAlert(reviewer, before, after, rules.window, rules.drift_threshold)


def _peer_agreement(
    reviewer: str, by_reviewer: Mapping[str, Mapping[tuple[str, str], int]]
) -> tuple[PeerAgreement, ...]:
    mine = by_reviewer[reviewer]
    peers = []
    for peer in sorted(by_reviewer):
        if peer == reviewer:
            continue
        shared = sorted(set(mine) & set(by_reviewer[peer]))
        if not shared:
            continue
        rows = [[mine[key], by_reviewer[peer][key]] for key in shared]
        peers.append(
            PeerAgreement(
                peer=peer,
                shared_ratings=len(rows),
                kappa=cohen_kappa(rows, weights="quadratic"),
                alpha=krippendorff_alpha(rows, level="interval"),
            )
        )
    return tuple(peers)


def calibrate(
    rubric: Rubric,
    gold_items: Sequence[GoldItem],
    reviews: Sequence[GoldReview],
    rules: CalibrationRules | None = None,
) -> CalibrationReport:
    """Score every reviewer's gold reviews and return their scorecards."""
    rules = rules or CalibrationRules()
    gold = {item.item_id: item for item in gold_items}
    for item in gold_items:
        validate_scores(rubric, item.scores)
    unknown = sorted({r.item_id for r in reviews if r.item_id not in gold})
    usable = [r for r in reviews if r.item_id in gold]
    grouped: dict[str, list[GoldReview]] = defaultdict(list)
    ratings: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for review in usable:
        validate_scores(rubric, review.scores)
        grouped[review.reviewer].append(review)
        for cid, value in review.scores.items():
            ratings[review.reviewer][(review.item_id, cid)] = value

    cards = []
    for reviewer in sorted(grouped):
        mine = grouped[reviewer]
        exact = sum(dict(r.scores) == dict(gold[r.item_id].scores) for r in mine)
        verdicts = sum(
            score_review(rubric, r.scores).passed
            == score_review(rubric, gold[r.item_id].scores).passed
            for r in mine
        )
        criteria = []
        for cid in rubric.criterion_ids:
            errors = [r.scores[cid] - gold[r.item_id].scores[cid] for r in mine]
            ci = signed_error_ci(errors, rules)
            criteria.append(CriterionBias(cid, float(np.mean(np.abs(errors))), ci, classify(ci)))
        drift = detect_drift(reviewer, mine, gold, rules)
        verdict_rate = verdicts / len(mine)
        flags = [f"{c.criterion_id} {c.bias.value}" for c in criteria if c.bias is not Bias.NEUTRAL]
        if drift is not None:
            flags.append("drift")
        if verdict_rate < rules.min_verdict_agreement:
            flags.append(f"verdict agreement {verdict_rate:.2f} < {rules.min_verdict_agreement:g}")
        cards.append(
            Scorecard(
                reviewer=reviewer,
                gold_reviews=len(mine),
                exact_match_rate=exact / len(mine),
                verdict_agreement=verdict_rate,
                criteria=tuple(criteria),
                drift=drift,
                peers=_peer_agreement(reviewer, ratings),
                flags=tuple(flags),
            )
        )
    return CalibrationReport(rubric.id, rules, tuple(cards), tuple(unknown))
