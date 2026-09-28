"""``rubricops calibration report``: reviewer scorecards from gold items.

Exit codes: 0 success, 1 invalid input files or an unknown ``--reviewer``, 2 usage
error.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ConfigDict, ValidationError

from rubricops.domain.calibration import (
    CalibrationReport,
    CalibrationRules,
    GoldItem,
    GoldReview,
    Scorecard,
    calibrate,
)
from rubricops.domain.scoring import InvalidScoresError
from rubricops.loaders import DocumentError, format_validation_error, load_document, load_rubric

calibration_app = typer.Typer(help="Calibrate reviewers against gold items.", no_args_is_help=True)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _GoldEntry(_Strict):
    id: str
    scores: dict[str, int]


class _GoldFile(_Strict):
    rubric: str | None = None
    items: list[_GoldEntry]


class _ReviewEntry(_Strict):
    reviewer: str
    item: str
    at: datetime
    scores: dict[str, int]


class _ReviewsFile(_Strict):
    reviews: list[_ReviewEntry]


class Format(StrEnum):
    TABLE = "table"
    JSON = "json"


def _read[T: BaseModel](path: Path, model: type[T]) -> T:
    try:
        return model.model_validate(load_document(path))
    except ValidationError as exc:
        raise DocumentError(path, format_validation_error(exc)) from exc


def _num(value: float) -> float | None:
    return None if math.isnan(value) else round(value, 4)


def _card_json(card: Scorecard) -> dict[str, Any]:
    return {
        "reviewer": card.reviewer,
        "gold_reviews": card.gold_reviews,
        "exact_match_rate": _num(card.exact_match_rate),
        "verdict_agreement": _num(card.verdict_agreement),
        "criteria": [
            {
                "criterion": c.criterion_id,
                "mae": _num(c.mae),
                "mean_signed_error": _num(c.signed.estimate),
                "ci": [_num(c.signed.low), _num(c.signed.high)],
                "bias": c.bias.value,
            }
            for c in card.criteria
        ],
        "drift": None
        if card.drift is None
        else {
            "previous_mae": _num(card.drift.previous_mae),
            "recent_mae": _num(card.drift.recent_mae),
            "window": card.drift.window,
        },
        "peers": [
            {
                "peer": p.peer,
                "shared_ratings": p.shared_ratings,
                "kappa_quadratic": _num(p.kappa.value),
                "alpha_interval": _num(p.alpha.value),
            }
            for p in card.peers
        ],
        "flags": list(card.flags),
    }


def report_json(report: CalibrationReport, cards: list[Scorecard]) -> dict[str, Any]:
    rules = report.rules
    return {
        "rubric": report.rubric_id,
        "confidence": rules.confidence,
        "n_resamples": rules.n_resamples,
        "seed": rules.seed,
        "window": rules.window,
        "drift_threshold": rules.drift_threshold,
        "flagged": sorted(report.flagged),
        "unknown_items": list(report.unknown_items),
        "scorecards": [_card_json(card) for card in cards],
    }


def _echo_table(report: CalibrationReport, cards: list[Scorecard]) -> None:
    rules = report.rules
    typer.echo(
        f"rubric {report.rubric_id}: {len(report.scorecards)} reviewers, "
        f"{rules.confidence:.0%} bootstrap CIs ({rules.n_resamples} resamples, seed {rules.seed})"
    )
    for card in cards:
        typer.echo(
            f"\n{card.reviewer}: {card.gold_reviews} gold reviews, exact match "
            f"{card.exact_match_rate:.2f}, pass/fail agreement {card.verdict_agreement:.2f}"
        )
        typer.echo("  criterion        MAE   mean error  CI                bias")
        for c in card.criteria:
            ci = f"[{c.signed.low:+.3f}, {c.signed.high:+.3f}]"
            typer.echo(
                f"  {c.criterion_id:<14} {c.mae:5.3f}  {c.signed.estimate:+10.3f}  "
                f"{ci:<17} {c.bias.value}"
            )
        if card.drift is not None:
            typer.echo(f"  drift: {card.drift.describe()}")
        for p in card.peers:
            typer.echo(
                f"  vs {p.peer:<8} kappa(quadratic) {p.kappa.value:+.3f}  "
                f"alpha(interval) {p.alpha.value:+.3f}  on {p.shared_ratings} ratings"
            )
        typer.echo(f"  flags: {', '.join(card.flags) if card.flags else 'none'}")
    typer.echo(f"\nflagged for QA: {', '.join(sorted(report.flagged)) or 'none'}")


@calibration_app.command("report")
def report(
    *,
    gold_path: Annotated[Path, typer.Option("--gold", help="Gold items file.")] = Path(
        "examples/calibration/gold.yaml"
    ),
    reviews_path: Annotated[Path, typer.Option("--reviews", help="Gold reviews file.")] = Path(
        "examples/calibration/reviews.yaml"
    ),
    rubric_path: Annotated[Path, typer.Option("--rubric", help="Rubric file.")] = Path(
        "examples/rubrics/code-explanation.yaml"
    ),
    reviewer: Annotated[
        str | None, typer.Option("--reviewer", help="Only this reviewer's scorecard.")
    ] = None,
    output: Annotated[Format, typer.Option("--format", help="table or json.")] = Format.TABLE,
    window: Annotated[int, typer.Option("--window", min=1, help="Drift window.")] = 10,
    drift_threshold: Annotated[
        float, typer.Option("--drift-threshold", min=0.0, help="MAE rise that is drift.")
    ] = 0.5,
    resamples: Annotated[int, typer.Option("--resamples", min=1)] = 2000,
    seed: Annotated[int, typer.Option("--seed")] = 7,
) -> None:
    """Accuracy, bias with bootstrap CIs, drift and peer agreement per reviewer."""
    if not math.isfinite(drift_threshold):
        raise typer.BadParameter("must be a finite number", param_hint="--drift-threshold")
    try:
        rubric = load_rubric(rubric_path)
        gold = _read(gold_path, _GoldFile)
        reviews = _read(reviews_path, _ReviewsFile)
        rules = CalibrationRules(
            n_resamples=resamples, seed=seed, window=window, drift_threshold=drift_threshold
        )
        result = calibrate(
            rubric,
            [GoldItem(item.id, item.scores) for item in gold.items],
            [GoldReview(r.reviewer, r.item, r.scores, r.at) for r in reviews.reviews],
            rules,
        )
    except DocumentError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except InvalidScoresError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    cards = list(result.scorecards)
    if reviewer is not None:
        try:
            cards = [result.scorecard(reviewer)]
        except KeyError as exc:
            typer.echo(f"error: no gold reviews by {reviewer!r}", err=True)
            raise typer.Exit(code=1) from exc
    if output is Format.JSON:
        typer.echo(json.dumps(report_json(result, cards), indent=2, sort_keys=True))
    else:
        _echo_table(result, cards)
