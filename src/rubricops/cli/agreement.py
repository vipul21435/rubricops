"""``rubricops agreement``: inter-rater agreement with bootstrap intervals from a CSV.

Exit codes: 0 success, 1 unreadable ratings or a metric that cannot apply to them
(for example Cohen's kappa on three raters), 2 usage error. A metric that is
well-formed but undefined on the data (one category only) is reported as ``n/a``
with its reason and does not change the exit code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from rubricops.loaders import DocumentError, RatingsTable, load_ratings
from rubricops.settings import get_settings
from rubricops.stats.agreement import METRICS, AgreementResult
from rubricops.stats.bootstrap import CIResult, bootstrap_ci
from rubricops.stats.data import AgreementInputError, ReliabilityData


class Metric(StrEnum):
    """CLI choices; a test keeps this in step with ``agreement.METRICS``."""

    cohen = "cohen"
    cohen_linear = "cohen-linear"
    cohen_quadratic = "cohen-quadratic"
    fleiss = "fleiss"
    alpha_nominal = "alpha-nominal"
    alpha_interval = "alpha-interval"
    ac1 = "ac1"
    ac2_linear = "ac2-linear"
    ac2_quadratic = "ac2-quadratic"


class Layout(StrEnum):
    wide = "wide"
    long = "long"


@dataclass(frozen=True, slots=True)
class _Row:
    metric: str
    result: AgreementResult | None
    ci: CIResult | None
    error: str | None


def _number(value: float) -> float | None:
    return None if math.isnan(value) else value


def _row_json(row: _Row) -> dict[str, object]:
    if row.result is None:
        return {"metric": row.metric, "error": row.error}
    r = row.result
    payload: dict[str, object] = {
        "metric": row.metric,
        "value": _number(r.value),
        "observed_disagreement": _number(r.observed_disagreement),
        "expected_disagreement": _number(r.expected_disagreement),
        "n_units": r.n_units,
        "n_raters": r.n_raters,
        "n_ratings": r.n_ratings,
        "reason": r.reason,
    }
    if row.ci is not None:
        payload["ci"] = {
            "confidence": row.ci.confidence,
            "low": _number(row.ci.low),
            "high": _number(row.ci.high),
            "n_resamples": row.ci.n_resamples,
            "n_degenerate": row.ci.n_degenerate,
            "reason": row.ci.reason,
        }
    return payload


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _row_text(row: _Row, width: int) -> str:
    if row.result is None:
        return f"{row.metric:<{width}}  error: {row.error}"
    r = row.result
    interval = ""
    if row.ci is not None and row.ci.defined:
        interval = f"[{row.ci.low:.3f}, {row.ci.high:.3f}]"
        if row.ci.n_degenerate:
            interval += f" ({row.ci.n_degenerate} degenerate resamples skipped)"
    elif row.ci is not None and r.defined:
        interval = f"n/a ({row.ci.reason})"
    parts = [f"{row.metric:<{width}}  {_fmt(r.value):>6}  {r.n_units:>5}  {r.n_ratings:>7}"]
    if interval:
        parts.append(interval)
    if r.reason:
        parts.append(f"({r.reason})")
    return "  ".join(parts)


def _compute(data: ReliabilityData, name: str, *, ci: float, resamples: int, seed: int) -> _Row:
    statistic = METRICS[name]
    try:
        result = statistic(data)
    except AgreementInputError as exc:
        return _Row(metric=name, result=None, ci=None, error=str(exc))
    interval = None
    if ci > 0:
        interval = bootstrap_ci(data, statistic, seed=seed, confidence=ci, n_resamples=resamples)
    return _Row(metric=name, result=result, ci=interval, error=None)


def agreement(
    path: Annotated[Path, typer.Argument(metavar="FILE.csv", help="Ratings CSV.")],
    *,
    metrics: Annotated[
        list[Metric] | None,
        typer.Option(
            "--metric",
            "-m",
            help="Coefficient to compute; repeat for several. Default: alpha-nominal.",
        ),
    ] = None,
    layout: Annotated[
        Layout,
        typer.Option(
            help="wide: unit column then one column per rater. long: columns unit, rater, rating."
        ),
    ] = Layout.wide,
    ci: Annotated[
        float,
        typer.Option(min=0.0, max=0.999, help="Bootstrap confidence level; 0 skips the bootstrap."),
    ] = 0.95,
    resamples: Annotated[int, typer.Option(min=1, help="Bootstrap resamples.")] = 2000,
    seed: Annotated[
        int | None,
        typer.Option(help="Bootstrap seed. Default: RUBRICOPS_RANDOM_SEED."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Compute agreement coefficients for a ratings table, with bootstrap intervals."""
    try:
        table: RatingsTable = load_ratings(path, layout=layout.value)
    except DocumentError as exc:
        typer.echo(f"error: {exc.path}", err=True)
        for issue in exc.issues:
            typer.echo(f"  {issue}", err=True)
        raise typer.Exit(code=1) from exc
    data = ReliabilityData.from_rows(table.rows)
    names = [m.value for m in metrics] if metrics else ["alpha-nominal"]
    names = list(dict.fromkeys(names))
    used_seed = get_settings().random_seed if seed is None else seed
    rows = [_compute(data, n, ci=ci, resamples=resamples, seed=used_seed) for n in names]
    if as_json:
        payload = {
            "file": str(path),
            "units": len(table.units),
            "raters": list(table.raters),
            "categories": list(data.categories),
            "seed": used_seed if ci > 0 else None,
            "metrics": [_row_json(row) for row in rows],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        categories = ", ".join(str(c) for c in data.categories)
        typer.echo(
            f"{path}  units={len(table.units)} raters={len(table.raters)} categories=[{categories}]"
        )
        width = max(len("metric"), *(len(n) for n in names))
        header = f"{'metric':<{width}}  {'value':>6}  {'units':>5}  {'ratings':>7}"
        if ci > 0:
            header += f"  {ci * 100:g}% CI ({resamples} resamples, seed {used_seed})"
        typer.echo(header)
        for row in rows:
            typer.echo(_row_text(row, width))
    if any(row.error for row in rows):
        raise typer.Exit(code=1)
