"""``rubricops agreement`` on the bundled examples and on edge-case CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rubricops.cli import app
from rubricops.cli.agreement import Metric
from rubricops.loaders import load_ratings
from rubricops.stats.agreement import METRICS, krippendorff_alpha
from rubricops.stats.data import ReliabilityData

runner = CliRunner()
RATINGS = Path(__file__).resolve().parents[1] / "examples" / "ratings"
CORRECTNESS = RATINGS / "correctness-3-reviewers.csv"
VERDICTS = RATINGS / "verdicts-long.csv"


def _csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ratings.csv"
    path.write_text(text, encoding="utf-8")
    return path


def _json(args: list[str]) -> dict[str, object]:
    result = runner.invoke(app, ["agreement", *args, "--json"])
    assert result.exit_code == 0, result.output
    payload: dict[str, object] = json.loads(result.stdout)
    return payload


def test_metric_choices_match_the_registry() -> None:
    assert [m.value for m in Metric] == list(METRICS)


def test_example_ratings_files_load() -> None:
    wide = load_ratings(CORRECTNESS)
    assert (len(wide.units), wide.raters) == (20, ("rev-ana", "rev-ben", "rev-chen"))
    long = load_ratings(VERDICTS, layout="long")
    assert (len(long.units), long.raters) == (16, ("rev-ana", "rev-ben"))


def test_default_metric_on_the_wide_example() -> None:
    result = runner.invoke(app, ["agreement", str(CORRECTNESS)])
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0] == f"{CORRECTNESS}  units=20 raters=3 categories=[1, 2, 3, 4]"
    assert lines[1].endswith("95% CI (2000 resamples, seed 20260929)")
    expected = krippendorff_alpha(ReliabilityData.from_rows(load_ratings(CORRECTNESS).rows))
    assert lines[2].startswith(f"alpha-nominal  {expected.value:>6.3f}     20       58  [")
    assert len(lines) == 3


def test_cohen_on_the_long_example_matches_a_hand_count() -> None:
    """16 responses, 4 split verdicts: p_o = 12/16. Each reviewer passes 9 of 16, so
    p_e = (9/16)^2 + (7/16)^2 = 130/256 and kappa = (0.75 - p_e) / (1 - p_e) = 62/126."""
    payload = _json([str(VERDICTS), "--layout", "long", "-m", "cohen", "--ci", "0"])
    assert payload["seed"] is None
    assert payload["categories"] == ["fail", "pass"]
    [row] = payload["metrics"]  # type: ignore[misc]
    assert row["value"] == pytest.approx(62 / 126)
    assert row["observed_disagreement"] == pytest.approx(0.25)
    assert "ci" not in row


def test_the_same_seed_gives_the_same_json() -> None:
    args = [str(CORRECTNESS), "-m", "alpha-interval", "--resamples", "200", "--seed", "7"]
    first, second = _json(args), _json(args)
    assert first == second
    [row] = first["metrics"]  # type: ignore[misc]
    ci = row["ci"]
    assert ci["n_resamples"] == 200
    assert ci["low"] <= row["value"] <= ci["high"]
    assert first["seed"] == 7


def test_seed_defaults_to_the_settings_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUBRICOPS_RANDOM_SEED", "42")
    result = runner.invoke(app, ["agreement", str(CORRECTNESS), "--resamples", "50"])
    assert "95% CI (50 resamples, seed 42)" in result.stdout


def test_repeated_metrics_are_reported_once() -> None:
    args = [str(CORRECTNESS), "-m", "alpha-interval", "-m", "alpha-nominal", "-m", "alpha-interval"]
    payload = _json([*args, "--ci", "0"])
    metrics = payload["metrics"]
    assert [m["metric"] for m in metrics] == ["alpha-interval", "alpha-nominal"]  # type: ignore[attr-defined]


def test_a_metric_that_cannot_apply_is_an_error_but_others_still_report() -> None:
    result = runner.invoke(
        app, ["agreement", str(CORRECTNESS), "-m", "cohen", "-m", "alpha-interval", "--ci", "0"]
    )
    assert result.exit_code == 1
    lines = result.stdout.splitlines()
    assert lines[1] == f"{'metric':<14}   value  units  ratings"
    assert lines[2] == f"{'cohen':<14}  error: Cohen's kappa compares exactly two raters; got 3"
    assert lines[3].startswith("alpha-interval   0.")


def test_json_reports_errors_per_metric() -> None:
    result = runner.invoke(app, ["agreement", str(CORRECTNESS), "-m", "fleiss", "--json"])
    assert result.exit_code == 1
    [row] = json.loads(result.stdout)["metrics"]
    assert row["metric"] == "fleiss"
    assert row["error"].startswith("Fleiss' kappa needs the same number of ratings")


def test_an_undefined_metric_is_reported_with_its_reason(tmp_path: Path) -> None:
    path = _csv(tmp_path, "item,a,b\nq1,pass,pass\nq2,pass,pass\n")
    result = runner.invoke(app, ["agreement", str(path), "-m", "cohen"])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[2] == (
        "cohen      n/a      2        4  (every usable rating is 'pass': with a single "
        "category the expected disagreement is 0, so the coefficient is 0/0)"
    )
    payload = _json([str(path), "-m", "cohen"])
    [row] = payload["metrics"]  # type: ignore[misc]
    assert row["value"] is None
    assert row["ci"]["low"] is None
    assert row["ci"]["reason"].startswith("point estimate undefined")


def test_degenerate_resamples_are_reported(tmp_path: Path) -> None:
    path = _csv(tmp_path, "item,a,b\nq1,pass,pass\nq2,fail,fail\n")
    result = runner.invoke(app, ["agreement", str(path), "-m", "cohen", "--resamples", "100"])
    assert result.exit_code == 0, result.output
    row = result.stdout.splitlines()[2]
    assert row.startswith("cohen    1.000      2        4  [1.000, 1.000] (")
    assert row.endswith(" degenerate resamples skipped)")


def test_an_interval_with_no_valid_resample_says_why(tmp_path: Path) -> None:
    """Seed 0 draws unit 2 twice in the single resample, which sees one category."""
    path = _csv(tmp_path, "item,a,b\nq1,pass,pass\nq2,fail,fail\n")
    args = ["agreement", str(path), "-m", "cohen", "--resamples", "1", "--seed", "0"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[2] == (
        "cohen    1.000      2        4  n/a (all 1 resamples were degenerate)"
    )


def test_an_unreadable_file_exits_1_with_the_issues(tmp_path: Path) -> None:
    path = _csv(tmp_path, "item,a\nq1,1\nq1,2\n")
    result = runner.invoke(app, ["agreement", str(path)])
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [f"error: {path}", "  line 3: unit 'q1' appears twice"]
    assert result.stdout == ""


@pytest.mark.parametrize("args", [["-m", "kappa"], ["--ci", "1.0"], ["--resamples", "0"]])
def test_bad_options_are_usage_errors(args: list[str]) -> None:
    result = runner.invoke(app, ["agreement", str(CORRECTNESS), *args])
    assert result.exit_code == 2
