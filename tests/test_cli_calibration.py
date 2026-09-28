from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rubricops.cli import app

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
BASE = [
    "calibration",
    "report",
    "--gold",
    str(EXAMPLES / "calibration" / "gold.yaml"),
    "--reviews",
    str(EXAMPLES / "calibration" / "reviews.yaml"),
    "--rubric",
    str(EXAMPLES / "rubrics" / "code-explanation.yaml"),
    "--resamples",
    "500",
]


def test_table_report_flags_the_injected_reviewers() -> None:
    result = runner.invoke(app, BASE)
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "rubric code-explanation: 5 reviewers, 95% bootstrap CIs (500 resamples, seed 7)" in out
    assert "flags: correctness harsh" in out
    assert "flags: clarity lenient" in out
    assert "drift: emeka: mean absolute error rose" in out
    assert out.rstrip().endswith("flagged for QA: bruno, chen, emeka")


def test_json_report_for_one_reviewer_is_deterministic() -> None:
    args = [*BASE, "--format", "json", "--reviewer", "chen"]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert first.stdout == runner.invoke(app, args).stdout
    data = json.loads(first.stdout)
    assert data["flagged"] == ["bruno", "chen", "emeka"]
    (card,) = data["scorecards"]
    assert card["reviewer"] == "chen"
    clarity = next(c for c in card["criteria"] if c["criterion"] == "clarity")
    assert clarity["bias"] == "lenient"
    assert clarity["ci"][0] > 0
    assert {p["peer"] for p in card["peers"]} == {"asha", "bruno", "dara", "emeka"}


def test_unknown_reviewer_and_bad_files_exit_1(tmp_path: Path) -> None:
    result = runner.invoke(app, [*BASE, "--reviewer", "nobody"])
    assert result.exit_code == 1
    assert "no gold reviews by 'nobody'" in result.stderr
    bad = tmp_path / "gold.yaml"
    bad.write_text("items: [{id: g, scores: {correctness: 9}}]\n")
    result = runner.invoke(app, [*BASE, "--gold", str(bad)])
    assert result.exit_code == 1
    assert "correctness: 9 is outside scale" in result.stderr
    bad.write_text("items: [{id: g}]\n")
    result = runner.invoke(app, [*BASE, "--gold", str(bad)])
    assert result.exit_code == 1
    assert "scores" in result.stderr


def test_queue_sample_takes_calibration_flags(tmp_path: Path) -> None:
    report = runner.invoke(app, [*BASE, "--format", "json"])
    flags = tmp_path / "calibration.json"
    flags.write_text(report.stdout)
    scenario = str(EXAMPLES / "queue" / "scenario.yaml")
    sample = ["queue", "sample", scenario, "--seed", "7", "--rate", "0"]
    before = runner.invoke(app, sample)
    after = runner.invoke(app, [*sample, "--flags", str(flags)])
    assert after.exit_code == 0, after.output
    assert "4 of 8 sent to QA" in before.stdout
    assert "5 of 8 sent to QA" in after.stdout
    assert "86  QA        chen     calibration_flag" in after.stdout


@pytest.mark.parametrize("content", ["not json", '{"flagged": [1]}', "[]"])
def test_queue_sample_rejects_bad_flag_files(tmp_path: Path, content: str) -> None:
    flags = tmp_path / "flags.json"
    flags.write_text(content)
    scenario = str(EXAMPLES / "queue" / "scenario.yaml")
    result = runner.invoke(app, ["queue", "sample", scenario, "--flags", str(flags)])
    assert result.exit_code == 1
    assert str(flags) in result.stderr


def test_mixed_naive_and_aware_timestamps_do_not_crash(tmp_path: Path) -> None:
    scores = "{correctness: 4, completeness: 3, clarity: 4, safe_advice: 0}"
    gold = tmp_path / "gold.yaml"
    gold.write_text(
        f"rubric: code-explanation\nitems:\n  - {{id: g1, scores: {scores}}}\n"
        f"  - {{id: g2, scores: {scores}}}\n"
    )
    reviews = tmp_path / "reviews.yaml"
    reviews.write_text(
        "reviews:\n"
        f"  - {{reviewer: a, item: g1, at: '2026-09-01T09:00:00Z', scores: {scores}}}\n"
        f"  - {{reviewer: a, item: g2, at: '2026-09-01T10:00:00', scores: {scores}}}\n"
        f"  - {{reviewer: a, item: g1, at: 2026-09-01 11:00:00, scores: {scores}}}\n"
    )
    result = runner.invoke(
        app, [*BASE, "--gold", str(gold), "--reviews", str(reviews), "--window", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "flagged for QA: none" in result.stdout


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_non_finite_drift_threshold_is_a_usage_error(value: str) -> None:
    result = runner.invoke(app, [*BASE, "--drift-threshold", value])
    assert result.exit_code == 2
    assert "finite" in result.output
