from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from rubricops.cli import app
from rubricops.domain.pipeline import Stage
from rubricops.loaders import DocumentError
from rubricops.scenario import load_scenario
from rubricops.settings import get_settings

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCENARIO = str(EXAMPLES / "queue" / "scenario.yaml")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BOX = re.compile("[\u2500-\u257f]")


def _plain(text: str) -> str:
    return " ".join(_BOX.sub(" ", _ANSI.sub("", text)).split())


def _run(*args: str) -> tuple[int, str, str]:
    result = runner.invoke(app, ["queue", *args])
    return result.exit_code, result.stdout, result.stderr


def _dump(tmp_path: Path, data: dict[str, Any], name: str = "s.yaml") -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(path)


# -- assign -------------------------------------------------------------------


def test_assign_round_robin_prints_the_cursor_to_persist() -> None:
    code, out, _ = _run("assign", SCENARIO, "--policy", "round-robin")
    assert code == 0
    assert "policy round-robin: 7 of 7 assigned" in out
    assert "     101  primary  python             asha (11)" in out
    assert out.rstrip().endswith("cursor: 13")
    # Resuming from the printed cursor continues after chen (13).
    _, resumed, _ = _run("assign", SCENARIO, "--policy", "round-robin", "--cursor", "13")
    assert "     101  primary  python             emeka (15)" in resumed


def test_assign_skill_match_reports_why_items_stay_unassigned() -> None:
    code, out, _ = _run("assign", SCENARIO, "--policy", "skill-match")
    assert code == 0
    assert "policy skill-match: 5 of 7 assigned" in out
    assert (
        "105  primary  rust               UNASSIGNED (asha=at_capacity, bruno=missing_skills, "
        "chen=missing_skills, dara=at_capacity, emeka=at_capacity)"
    ) in out
    assert "106  qa       python,sql         UNASSIGNED (asha=primary_reviewer," in out


def test_assign_load_balanced_is_the_default_and_levels_loads() -> None:
    code, out, _ = _run("assign", SCENARIO)
    assert code == 0
    assert "policy load-balanced: 7 of 7 assigned" in out
    assert "open loads after: asha=3, bruno=3, chen=3, dara=3, emeka=2" in out
    assert "cursor" not in out


def test_assign_with_no_reviewers(tmp_path: Path) -> None:
    path = _dump(tmp_path, {"queue": [{"submission": 1, "author": 1}]})
    code, out, _ = _run("assign", path)
    assert code == 0
    assert "UNASSIGNED (no reviewers)" in out


def test_assign_usage_errors() -> None:
    code, _, err = _run("assign", SCENARIO, "--policy", "random")
    assert code == 2
    assert "expected one of round-robin, skill-match, load-balanced" in _plain(err)
    code, _, err = _run("assign", SCENARIO, "--policy", "skill-match", "--cursor", "3")
    assert code == 2
    assert "only round-robin takes a cursor" in _plain(err)


# -- overdue ------------------------------------------------------------------


def test_overdue_lists_late_assignments_most_late_first() -> None:
    code, out, _ = _run("overdue", SCENARIO)
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == (
        "as of 2026-09-29 09:00 UTC: 3 of 5 open assignments overdue (default SLA 24h)"
    )
    assert [line.split()[0] for line in lines[2:]] == ["91", "94", "90"]
    # 90 uses the 8h code-explanation override: due 04:00, 5h late at 09:00.
    assert lines[-1].endswith("asha         2026-09-29 04:00  05h 00m")


def test_overdue_follows_now_and_sla_options() -> None:
    code, out, _ = _run("overdue", SCENARIO, "--now", "2026-09-28T08:00+0000")
    assert code == 0
    assert "0 of 5 open assignments overdue" in out
    code, out, _ = _run(
        "overdue", SCENARIO, "--now", "2026-09-30T12:00:00+0000", "--sla-hours", "1"
    )
    assert code == 0
    # --sla-hours replaces the file's default; per-rubric overrides still apply.
    assert "5 of 5 open assignments overdue (default SLA 1h)" in out


def test_overdue_without_a_scenario_time_uses_settings_and_system_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUBRICOPS_REVIEW_SLA_HOURS", "2")
    get_settings.cache_clear()
    data = {
        "reviewers": [{"id": 1, "handle": "r1"}],
        "assignments": [
            {"submission": 5, "reviewer": 1, "assigned_at": "2020-01-01T00:00:00+00:00"}
        ],
    }
    code, out, _ = _run("overdue", _dump(tmp_path, data))
    assert code == 0
    assert "1 of 1 open assignments overdue (default SLA 2h)" in out


# -- sample -------------------------------------------------------------------


def test_sample_records_reasons_for_every_decision() -> None:
    code, out, _ = _run("sample", SCENARIO)
    assert code == 0
    assert out.splitlines()[0] == "seed 20260929, rate 0.1: 4 of 8 sent to QA"
    assert "81  QA        bruno    random+new_reviewer" in out
    assert "82  QA        chen     near_threshold" in out
    assert "83  QA        dara     low_agreement                   item scores span 0.3500" in out
    assert "84  QA        emeka    calibration_flag+near_threshold" in out
    assert "80  finalize  asha     none                            draw 0.1740 >= rate 0.1" in out


def test_sample_is_deterministic_and_follows_options(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run("sample", SCENARIO)[1] == _run("sample", SCENARIO)[1]
    _, everything, _ = _run("sample", SCENARIO, "--rate", "1")
    assert "8 of 8 sent to QA" in everything
    _, strict, _ = _run(
        "sample",
        SCENARIO,
        "--rate",
        "0",
        "--min-reviews",
        "0",
        "--margin",
        "0",
        "--max-spread",
        "1",
    )
    assert "1 of 8 sent to QA" in strict  # only the calibration flag remains
    monkeypatch.setenv("RUBRICOPS_QA_SAMPLE_RATE", "0.5")
    monkeypatch.setenv("RUBRICOPS_RANDOM_SEED", "7")
    get_settings.cache_clear()
    _, from_env, _ = _run("sample", SCENARIO)
    assert from_env.startswith("seed 7, rate 0.5:")


# -- scenario validation ------------------------------------------------------


def test_sample_names_each_rounds_own_reviewer(tmp_path: Path) -> None:
    path = _dump(
        tmp_path,
        {
            "reviewers": [{"id": 11, "handle": "asha"}, {"id": 12, "handle": "bruno"}],
            "reviews": [
                {"submission": 80, "reviewer": 11, "round": 1, "score": 0.95, "threshold": 0.5},
                {"submission": 80, "reviewer": 12, "round": 2, "score": 0.95, "threshold": 0.5},
            ],
        },
    )
    code, out, _ = _run("sample", path, "--rate", "0", "--min-reviews", "0")
    assert code == 0
    rows = out.splitlines()[1:]
    assert [row.split()[2] for row in rows] == ["asha", "bruno"]


def test_the_example_scenario_loads() -> None:
    scenario = load_scenario(Path(SCENARIO))
    assert len(scenario.reviewer_pool()) == 5
    assert [i.stage for i in scenario.queue_items()].count(Stage.QA) == 2
    assert len(scenario.sample_candidates()) == 8


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([1, 2], "expected a scenario mapping, got list"),
        ({"reviewers": [{"id": 1, "handle": "a"}, {"id": 1, "handle": "b"}]}, "unique"),
        (
            {"assignments": [{"submission": 1, "reviewer": 9, "assigned_at": "2026-01-01T00:00Z"}]},
            "assignments name reviewers that are not listed: [9]",
        ),
        (
            {"reviews": [{"submission": 1, "reviewer": 9, "score": 0.5, "threshold": 0.5}]},
            "reviews name reviewers that are not listed: [9]",
        ),
        ({"queue": [{"submission": 1, "author": 1, "stage": "qa"}]}, "needs its primary_reviewer"),
        (
            {
                "reviewers": [{"id": 1, "handle": "a"}],
                "assignments": [
                    {
                        "submission": 1,
                        "reviewer": 1,
                        "assigned_at": "2026-01-02T00:00Z",
                        "due_at": "2026-01-01T00:00Z",
                    }
                ],
            },
            "due_at is before assigned_at",
        ),
        (
            {
                "reviewers": [{"id": 1, "handle": "a"}],
                "assignments": [
                    {"submission": 1, "reviewer": 1, "assigned_at": "2026-01-02T00:00:00"}
                ],
            },
            "timezone",
        ),
        ({"reviewers": [], "extra": 1}, "extra: Extra inputs are not permitted"),
        ({"sla": {"default_hours": 24, "rubrics": {"fast": 0}}}, "greater than 0"),
        ({"sla": {"rubrics": {"fast": float("nan")}}}, "finite number"),
        ({"sla": {"default_hours": float("inf")}}, "finite number"),
        ({"sla": {"rubrics": {"fast": 1e-300}}}, "SLA for fast must be positive"),
        ({"sla": {"rubrics": {"fast": 1e20}}}, "sla hours are too large"),
        (
            {
                "reviewers": [{"id": 1, "handle": "a"}],
                "reviews": [
                    {
                        "submission": 1,
                        "reviewer": 1,
                        "score": 0.9,
                        "threshold": 0.5,
                        "item_scores": [1.5],
                    }
                ],
            },
            "less than or equal to 1",
        ),
        (
            {
                "reviewers": [{"id": 1, "handle": "a"}, {"id": 2, "handle": "b"}],
                "reviews": [
                    {"submission": 80, "reviewer": 1, "score": 0.9, "threshold": 0.5},
                    {"submission": 80, "reviewer": 2, "score": 0.9, "threshold": 0.5},
                ],
            },
            "reviews repeat a (submission, round) pair: [(80, 1)]",
        ),
    ],
)
def test_invalid_scenarios_exit_1(tmp_path: Path, data: object, message: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DocumentError):
        load_scenario(path)
    for command in ("assign", "overdue", "sample"):
        code, _, err = _run(command, str(path))
        assert code == 1
        assert message in err
