from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from typer.testing import CliRunner

from rubricops.cli import app
from rubricops.db.engine import make_engine

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
RUBRIC = str(EXAMPLES / "rubrics" / "code-explanation.yaml")
SCORES = str(EXAMPLES / "reviews" / "walkthrough-scores.yaml")


def _walk(url: str, scores: str = SCORES) -> tuple[int, str, str]:
    result = runner.invoke(
        app, ["pipeline", "walkthrough", "--url", url, "--rubric", RUBRIC, "--scores", scores]
    )
    return result.exit_code, result.stdout, result.stderr


@pytest.fixture
def walked(tmp_path: Path) -> str:
    url = f"sqlite:///{tmp_path / 'walk.db'}"
    code, _out, err = _walk(url)
    assert code == 0, err
    return url


def test_walkthrough_takes_three_routes_and_verifies(tmp_path: Path) -> None:
    code, out, _ = _walk(f"sqlite:///{tmp_path / 'a.db'}")
    assert code == 0
    assert "submit_qa            qa_pending -> qa_passed" in out
    assert "submit_qa            qa_pending -> qa_failed" in out
    assert "adjudicate      in_adjudication -> finalized        by lead-2" in out
    assert "refused: submission 1 is at version 5, not version 4; reload and retry" in out
    assert "refused: submit_qa forbidden by auditor_is_not_primary" in out
    assert "         3  finalized       0.6417  FAIL     adjudication review" in out
    assert "audit chain ok: 26 events, head seq 26" in out


def test_walkthrough_is_deterministic(tmp_path: Path) -> None:
    first = _walk(f"sqlite:///{tmp_path / 'a.db'}")[1]
    second = _walk(f"sqlite:///{tmp_path / 'b.db'}")[1]
    assert first == second


def test_walkthrough_refuses_a_used_database(walked: str) -> None:
    code, _, err = _walk(walked)
    assert code == 1
    assert "already has users" in err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("good: {correctness: 3}\n", "missing score maps ['better', 'close', 'gated', 'settled']"),
        ("good: [1, 2]\n", "good: Input should be a valid dictionary"),
        ("good: {a: 1}\ngood: {a: 2}\n", "duplicate key"),
    ],
)
def test_walkthrough_rejects_bad_score_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "scores.yaml"
    path.write_text(content)
    code, _, err = _walk(f"sqlite:///{tmp_path / 'x.db'}", str(path))
    assert code == 1
    assert message in err


def test_walkthrough_rejects_scores_that_do_not_fit_the_rubric(tmp_path: Path) -> None:
    path = tmp_path / "scores.yaml"
    full = "{correctness: 3, completeness: 3, clarity: 3, safe_advice: 1}"
    path.write_text("\n".join(f"{k}: {full}" for k in ("better", "close", "gated", "settled")))
    path.write_text(path.read_text() + "\ngood: {correctness: 9}\n")
    code, _, err = _walk(f"sqlite:///{tmp_path / 'x.db'}", str(path))
    assert code == 1
    assert "missing scores for ['completeness', 'clarity', 'safe_advice']" in err


def test_audit_verify_ok_and_anchor(walked: str) -> None:
    result = runner.invoke(app, ["audit", "verify", "--url", walked])
    assert result.exit_code == 0
    assert result.stdout.startswith("ok: 26 events, head seq 26 hash ")
    anchor = result.stdout.splitlines()[1].removeprefix("anchor: ")
    again = runner.invoke(app, ["audit", "verify", "--url", walked, "--anchor", anchor])
    assert again.exit_code == 0
    wrong = anchor[:-1] + ("0" if anchor[-1] != "0" else "1")
    bad = runner.invoke(app, ["audit", "verify", "--url", walked, "--anchor", wrong])
    assert bad.exit_code == 1
    assert "BROKEN at seq 26: hash differs from the anchored hash" in bad.stdout


def test_audit_verify_reports_tampering(walked: str) -> None:
    engine = make_engine(walked)
    with engine.begin() as conn:
        conn.execute(text("DROP TRIGGER audit_events_no_update"))
        conn.execute(text("UPDATE audit_events SET actor_id = 2 WHERE seq = 17"))
    engine.dispose()
    result = runner.invoke(app, ["audit", "verify", "--url", walked])
    assert result.exit_code == 1
    assert result.stdout.strip() == (
        "BROKEN at seq 17: hash does not match the event's content (16 events checked)"
    )


@pytest.mark.parametrize("anchor", ["x:" + "a" * 64, "3:abc", "3"])
def test_audit_verify_rejects_malformed_anchor(walked: str, anchor: str) -> None:
    result = runner.invoke(app, ["audit", "verify", "--url", walked, "--anchor", anchor])
    assert result.exit_code == 2


def _scores_with(tmp_path: Path, **overrides: str) -> str:
    """The bundled walkthrough scores with some maps replaced."""
    lines = []
    for line in Path(SCORES).read_text().splitlines():
        key = line.split(":", 1)[0]
        lines.append(f"{key}: {overrides[key]}" if key in overrides else line)
    path = tmp_path / "scores.yaml"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"close": "{correctness: 4, completeness: 1, clarity: 1, safe_advice: 2}"},
            "'close' must have the same verdict as 'better' and be within the QA tolerance",
        ),
        (
            {"gated": "{correctness: 4, completeness: 3, clarity: 3, safe_advice: 2}"},
            "'gated' must disagree with 'better'",
        ),
    ],
)
def test_walkthrough_rejects_scores_that_change_the_routes_without_writing(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    db = tmp_path / "w2.db"
    code, out, err = _walk(f"sqlite:///{db}", _scores_with(tmp_path, **overrides))
    assert code == 1
    assert message in err
    assert "Traceback" not in out + err
    assert not db.exists()
    # The same URL still works afterwards with scores that take the routes.
    assert _walk(f"sqlite:///{db}")[0] == 0


def test_audit_verify_does_not_create_a_missing_database(tmp_path: Path) -> None:
    db = tmp_path / "typo" / "walkthru.db"
    result = runner.invoke(app, ["audit", "verify", "--url", f"sqlite:///{db}"])
    assert result.exit_code == 1
    assert f"error: no database at {db}; nothing to verify" in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert not db.exists()
    assert not db.parent.exists()


def test_audit_verify_reports_an_unmigrated_database(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    engine = make_engine(f"sqlite:///{db}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE unrelated (id INTEGER)"))
    engine.dispose()
    result = runner.invoke(app, ["audit", "verify", "--url", f"sqlite:///{db}"])
    assert result.exit_code == 1
    assert "has no audit log table; run rubricops db upgrade first" in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_audit_verify_names_a_link_whose_data_is_not_an_object(walked: str) -> None:
    engine = make_engine(walked)
    with engine.begin() as connection:
        connection.execute(text("DROP TRIGGER audit_events_no_update"))
        connection.execute(text("UPDATE audit_events SET data = 'null' WHERE seq = 20"))
    engine.dispose()
    result = runner.invoke(app, ["audit", "verify", "--url", walked])
    assert result.exit_code == 1
    assert result.stdout.startswith("BROKEN at seq 20: data is not a JSON object")
