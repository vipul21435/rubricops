from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rubricops.cli import app

runner = CliRunner()


def test_upgrade_current_downgrade(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    result = runner.invoke(app, ["db", "current", "--url", url])
    assert result.stdout.strip() == f"{url} is at revision none"
    result = runner.invoke(app, ["db", "upgrade", "--url", url])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == f"{url} is at revision 0001"
    result = runner.invoke(app, ["db", "downgrade", "base", "--url", url])
    assert result.stdout.strip() == f"{url} is at revision base (empty)"


def test_default_url_from_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'env.db'}"
    monkeypatch.setenv("RUBRICOPS_DATABASE_URL", url)
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.stdout.strip() == f"{url} is at revision 0001"


def test_upgrade_sql_prints_ddl(tmp_path: Path) -> None:
    result = runner.invoke(app, ["db", "upgrade", "--sql", "--url", "sqlite:///x.db"])
    assert result.exit_code == 0
    assert "CREATE TABLE submissions" in result.stdout
    assert not (tmp_path / "x.db").exists()


def test_password_is_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("rubricops.db.migrate.current_revision", calls.append)
    result = runner.invoke(app, ["db", "current", "--url", "postgresql://u:hunter2@db/r"])
    assert "hunter2" not in result.stdout
    assert "postgresql://u:***@db/r is at revision none" in result.stdout
    assert calls == ["postgresql://u:hunter2@db/r"]
