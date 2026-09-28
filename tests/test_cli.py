from __future__ import annotations

from typer.testing import CliRunner

from rubricops import __version__
from rubricops.cli import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"rubricops {__version__}"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_config_masks_secret() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "database_url=sqlite:///" in result.stdout
    assert "jwt_secret=**********" in result.stdout
