from __future__ import annotations

from pathlib import Path

import pytest

from rubricops.db.engine import missing_sqlite_file


@pytest.mark.parametrize(
    "url", ["sqlite://", "sqlite:///:memory:", "postgresql+psycopg://u:p@db/rubricops"]
)
def test_only_sqlite_files_can_be_missing(url: str) -> None:
    assert missing_sqlite_file(url) is None


def test_a_missing_file_is_reported_without_creating_it(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b.db"
    assert missing_sqlite_file(f"sqlite:///{path}") == path
    assert not path.parent.exists()
    path.parent.mkdir()
    path.touch()
    assert missing_sqlite_file(f"sqlite:///{path}") is None
