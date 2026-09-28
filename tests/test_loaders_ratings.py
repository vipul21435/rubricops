"""Ratings CSV loading: wide and long layouts, missing cells and every error path."""

from __future__ import annotations

from pathlib import Path

import pytest

from rubricops.loaders import DocumentError, RatingsTable, load_ratings, parse_rating


def _csv(tmp_path: Path, text: str, name: str = "ratings.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _issues(path: Path, **kwargs: object) -> list[str]:
    with pytest.raises(DocumentError) as info:
        load_ratings(path, **kwargs)  # type: ignore[arg-type]
    return list(info.value.issues)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("3", 3),
        (" 3 ", 3),
        ("-1", -1),
        ("2.5", 2.5),
        ("pass", "pass"),
        (" Needs work ", "Needs work"),
        ("", None),
        ("   ", None),
        ("NA", None),
        ("na", None),
        ("NaN", None),
    ],
)
def test_parse_rating(cell: str, expected: object) -> None:
    assert parse_rating(cell) == expected


@pytest.mark.parametrize("cell", ["inf", "-inf", "Infinity"])
def test_parse_rating_rejects_non_finite_numbers(cell: str) -> None:
    with pytest.raises(ValueError, match="is not a finite number"):
        parse_rating(cell)


def test_wide_layout(tmp_path: Path) -> None:
    path = _csv(tmp_path, "item,ana,ben\nq1,3,4\nq2,NA,2\n\nq3,pass,\n")
    assert load_ratings(path) == RatingsTable(
        units=("q1", "q2", "q3"),
        raters=("ana", "ben"),
        rows=((3, 4), (None, 2), ("pass", None)),
    )


def test_wide_layout_strips_a_utf8_bom_and_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes("﻿item , ana\r\n q1 , 2\r\n".encode())
    table = load_ratings(path)
    assert (table.units, table.raters, table.rows) == (("q1",), ("ana",), ((2,),))


def test_long_layout_orders_units_and_raters_by_first_appearance(tmp_path: Path) -> None:
    path = _csv(
        tmp_path,
        "note,Rating,Unit,Rater\nx,3,q2,ben\ny,4,q1,ana\nz,NA,q1,ben\nw,2,q2,ana\n",
    )
    assert load_ratings(path, layout="long") == RatingsTable(
        units=("q2", "q1"),
        raters=("ben", "ana"),
        rows=((3, 2), (None, 4)),
    )


def test_rejects_a_non_csv_suffix(tmp_path: Path) -> None:
    path = _csv(tmp_path, "item,a\nq1,1\n", name="ratings.tsv")
    assert _issues(path) == ["unsupported file type '.tsv'; use .csv"]


def test_rejects_a_missing_file(tmp_path: Path) -> None:
    [issue] = _issues(tmp_path / "absent.csv")
    assert issue.startswith("cannot read file: ")


def test_rejects_non_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "latin1.csv"
    path.write_bytes(b"item,a\nq\xe9,1\n")
    assert _issues(path) == ["cannot read file: not UTF-8 text"]


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ('item,a\nq1,"unterminated\nq2,1\n', "unexpected end of data"),
        ('item,a\nq1,"ab"c\n', "',' expected after '\"'"),
    ],
)
def test_rejects_malformed_quoting(tmp_path: Path, text: str, detail: str) -> None:
    assert _issues(_csv(tmp_path, text)) == [f"cannot parse CSV: {detail}"]


def test_rejects_an_empty_file(tmp_path: Path) -> None:
    path = _csv(tmp_path, "\n , \n")
    assert _issues(path) == ["the file is empty; expected a header row"]


def test_rejects_a_header_without_rows(tmp_path: Path) -> None:
    path = _csv(tmp_path, "item,ana,ben\n")
    assert _issues(path) == ["no ratings found below the header"]


def test_wide_header_problems(tmp_path: Path) -> None:
    assert _issues(_csv(tmp_path, "item\nq1\n")) == [
        "the header needs a unit column followed by one column per rater"
    ]
    assert _issues(_csv(tmp_path, "item,ana,,ana,ben,ben\nq1,1,2,3,4,5\n")) == [
        "every rater column needs a name in the header",
        "rater columns repeat: ana, ben",
    ]


def test_wide_row_problems(tmp_path: Path) -> None:
    path = _csv(tmp_path, "item,ana,ben\nq1,1,2\nq2,1\n,1,2\nq1,3,4\nq3,inf,1\n")
    assert _issues(path) == [
        "line 3: expected 3 cells, got 2",
        "line 4: the unit id is empty",
        "line 5: unit 'q1' appears twice",
        "line 6, column ana: 'inf' is not a finite number",
    ]


def test_long_layout_needs_its_columns(tmp_path: Path) -> None:
    path = _csv(tmp_path, "unit,who,score\nq1,ana,1\n")
    assert _issues(path, layout="long") == [
        "long layout needs columns unit, rater, rating; missing ['rater', 'rating']"
    ]


def test_long_row_problems(tmp_path: Path) -> None:
    path = _csv(
        tmp_path,
        "unit,rater,rating\nq1,ana,1\nq1,ben\nq2,,2\nq1,ana,3\nq2,ana,nan\nq3,ana,-inf\n",
    )
    assert _issues(path, layout="long") == [
        "line 3: expected 3 cells, got 2",
        "line 4: unit and rater must not be empty",
        "line 5: unit 'q1' already has a rating from 'ana'",
        "line 7, column rating: '-inf' is not a finite number",
    ]


def test_long_layout_keeps_units_whose_only_rating_is_missing(tmp_path: Path) -> None:
    path = _csv(tmp_path, "unit,rater,rating\nq1,ana,1\nq2,ana,NA\n")
    table = load_ratings(path, layout="long")
    assert table.units == ("q1", "q2")
    assert table.rows == ((1,), (None,))


def test_long_extra_cells_are_ignored(tmp_path: Path) -> None:
    path = _csv(tmp_path, "unit,rater,rating\nq1,ana,1,extra\n")
    assert load_ratings(path, layout="long").rows == ((1,),)


def test_issue_list_is_capped(tmp_path: Path) -> None:
    rows = "".join(f"q{i},inf\n" for i in range(15))
    issues = _issues(_csv(tmp_path, f"item,ana\n{rows}"))
    assert len(issues) == 11
    assert issues[-1] == "... and 5 more"
