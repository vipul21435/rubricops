"""Read rubric and score documents (YAML or JSON) and ratings tables (CSV).

Parsing is stricter than the libraries' defaults: a mapping with a repeated key is
an error rather than "last one wins", because a rubric file with two ``weight:``
lines for the same criterion would otherwise publish a silently different rubric.
In the same spirit a ratings CSV with a repeated unit, or a repeated (unit, rater)
pair, is rejected instead of keeping one of the two ratings.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from rubricops.domain.rubric import Rubric

YAML_SUFFIXES = frozenset({".yaml", ".yml"})
JSON_SUFFIXES = frozenset({".json"})


class DocumentError(ValueError):
    """A file could not be read, parsed or validated; ``issues`` says why."""

    def __init__(self, path: Path, issues: list[str]) -> None:
        self.path = path
        self.issues = tuple(issues)
        super().__init__(f"{path}: " + "; ".join(issues))


_MERGE_TAG = "tag:yaml.org,2002:merge"


class _StrictSafeLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _StrictSafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    # flatten_mapping() resolves "<<" merge keys by prepending the merged pairs. Keys
    # written explicitly may override merged ones, but not repeat each other.
    explicit = sum(1 for key_node, _ in node.value if key_node.tag != _MERGE_TAG)
    loader.flatten_mapping(node)
    first_explicit = len(node.value) - explicit
    mapping: dict[Any, Any] = {}
    written: set[Any] = set()
    for index, (key_node, value_node) in enumerate(node.value):
        key = loader.construct_object(key_node, deep=True)
        if index >= first_explicit:
            if key in written:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            written.add(key)
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate key {key!r}"
            raise ValueError(msg)
        result[key] = value
    return result


def load_document(path: Path) -> object:
    """Parse ``path`` as YAML (``.yaml``/``.yml``) or JSON (``.json``)."""
    suffix = path.suffix.lower()
    if suffix not in YAML_SUFFIXES | JSON_SUFFIXES:
        raise DocumentError(path, [f"unsupported file type {suffix!r}; use .yaml, .yml or .json"])
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DocumentError(path, [f"cannot read file: {exc.strerror or exc}"]) from exc
    try:
        if suffix in JSON_SUFFIXES:
            return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        return yaml.load(text, Loader=_StrictSafeLoader)  # noqa: S506 - a SafeLoader subclass
    except (yaml.YAMLError, ValueError) as exc:
        detail = " ".join(str(exc).split())
        raise DocumentError(path, [f"cannot parse: {detail}"]) from exc


def _require_mapping(path: Path, data: object, what: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DocumentError(path, [f"expected a {what} mapping, got {type(data).__name__}"])
    return data


def format_validation_error(exc: ValidationError) -> list[str]:
    """Flatten a pydantic error into ``location: message`` lines."""
    lines = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "<document>"
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"{where}: {message}")
    return lines


def load_rubric(path: Path) -> Rubric:
    """Load and validate a rubric document."""
    data = _require_mapping(path, load_document(path), "rubric")
    try:
        return Rubric.model_validate(data)
    except ValidationError as exc:
        raise DocumentError(path, format_validation_error(exc)) from exc


def load_scores(path: Path) -> dict[str, object]:
    """Load a ``criterion id -> score`` mapping (values are validated when scoring)."""
    data = _require_mapping(path, load_document(path), "criterion -> score")
    return {str(key): value for key, value in data.items()}


RatingsLayout = Literal["wide", "long"]
MISSING_CELLS = frozenset({"", "na", "nan"})
"""Cell text (compared case-insensitively, after trimming) that means "not rated"."""
LONG_COLUMNS = ("unit", "rater", "rating")
_MAX_ISSUES = 10


@dataclass(frozen=True, slots=True)
class RatingsTable:
    """A units x raters table read from CSV; ``None`` marks a missing rating."""

    units: tuple[str, ...]
    raters: tuple[str, ...]
    rows: tuple[tuple[int | float | str | None, ...], ...]


def parse_rating(cell: str) -> int | float | str | None:
    """``"3"`` -> 3, ``"2.5"`` -> 2.5, ``"pass"`` -> ``"pass"``, blank/NA/NaN -> None.

    Raises ``ValueError`` for a non-finite number such as ``inf``.
    """
    text = cell.strip()
    if text.lower() in MISSING_CELLS:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        msg = f"{text!r} is not a finite number"
        raise ValueError(msg)
    return number


def _read_csv_rows(path: Path) -> list[tuple[int, list[str]]]:
    if path.suffix.lower() != ".csv":
        raise DocumentError(path, [f"unsupported file type {path.suffix!r}; use .csv"])
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        detail = exc.strerror if isinstance(exc, OSError) else "not UTF-8 text"
        raise DocumentError(path, [f"cannot read file: {detail or exc}"]) from exc
    # strict: an unterminated quote is an error instead of swallowing the rest of the file.
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        rows = [(reader.line_num, row) for row in reader]
    except csv.Error as exc:
        raise DocumentError(path, [f"cannot parse CSV: {exc}"]) from exc
    rows = [(line, row) for line, row in rows if any(cell.strip() for cell in row)]
    if not rows:
        raise DocumentError(path, ["the file is empty; expected a header row"])
    return rows


def _cell(issues: list[str], line: int, column: str, raw: str) -> int | float | str | None:
    try:
        return parse_rating(raw)
    except ValueError as exc:
        issues.append(f"line {line}, column {column}: {exc}")
        return None


def _read_wide(
    rows: list[tuple[int, list[str]]], issues: list[str]
) -> tuple[list[str], list[str], list[list[int | float | str | None]]]:
    header = [cell.strip() for cell in rows[0][1]]
    raters = header[1:]
    if not raters:
        issues.append("the header needs a unit column followed by one column per rater")
    if any(not name for name in raters):
        issues.append("every rater column needs a name in the header")
    repeated = sorted({name for name in raters if raters.count(name) > 1})
    if repeated:
        issues.append(f"rater columns repeat: {', '.join(repeated)}")
    units: list[str] = []
    table: list[list[int | float | str | None]] = []
    for line, row in rows[1:]:
        if len(row) != len(header):
            issues.append(f"line {line}: expected {len(header)} cells, got {len(row)}")
            continue
        unit = row[0].strip()
        if not unit:
            issues.append(f"line {line}: the unit id is empty")
        elif unit in units:
            issues.append(f"line {line}: unit {unit!r} appears twice")
        units.append(unit)
        table.append([_cell(issues, line, r, raw) for r, raw in zip(raters, row[1:], strict=True)])
    return units, raters, table


def _read_long(
    rows: list[tuple[int, list[str]]], issues: list[str]
) -> tuple[list[str], list[str], list[list[int | float | str | None]]]:
    header = [cell.strip().lower() for cell in rows[0][1]]
    absent = [name for name in LONG_COLUMNS if name not in header]
    if absent:
        issues.append(f"long layout needs columns {', '.join(LONG_COLUMNS)}; missing {absent}")
        return [], [], []
    at = [header.index(name) for name in LONG_COLUMNS]
    ratings: dict[tuple[str, str], int | float | str] = {}
    units: dict[str, None] = {}
    raters: dict[str, None] = {}
    for line, row in rows[1:]:
        if len(row) < len(header):
            issues.append(f"line {line}: expected {len(header)} cells, got {len(row)}")
            continue
        unit, rater, raw = (row[i].strip() for i in at)
        if not unit or not rater:
            issues.append(f"line {line}: unit and rater must not be empty")
            continue
        if (unit, rater) in ratings:
            issues.append(f"line {line}: unit {unit!r} already has a rating from {rater!r}")
            continue
        units.setdefault(unit)
        raters.setdefault(rater)
        value = _cell(issues, line, "rating", raw)
        if value is not None:
            ratings[unit, rater] = value
    table = [[ratings.get((unit, rater)) for rater in raters] for unit in units]
    return list(units), list(raters), table


def load_ratings(path: Path, *, layout: RatingsLayout = "wide") -> RatingsTable:
    """Read a ratings CSV into a units x raters table.

    ``wide``: a header row, then one row per unit; the first column is the unit id
    and every other column is a rater. ``long``: one row per rating with columns
    ``unit``, ``rater`` and ``rating`` (any order, other columns ignored). Blank,
    ``NA`` and ``NaN`` cells are missing ratings. Numbers are read as numbers and
    anything else as a text label.
    """
    rows = _read_csv_rows(path)
    issues: list[str] = []
    reader = _read_wide if layout == "wide" else _read_long
    units, raters, table = reader(rows, issues)
    if not issues and not units:
        issues.append("no ratings found below the header")
    if issues:
        extra = len(issues) - _MAX_ISSUES
        shown = issues[:_MAX_ISSUES] + ([f"... and {extra} more"] if extra > 0 else [])
        raise DocumentError(path, shown)
    return RatingsTable(
        units=tuple(units), raters=tuple(raters), rows=tuple(tuple(row) for row in table)
    )
