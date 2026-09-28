"""Read rubric and score documents from YAML or JSON files.

Parsing is stricter than the libraries' defaults: a mapping with a repeated key is
an error rather than "last one wins", because a rubric file with two ``weight:``
lines for the same criterion would otherwise publish a silently different rubric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
