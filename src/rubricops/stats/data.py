"""Reliability data: a units x raters table of ratings, encoded once for fast statistics.

Every agreement statistic in :mod:`rubricops.stats.agreement` reads the same shape:
one row per *unit* (the item being rated) and one column per *rater*, with missing
ratings allowed. :class:`ReliabilityData` encodes the labels to integer codes against
a fixed, ordered category list, so a bootstrap can resample rows cheaply while every
resample keeps the category set (and therefore the weights) of the full data.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import numpy.typing as npt

MISSING = -1
"""The code stored for a missing rating."""

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


class AgreementInputError(ValueError):
    """Ratings that no agreement statistic can be computed from (bad shape or labels).

    Degenerate but well-formed data (one category, no pairable units) is not an
    error: the statistic returns NaN with a reason instead.
    """


def is_missing(value: object) -> bool:
    """``None`` and float NaN (including numpy floats) mark a missing rating."""
    if value is None:
        return True
    if isinstance(value, float | np.floating):
        return bool(np.isnan(value))
    return False


def _label(value: object) -> Hashable:
    """Validate one rating and fold equal numbers (``2``, ``2.0``, ``np.int64(2)``)."""
    if isinstance(value, bool | np.bool_):
        msg = f"boolean rating {value!r} is ambiguous; use 0/1 or text labels"
        raise AgreementInputError(msg)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            msg = f"rating {value!r} is not a finite number"
            raise AgreementInputError(msg)
        return int(number) if number.is_integer() else number
    if isinstance(value, np.generic):
        return _label(value.item())
    try:
        hash(value)
    except TypeError:
        msg = f"rating {value!r} is not a hashable label"
        raise AgreementInputError(msg) from None
    return value


def _sort_key(label: Hashable) -> tuple[int, float, str]:
    if isinstance(label, int | float):
        return (0, float(label), "")
    return (1, 0.0, f"{type(label).__name__}:{label}")


@dataclass(frozen=True, slots=True, eq=False)
class ReliabilityData:
    """Encoded ratings: ``codes[u, r]`` indexes ``categories`` or is ``MISSING``.

    Build it with :meth:`from_rows` (a units x raters table) or :meth:`from_columns`
    (one sequence per rater). ``categories`` is the ordered category list: the
    explicit one when given, otherwise the distinct observed labels, numbers sorted
    numerically and other labels sorted by text.
    """

    codes: IntArray
    categories: tuple[Hashable, ...]
    _values: FloatArray | None = field(default=None, repr=False)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[object]] | npt.NDArray[np.generic],
        *,
        categories: Sequence[object] | None = None,
    ) -> ReliabilityData:
        """Encode a units x raters table; ``None`` or NaN marks a missing rating."""
        table = [list(row) for row in rows]
        width = len(table[0]) if table else 0
        ragged = [i for i, row in enumerate(table) if len(row) != width]
        if ragged:
            msg = f"every unit needs one entry per rater ({width}); rows {ragged[:5]} differ"
            raise AgreementInputError(msg)
        labels = [[None if is_missing(v) else _label(v) for v in row] for row in table]
        observed = {label for row in labels for label in row if label is not None}
        if categories is None:
            ordered = tuple(sorted(observed, key=_sort_key))
        else:
            ordered = tuple(_label(label) for label in categories)
            if len(set(ordered)) != len(ordered):
                msg = "categories must not repeat"
                raise AgreementInputError(msg)
            unknown = sorted(observed - set(ordered), key=_sort_key)
            if unknown:
                msg = f"ratings {unknown[:5]} are not in the given categories {list(ordered)}"
                raise AgreementInputError(msg)
        index = {label: code for code, label in enumerate(ordered)}
        codes = np.array(
            [[MISSING if label is None else index[label] for label in row] for row in labels],
            dtype=np.int64,
        ).reshape(len(table), width)
        values = None
        if ordered and all(isinstance(label, int | float) for label in ordered):
            values = np.array(ordered, dtype=np.float64)
        return cls(codes=codes, categories=ordered, _values=values)

    @classmethod
    def from_counts(
        cls,
        counts: Sequence[Sequence[int]] | IntArray,
        *,
        categories: Sequence[object] | None = None,
    ) -> ReliabilityData:
        """Expand a units x categories count table (the layout Fleiss' kappa is often
        published in) into ratings. Rater columns are anonymous; a unit with fewer
        ratings than the widest one is padded with missing entries.

        ``categories`` names the columns; by default they are ``0 .. k-1``.
        """
        table = np.asarray(counts)
        if table.ndim != 2 or not np.issubdtype(table.dtype, np.integer) or (table < 0).any():
            msg = "counts must be a units x categories table of non-negative integers"
            raise AgreementInputError(msg)
        names = list(range(table.shape[1])) if categories is None else list(categories)
        if len(names) != table.shape[1]:
            msg = f"got {len(names)} category names for {table.shape[1]} count columns"
            raise AgreementInputError(msg)
        width = int(table.sum(axis=1).max()) if table.size else 0
        rows: list[list[object]] = []
        for unit in table:
            ratings: list[object] = [
                name for name, n in zip(names, unit.tolist(), strict=True) for _ in range(n)
            ]
            rows.append(ratings + [None] * (width - len(ratings)))
        return cls.from_rows(rows, categories=names)

    @classmethod
    def from_columns(
        cls,
        *columns: Sequence[object],
        categories: Sequence[object] | None = None,
    ) -> ReliabilityData:
        """Encode one equal-length sequence per rater (``from_columns(a, b)``)."""
        lengths = sorted({len(column) for column in columns})
        if len(lengths) > 1:
            msg = f"every rater needs one entry per unit; got lengths {lengths}"
            raise AgreementInputError(msg)
        return cls.from_rows(list(zip(*columns, strict=True)), categories=categories)

    @property
    def n_units(self) -> int:
        return int(self.codes.shape[0])

    @property
    def n_raters(self) -> int:
        return int(self.codes.shape[1])

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    @property
    def is_numeric(self) -> bool:
        """Whether every category is a finite number (needed for interval metrics)."""
        return self._values is not None

    def scale_values(self) -> FloatArray:
        """Category values as floats, in category order; raises for text labels."""
        if self._values is None:
            text = [label for label in self.categories if not isinstance(label, int | float)]
            msg = f"this metric needs numeric ratings; found labels {text[:5]}"
            raise AgreementInputError(msg)
        return self._values

    def take(self, units: IntArray) -> ReliabilityData:
        """The rows ``units`` (repeats allowed), keeping the full category list."""
        return ReliabilityData(
            codes=self.codes[units], categories=self.categories, _values=self._values
        )

    def unit_counts(self) -> IntArray:
        """``counts[u, k]``: how many raters gave unit ``u`` category ``k``."""
        counts = np.zeros((self.n_units, self.n_categories), dtype=np.int64)
        present = self.codes != MISSING
        units, _ = np.nonzero(present)
        np.add.at(counts, (units, self.codes[present]), 1)
        return counts
