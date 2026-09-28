"""Encoding ratings into ReliabilityData: labels, categories, missing values, errors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rubricops.stats.data import MISSING, AgreementInputError, ReliabilityData


def test_rows_are_encoded_against_sorted_numeric_categories() -> None:
    data = ReliabilityData.from_rows([[3, 1], [2, None], [1, 3]])
    assert data.categories == (1, 2, 3)
    assert data.codes.tolist() == [[2, 0], [1, MISSING], [0, 2]]
    assert (data.n_units, data.n_raters, data.n_categories) == (3, 2, 3)
    assert data.is_numeric
    assert data.scale_values().tolist() == [1.0, 2.0, 3.0]


def test_text_labels_are_sorted_and_have_no_scale_values() -> None:
    data = ReliabilityData.from_rows([["pass", "fail"], ["fail", "fail"]])
    assert data.categories == ("fail", "pass")
    assert not data.is_numeric
    with pytest.raises(AgreementInputError, match="needs numeric ratings"):
        data.scale_values()


def test_equal_numbers_fold_to_one_category() -> None:
    data = ReliabilityData.from_rows([[2, 2.0], [np.int64(2), np.float64(2.5)]])
    assert data.categories == (2, 2.5)
    assert data.codes.tolist() == [[0, 0], [0, 1]]


def test_numpy_arrays_with_nan_and_numpy_strings_are_accepted() -> None:
    numeric = ReliabilityData.from_rows(np.array([[1.0, np.nan], [2.0, 1.0]]))
    assert numeric.codes.tolist() == [[0, MISSING], [1, 0]]
    text = ReliabilityData.from_rows(np.array([["a", "b"], ["b", "b"]]))
    assert text.categories == ("a", "b")
    assert all(type(label) is str for label in text.categories)


def test_explicit_categories_keep_their_order_and_unused_levels() -> None:
    data = ReliabilityData.from_rows([["low", "high"]], categories=["low", "mid", "high"])
    assert data.categories == ("low", "mid", "high")
    assert data.codes.tolist() == [[0, 2]]


def test_mixed_labels_sort_numbers_before_text() -> None:
    data = ReliabilityData.from_rows([[1, "n/a"], ["1", 0.5]])
    assert data.categories == (0.5, 1, "1", "n/a")
    assert not data.is_numeric


def test_from_columns_transposes_rater_lists() -> None:
    data = ReliabilityData.from_columns(["a", "b", "a"], ["a", "a", None])
    assert data.n_units == 3
    assert data.n_raters == 2
    assert data.codes[:, 1].tolist() == [0, 0, MISSING]


def test_take_resamples_units_and_keeps_categories() -> None:
    data = ReliabilityData.from_rows([[1, 1], [2, 3]])
    again = data.take(np.array([1, 1], dtype=np.int64))
    assert again.codes.tolist() == [[1, 2], [1, 2]]
    assert again.categories == data.categories
    assert again.scale_values().tolist() == [1.0, 2.0, 3.0]


def test_unit_counts_ignore_missing_ratings() -> None:
    data = ReliabilityData.from_rows([["a", "a", None], ["b", "a", "b"], [None, None, None]])
    assert data.unit_counts().tolist() == [[2, 0], [1, 2], [0, 0]]


def test_empty_table() -> None:
    data = ReliabilityData.from_rows([])
    assert (data.n_units, data.n_raters, data.n_categories) == (0, 0, 0)
    assert not data.is_numeric
    assert data.unit_counts().shape == (0, 0)


@pytest.mark.parametrize(
    ("rows", "categories", "message"),
    [
        ([[1, 2], [1]], None, r"one entry per rater \(2\); rows \[1\] differ"),
        ([[True, False]], None, "boolean rating"),
        ([[np.bool_(True), 1]], None, "boolean rating"),
        ([[math.inf, 1]], None, "not a finite number"),
        ([[[1], [2]]], None, "not a hashable label"),
        ([[1, 2]], [1, 1, 2], "must not repeat"),
        ([[1, 4]], [1, 2, 3], r"ratings \[4\] are not in the given categories"),
    ],
)
def test_malformed_input_is_rejected(
    rows: list[list[object]], categories: list[object] | None, message: str
) -> None:
    with pytest.raises(AgreementInputError, match=message):
        ReliabilityData.from_rows(rows, categories=categories)


def test_from_columns_rejects_unequal_lengths() -> None:
    with pytest.raises(AgreementInputError, match=r"got lengths \[2, 3\]"):
        ReliabilityData.from_columns([1, 2], [1, 2, 3])
