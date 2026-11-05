"""Test the `_rows` module of `skcomplex.experiments`."""

import pytest

from skcomplex.errors import ValidationError
from skcomplex.experiments import RowsDesign


def test_rows_design_returns_its_rows():
    """Test that a rows design prepared without bounds returns its rows."""
    assert RowsDesign([{'K': 10}, {'K': 20}]).prepare().run().rows_ == [{'K': 10}, {'K': 20}]


def test_rows_design_rejects_bounds_and_rows_that_are_not_mappings():
    """Test that a row that is not a mapping fails the check and that bounds fail the preparation."""
    with pytest.raises(ValidationError, match='mapping from name to value'):
        RowsDesign([1.0]).prepare()
    with pytest.raises(ValidationError, match='not from the bounds'):
        RowsDesign([{'K': 10}]).prepare({'r': (0.0, 1.0)})
