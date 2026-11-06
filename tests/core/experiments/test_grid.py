"""Test the `_grid` module of `skcomplex.core.experiments`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import GridDesign


def test_grid_design_runs_every_combination():
    """Test that a grid design gives every combination of the values, and that running keeps its rows."""
    design = GridDesign().prepare({'r': [1, 2], 'K': [10, 20, 30]})
    assert len(design.rows_) == 6
    assert design.run().rows_[0] == {'r': 1, 'K': 10}


def test_grid_design_rejects_a_value_that_is_not_a_sequence():
    """Test that a grid design prepared on one number instead of a sequence raises an error naming it."""
    with pytest.raises(ValidationError, match='sequence of values per name, but r'):
        GridDesign().prepare({'r': 0.5})
