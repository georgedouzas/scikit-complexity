"""Test the `_one_at_a_time` module of `skcomplex.experiments`."""

from skcomplex.experiments import OneAtATimeDesign


def test_one_at_a_time_design_moves_one_name_at_a_time():
    """Test that every pair of rows moves one name between its bounds with the others at their midpoints."""
    rows = OneAtATimeDesign().prepare({'a': (0.0, 2.0), 'b': (10.0, 20.0), 'c': (-1.0, 1.0)}).run().rows_
    assert len(rows) == 6
    assert rows[2] == {'a': 1.0, 'b': 10.0, 'c': 0.0}
    assert rows[3] == {'a': 1.0, 'b': 20.0, 'c': 0.0}
