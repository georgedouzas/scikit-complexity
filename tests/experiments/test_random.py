"""Test the `_random` module of `skcomplex.experiments`."""

from skcomplex.experiments import RandomDesign


def test_random_design_stays_in_bounds():
    """Test that a random design gives the requested number of rows within the bounds."""
    values = [
        row['r'] for row in RandomDesign(n=16, random_state=0).check().check().prepare({'r': (0.2, 1.0)}).run().rows_
    ]
    assert len(values) == 16
    assert all(0.2 <= value <= 1.0 for value in values)
