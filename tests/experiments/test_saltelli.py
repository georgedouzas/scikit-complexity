"""Test the `_saltelli` module of `skcomplex.experiments`."""

from skcomplex.experiments import SaltelliDesign


def test_saltelli_design_crosses_two_samples():
    """Test that Saltelli's sample of `n` base points over three names has `n * 5` rows within the bounds."""
    bounds = {'a': (0.0, 1.0), 'b': (10.0, 20.0), 'c': (-1.0, 1.0)}
    rows = SaltelliDesign(n=16, random_state=0).check().prepare(bounds).run()
    assert len(rows) == 16 * 5
    assert all(low <= row[name] <= high for row in rows for name, (low, high) in bounds.items())
