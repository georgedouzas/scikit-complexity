"""Test the `_latin_hypercube` module of `skcomplex.experiments`."""

from skcomplex.experiments import LatinHypercubeDesign


def test_latin_hypercube_design_fills_every_stratum():
    """Test that a Latin hypercube of 16 rows puts one value in every sixteenth of the bounds."""
    rows = LatinHypercubeDesign(n=16, random_state=0).check().check().prepare({'r': (0.2, 1.0)}).run().rows_
    assert sorted(int((row['r'] - 0.2) / 0.8 * 16) for row in rows) == list(range(16))
