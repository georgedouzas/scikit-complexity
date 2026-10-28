"""Test the `_sobol` module of `skcomplex.experiments`."""

from skcomplex.experiments import SobolDesign


def test_sobol_design_without_an_experiment():
    """Test that a Sobol design of 64 rows prepared on two bounds gives 64 rows within the bounds."""
    rows = SobolDesign(n=64).check().prepare({'r': (0.5, 1.5), 'K': (100, 1000)}).run()
    assert len(rows) == 64
    assert all(0.5 <= row['r'] <= 1.5 and 100 <= row['K'] <= 1000 for row in rows)


def test_sobol_design_is_seeded():
    """Test that the same seed gives the same scrambled sequence."""
    first = SobolDesign(n=8, random_state=3).check().prepare({'r': (0.0, 1.0)}).run()
    second = SobolDesign(n=8, random_state=3).check().prepare({'r': (0.0, 1.0)}).run()
    assert first == second
