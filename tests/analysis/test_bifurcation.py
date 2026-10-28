"""Test orbit diagrams built from experiments."""

import numpy as np
import pytest

from skcomplex.analysis import orbit_diagram
from skcomplex.models.mathematics.dynamical_systems import logistic_map


@pytest.mark.slow
def test_period_doubling_of_logistic_map():
    """Test that the logistic map has period 1 below 3, 2 between 3 and 3.449, and 4 between 3.449 and 3.544."""
    rates = np.linspace(2.8, 4.0, 241)
    diagram = orbit_diagram(logistic_map, 'map.r', rates, 'map.x', transient=3000, keep=64, dt=1)
    periods = diagram.periods()
    assert len(diagram.points) == 241
    assert np.all(periods[(rates > 2.8) & (rates < 2.98)] == 1)
    assert np.all(periods[(rates > 3.02) & (rates < 3.43)] == 2)
    assert np.all(periods[(rates > 3.47) & (rates < 3.53)] == 4)
    assert periods[rates > 3.9].min() > 32


def test_vectorised_orbit_diagram_matches_the_serial_sweep():
    """Test that the orbit diagram of the logistic map run as one ensemble equals the sweep of separate runs."""
    rates = np.linspace(2.8, 4.0, 200)
    serial = orbit_diagram(logistic_map, 'map.r', rates, 'map.x', transient=200, keep=16, dt=1)
    vectorised = orbit_diagram(logistic_map, 'map.r', rates, 'map.x', transient=200, keep=16, dt=1, vectorized=True)
    assert len(vectorised.points) == 200
    for first, second in zip(serial.points, vectorised.points, strict=True):
        np.testing.assert_allclose(np.ravel(first), np.ravel(second), atol=1e-12)
