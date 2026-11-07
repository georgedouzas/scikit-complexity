"""Test the `_orbit_diagram` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import orbit_diagram
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem


def _orbit_diagram(rates, **settings):
    """Return the orbit diagram of `map.x` of the logistic map over growth rates."""
    return orbit_diagram(LogisticMapSystem(), 'map.r', rates, 'map.x', dt=1, **settings)


@pytest.mark.slow
def test_period_doubling_of_logistic_map():
    """Test that the logistic map has period 1 below 3, 2 between 3 and 3.449, and 4 between 3.449 and 3.544."""
    rates = np.linspace(2.8, 4.0, 241)
    diagram = _orbit_diagram(rates, transient=3000, keep=64)
    periods = diagram.periods
    assert len(diagram.points) == 241
    assert np.all(periods[(rates > 2.8) & (rates < 2.98)] == 1)
    assert np.all(periods[(rates > 3.02) & (rates < 3.43)] == 2)
    assert np.all(periods[(rates > 3.47) & (rates < 3.53)] == 4)
    assert periods[rates > 3.9].min() > 32


def test_vectorised_orbit_diagram_matches_the_serial_sweep():
    """Test that the orbit diagram of the logistic map run as one ensemble equals the sweep of separate runs."""
    rates = np.linspace(2.8, 4.0, 200)
    serial = _orbit_diagram(rates, transient=200, keep=16)
    vectorised = _orbit_diagram(rates, transient=200, keep=16, vectorized=True)
    assert len(vectorised.points) == 200
    for first, second in zip(serial.points, vectorised.points, strict=True):
        np.testing.assert_allclose(np.ravel(first), np.ravel(second), atol=1e-12)


def test_orbit_diagram_of_a_model_parameter():
    """Test that a model parameter sweeps like the dotted path of the parameter it sets."""
    by_parameter = orbit_diagram(LogisticMapSystem(), 'r', [2.9, 3.2], 'map.x', transient=1000, keep=8)
    by_path = orbit_diagram(LogisticMapSystem(), 'map.r', [2.9, 3.2], 'map.x', transient=1000, keep=8)
    assert by_parameter.periods.tolist() == by_path.periods.tolist() == [1, 2]


def test_orbit_diagram_needs_kept_frames_and_values():
    """Test that keeping no frame or giving no value raises an error naming it."""
    with pytest.raises(ValidationError, match='keep = 0'):
        orbit_diagram(LogisticMapSystem(), 'map.r', [3.0], 'map.x', transient=10, keep=0)
    with pytest.raises(ValidationError, match='at least one value'):
        orbit_diagram(LogisticMapSystem(), 'map.r', [], 'map.x', transient=10, keep=1)
