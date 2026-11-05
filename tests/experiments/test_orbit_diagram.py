"""Test orbit diagrams of a model over a range of a parameter."""

import numpy as np
import pytest

from skcomplex.errors import ValidationError
from skcomplex.experiments import OrbitDiagramExperiment
from skcomplex.models.mathematics.dynamical_systems import logistic_map


def _orbit_diagram(rates, **settings):
    """Return the orbit diagram of `map.x` of the logistic map over growth rates, run."""
    experiment = OrbitDiagramExperiment(logistic_map, 'map.r', rates, 'map.x', dt=1, **settings)
    return experiment.prepare().run()


@pytest.mark.slow
def test_period_doubling_of_logistic_map():
    """Test that the logistic map has period 1 below 3, 2 between 3 and 3.449, and 4 between 3.449 and 3.544."""
    rates = np.linspace(2.8, 4.0, 241)
    diagram = _orbit_diagram(rates, transient=3000, keep=64)
    periods = diagram.periods_
    assert len(diagram.points_) == 241
    assert np.all(periods[(rates > 2.8) & (rates < 2.98)] == 1)
    assert np.all(periods[(rates > 3.02) & (rates < 3.43)] == 2)
    assert np.all(periods[(rates > 3.47) & (rates < 3.53)] == 4)
    assert periods[rates > 3.9].min() > 32


def test_vectorised_orbit_diagram_matches_the_serial_sweep():
    """Test that the orbit diagram of the logistic map run as one ensemble equals the sweep of separate runs."""
    rates = np.linspace(2.8, 4.0, 200)
    serial = _orbit_diagram(rates, transient=200, keep=16)
    vectorised = _orbit_diagram(rates, transient=200, keep=16, vectorized=True)
    assert len(vectorised.points_) == 200
    for first, second in zip(serial.points_, vectorised.points_, strict=True):
        np.testing.assert_allclose(np.ravel(first), np.ravel(second), atol=1e-12)


def test_orbit_diagram_needs_kept_frames():
    """Test that keeping no frame raises an error naming keep."""
    with pytest.raises(ValidationError, match='keep = 0'):
        OrbitDiagramExperiment(logistic_map, 'map.r', [3.0], 'map.x', transient=10, keep=0).prepare()
