"""Test the `_variability` module of `skcomplex.systems.physics.climate`."""

from skcomplex.systems.physics.climate import ENSOSystem, HasselmannSystem

from ..conftest import _run_default
from .conftest import _sign_changes


def test_hasselmann_has_stationary_variance():
    """Test that the stochastic climate fluctuates with the variance `sigma**2 / (2 * lam)`."""
    solver = HasselmannSystem.solver
    anomaly = _run_default(HasselmannSystem(lam=1.0, sigma=0.5), until=400.0, dt=0.5, solver=solver).get_values(
        'ocean.T',
    )[20:]
    assert abs(anomaly.var() - 0.125) < 0.04
    assert abs(anomaly.mean()) < 0.1


def test_enso_oscillates_with_period_longer_than_twice_the_delay():
    """Test that the delayed oscillator alternates between El Niño and La Niña with a period above twice the delay."""
    anomaly = _run_default(ENSOSystem(), until=200.0, dt=0.5).get_values('pacific.T')[100:]
    changes = _sign_changes(anomaly)
    assert changes >= 6
    assert 150.0 / (changes / 2) > 2 * 6.0
    assert anomaly.max() > 1.0
    assert anomaly.min() < -1.0
