"""Test basin stability built from experiments."""

import pytest

from skcomplex.analysis import basin_stability
from skcomplex.models.physics.nonlinear_oscillators import duffing


def side(trajectory):
    """Return the well an unforced double-well oscillator settles in."""
    return 'right' if trajectory.values('duffing.x')[-1] > 0 else 'left'


def test_double_well_basins_are_symmetric():
    """Test that the damped double-well oscillator's two basins hold about half of the initial conditions each."""
    result = basin_stability(
        duffing(gamma=0.0),
        {'duffing.x': (-2.0, 2.0), 'duffing.x_d1': (-2.0, 2.0)},
        side,
        n=80,
        until=40,
        dt=0.5,
        random_state=0,
    )
    assert set(result.fractions) == {'left', 'right'}
    assert result.fractions['left'] == pytest.approx(0.5, abs=3 * result.errors['left'] + 0.05)


@pytest.mark.slow
def test_vectorised_basins_match_the_serial_runs():
    """Test that basin stability over 200 initial conditions run as one ensemble equals the serial runs."""
    bounds = {'duffing.x': (-2.0, 2.0), 'duffing.x_d1': (-2.0, 2.0)}
    settings = {'n': 200, 'until': 40, 'dt': 0.5, 'random_state': 0}
    serial = basin_stability(duffing(gamma=0.0), bounds, side, **settings)
    vectorised = basin_stability(duffing(gamma=0.0), bounds, side, vectorized=True, **settings)
    assert vectorised.fractions == serial.fractions
