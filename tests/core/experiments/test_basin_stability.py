"""Test the `_basin_stability` module of `skcomplex.core.experiments`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import basin_stability
from skcomplex.systems.physics.nonlinear_oscillators import DuffingSystem

BOUNDS = {'duffing.x': (-2.0, 2.0), 'duffing.x_d1': (-2.0, 2.0)}


def side(trajectory):
    """Return the well an unforced double-well oscillator settles in."""
    return 'right' if trajectory.values('duffing.x')[-1] > 0 else 'left'


def test_double_well_basins_are_symmetric():
    """Test that the damped double-well oscillator's two basins hold about half of the initial states each."""
    result = basin_stability(DuffingSystem(gamma=0.0), BOUNDS, side, n=80, until=40, dt=0.5, random_state=0)
    assert set(result.fractions) == {'left', 'right'}
    assert result.fractions['left'] == pytest.approx(0.5, abs=3 * result.errors['left'] + 0.05)
    assert len(result.labels) == 80
    assert len(result.rows) == 80


@pytest.mark.slow
def test_vectorised_basins_match_the_serial_runs():
    """Test that basin stability over 200 initial states run as one ensemble equals the serial runs."""
    settings = {'n': 200, 'until': 40, 'dt': 0.5, 'random_state': 0}
    serial = basin_stability(DuffingSystem(gamma=0.0), BOUNDS, side, **settings)
    vectorised = basin_stability(DuffingSystem(gamma=0.0), BOUNDS, side, vectorized=True, **settings)
    assert vectorised.fractions == serial.fractions


def test_invalid_number_of_initial_states_and_bounds():
    """Test that zero initial states or no bounds raise errors naming them."""
    with pytest.raises(ValidationError, match='n = 0'):
        basin_stability(DuffingSystem(gamma=0.0), BOUNDS, side, n=0)
    with pytest.raises(ValidationError, match='bounds of at least one initial state'):
        basin_stability(DuffingSystem(gamma=0.0), {}, side)
