"""Test orbit diagrams and basin stability built from experiments."""

import numpy as np
import pytest

from skcomplex.core.analysis import basin_stability, orbit_diagram
from skcomplex.models.nonlinear import duffing, logistic_map


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


def side(trajectory):
    """Return the well an unforced double-well oscillator settles in."""
    return 'right' if trajectory['duffing.x'][-1] > 0 else 'left'


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
