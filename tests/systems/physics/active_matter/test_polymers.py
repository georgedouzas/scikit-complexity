"""Test the `_polymers` module of `skcomplex.systems.physics.active_matter`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.active_matter import ActivePolymerSystem

DURATION = 600.0


def _chain(v0):
    """Return the coordinates of the beads of a run of an active polymer, with frames every half time unit."""
    trajectory = simulate(ActivePolymerSystem(v0=v0), until=DURATION, random_state=0)
    return np.asarray(trajectory.get_values('chain.x')), np.asarray(trajectory.get_values('chain.y'))


@pytest.mark.parametrize('v0', [0.0, 1.0])
def test_active_polymer_centre_of_mass_moves_as_an_active_particle_of_n_times_less_noise(v0):
    """Test the mean squared displacement of the centre of mass over 5 time units against its active Brownian form."""
    x, y = _chain(v0)
    centre_x, centre_y = x.mean(axis=1), y.mean(axis=1)
    lag, interval = 10, 5.0
    measured = np.mean((centre_x[lag:] - centre_x[:-lag]) ** 2 + (centre_y[lag:] - centre_y[:-lag]) ** 2)
    expected = (4 * 0.1 * interval + 2 * v0**2 * (interval - 1 + math.exp(-interval))) / 20
    assert measured == pytest.approx(expected, rel=0.3)


def test_active_polymer_swells_above_the_rouse_size():
    """Test that a passive chain has the Rouse size `2 D_t (N - 1) / k` and that self-propulsion swells it."""
    sizes = []
    for v0 in (0.0, 1.0):
        x, y = _chain(v0)
        sizes.append(np.mean(((x[:, -1] - x[:, 0]) ** 2 + (y[:, -1] - y[:, 0]) ** 2)[40:]))
    assert sizes[0] == pytest.approx(2 * 0.1 * 19 / 10, rel=0.3)
    assert sizes[1] > 2 * sizes[0]
