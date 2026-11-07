"""Test the `_kinetic_crowd` module of `skcomplex.systems.social.crowds`."""

import numpy as np
import pytest

from skcomplex.systems.social.crowds import kinetic_crowd

from ..conftest import _run, _values


def _diagram(density, gamma, deceleration=0.95):
    """Return the mean speed of the fundamental diagram of Eq. 6.30."""
    free = gamma**2 * (1 - density) ** 2
    return gamma * free / (free + (1 - deceleration * gamma) * density**2)


@pytest.mark.parametrize(('walkers', 'stress'), [(120, 0.4), (280, 0.8)])
def test_homogeneous_crowd_follows_the_fundamental_diagram(walkers, stress):
    """Test that with one cell over the street the mean speed is the fundamental diagram of the density."""
    trajectory = _run(kinetic_crowd(walkers, stress=stress, cell=20.0), 300)
    speed = trajectory.values('decisions.mean_speed')[-100:].mean()
    assert speed == pytest.approx(_diagram(walkers / 400, stress), rel=0.1)


def test_walkers_stay_between_the_walls_and_panic_speeds_them_up():
    """Test that walkers stay in the street and walk faster after the panic raises the stress."""
    trajectory = _run(kinetic_crowd(panic_time=150, panic_stress=0.9), 300)
    for frame in (50, 150, 300):
        y = _values(trajectory, 'walkers', 'y', frame)
        assert np.all((y >= 0) & (y <= 5))
    speeds = trajectory.values('decisions.mean_speed')
    assert speeds[250:].mean() > speeds[100:150].mean() + 0.2
    band = trajectory.values('decisions.band_index')
    assert np.all((band[1:] >= 0) & (band[1:] <= 1))
