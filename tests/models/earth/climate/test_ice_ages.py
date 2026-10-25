"""Test the `_ice_ages` module of `skcomplex.models.earth.climate`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.earth.climate import (
    carbon_cycle,
    saltzman_glaciation,
    van_der_pol_ice_ages,
)

from ..conftest import _run


def test_saltzman_glaciation_has_a_100_kyr_cycle():
    """Test that the unforced glacial model oscillates with a period of about 100 kyr."""
    trajectory = _run(saltzman_glaciation(), until=50.0, dt=0.05)
    ice = trajectory.values('climate.x')
    rising = trajectory.times[1:][(ice[:-1] < 0) & (ice[1:] >= 0)]
    periods = np.diff(rising[rising > 10.0])
    assert len(periods) >= 2
    assert np.allclose(periods, 10.6, atol=0.5)


def test_carbon_cycle_is_excitable():
    """Test that a small loss of dissolved carbon decays while a larger one sets off a large excursion."""
    small = _run(carbon_cycle(w=2150.0), until=20.0, dt=0.1)
    large = _run(carbon_cycle(w=2050.0), until=20.0, dt=0.1)
    for trajectory in (small, large):
        assert np.allclose(
            [trajectory.values('ocean.c')[-1], trajectory.values('ocean.w')[-1]], [83.58, 2197.37], atol=0.1
        )
    assert np.ptp(small.values('ocean.c')) < 20.0
    assert np.ptp(large.values('ocean.c')) > 100.0


def test_van_der_pol_ice_ages_cycle_every_hundred_thousand_years():
    """Test that the unforced ice-age oscillator settles on a relaxation cycle of about 100 thousand years."""
    trajectory = Simulation(van_der_pol_ice_ages(), dt=0.5).run(until=400.0).trajectory_
    climate = np.asarray(trajectory.values('climate.y'), dtype=float)
    times = np.asarray(trajectory.times)
    crossings = times[np.flatnonzero((climate[:-1] < 0) & (climate[1:] >= 0))]
    assert len(crossings) >= 3
    assert np.allclose(np.diff(crossings), 100.0, rtol=0.05)
