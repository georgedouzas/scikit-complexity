"""Test the `_reactors` module of `skcomplex.models.chemistry.kinetics`."""

import numpy as np

from skcomplex.models.chemistry.kinetics import (
    catalyst_pellet,
    pulsed_electrochemical_reactor,
)

from .conftest import _run


def test_catalyst_pellet_is_bistable():
    """Test that a catalyst pellet settles on a reacting or an inhibited state depending on its start."""
    reacting = _run(catalyst_pellet(c=0.0), until=200.0, dt=10.0)
    inhibited = _run(catalyst_pellet(c=1.0), until=200.0, dt=10.0)
    for trajectory in (reacting, inhibited):
        c = trajectory.values('pellet.c')[-1]
        assert abs(0.1 * (1 - c) - 10.0 * c / (1 + 20.0 * c) ** 2) < 1e-6
    assert reacting.values('pellet.c')[-1] < 0.05
    assert inhibited.values('pellet.c')[-1] > 0.5


def test_pulsed_electrochemical_reactor_follows_the_pulses():
    """Test that every pulse raises the overpotential and depletes the reactant, and the response becomes periodic."""
    trajectory = _run(pulsed_electrochemical_reactor(), until=100.0, dt=0.5)
    times, eta, c = trajectory.times, trajectory.values('electrode.eta'), trajectory.values('electrode.c')
    starts, ends, recovered = (np.isin(times, 10.0 * np.arange(6) + phase) for phase in (0.0, 2.0, 9.5))
    assert (eta[ends] > eta[starts]).all()
    assert (c[ends] < c[starts]).all()
    assert (c[recovered] > c[ends]).all()
    assert np.isclose(c[times == 99.5][0], c[times == 89.5][0], atol=1e-3)
