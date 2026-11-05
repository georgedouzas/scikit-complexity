"""Test the `_neurons` module of `skcomplex.models.biology.neuroscience`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.biology.neuroscience import integrate_and_fire, izhikevich


def test_izhikevich_regular_spiking():
    """Test that the regular-spiking Izhikevich neuron with input 10 fires periodically and records every reset."""
    trajectory = Simulation(izhikevich(a=0.02, b=0.2, c=-65, d=8, I=10), dt=0.1).prepare().run(until=400).trajectory_
    spikes = [event for event in trajectory.events if event.label == 'spike']
    intervals = np.diff([event.time for event in spikes])
    assert len(spikes) > 5
    assert np.ptp(intervals[2:]) < 0.1
    assert all(event.after[0] == -65 and event.after[1] == pytest.approx(event.before[1] + 8) for event in spikes)


def test_integrate_and_fire_period():
    """Test that the leaky integrate-and-fire neuron spikes with the analytic period at tight solver tolerances."""
    solver = ScipyODESolver(rtol=1e-9, atol=1e-11)
    trajectory = Simulation(integrate_and_fire(), dt=0.1, solver=solver).prepare().run(until=100).trajectory_
    intervals = np.diff([0.0] + [event.time for event in trajectory.events])
    assert intervals == pytest.approx(10 * math.log(20 / 5), abs=1e-5)
