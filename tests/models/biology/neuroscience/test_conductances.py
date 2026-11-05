"""Test the `_conductances` module of `skcomplex.models.biology.neuroscience`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.neuroscience import (
    fitzhugh_nagumo_ode,
    hodgkin_huxley,
)

from ..conftest import TOLERANCE

HODGKIN_HUXLEY_THRESHOLD = 6.2


def _spike_times(trajectory):
    """Return the times of the spike events of a trajectory."""
    return np.array([event.time for event in trajectory.events if event.label == 'spike'])


def test_hodgkin_huxley_spikes():
    """Test that Hodgkin-Huxley fires periodically above a threshold current within 10% of the published 6.2."""
    trajectory = Simulation(hodgkin_huxley(I=10.0), dt=0.5).prepare().run(until=100).trajectory_
    intervals = np.diff(_spike_times(trajectory))[1:]
    assert len(intervals) >= 4
    assert np.ptp(intervals) < 0.01 * intervals.mean()
    below = Simulation(hodgkin_huxley(I=(1 - TOLERANCE) * HODGKIN_HUXLEY_THRESHOLD), dt=0.5).prepare().run(until=150)
    above = Simulation(hodgkin_huxley(I=(1 + TOLERANCE) * HODGKIN_HUXLEY_THRESHOLD), dt=0.5).prepare().run(until=150)
    assert len(_spike_times(below.trajectory_)) <= 1
    assert len(_spike_times(above.trajectory_)) >= 6


def test_fitzhugh_nagumo_neuron_needs_current_to_oscillate():
    """Test that the FitzHugh-Nagumo neuron oscillates with current and rests without."""
    amplitudes = []
    for current in (0.0, 0.5):
        trajectory = Simulation(fitzhugh_nagumo_ode(I=current), dt=0.5).prepare().run(until=200).trajectory_
        amplitudes.append(np.ptp(trajectory.values('neuron.v')[trajectory.times > 100]))
    assert amplitudes[0] < 0.01
    assert amplitudes[1] > 3
