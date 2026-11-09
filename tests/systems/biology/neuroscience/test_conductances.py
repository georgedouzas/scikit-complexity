"""Test the `_conductances` module of `skcomplex.systems.biology.neuroscience`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.neuroscience import FitzhughNagumoODESystem, HodgkinHuxleySystem

from ..conftest import TOLERANCE

HODGKIN_HUXLEY_THRESHOLD = 6.2


def _spike_times(trajectory):
    """Return the times of the spike events of a trajectory."""
    return np.array([event.time for event in trajectory.events if event.label == 'spike'])


def test_hodgkin_huxley_spikes():
    """Test that Hodgkin-Huxley fires periodically above a threshold current within 10% of the published 6.2."""
    trajectory = simulate(HodgkinHuxleySystem(I=10.0), dt=0.5, until=100)
    intervals = np.diff(_spike_times(trajectory))[1:]
    assert len(intervals) >= 4
    assert np.ptp(intervals) < 0.01 * intervals.mean()
    below = simulate(
        HodgkinHuxleySystem(I=(1 - TOLERANCE) * HODGKIN_HUXLEY_THRESHOLD).prepare().system_,
        until=150,
        dt=0.5,
    )
    above = simulate(
        HodgkinHuxleySystem(I=(1 + TOLERANCE) * HODGKIN_HUXLEY_THRESHOLD).prepare().system_,
        until=150,
        dt=0.5,
    )
    assert len(_spike_times(below)) <= 1
    assert len(_spike_times(above)) >= 6


def test_fitzhugh_nagumo_neuron_needs_current_to_oscillate():
    """Test that the FitzHugh-Nagumo neuron oscillates with current and rests without."""
    amplitudes = []
    for current in (0.0, 0.5):
        trajectory = simulate(FitzhughNagumoODESystem(I=current), dt=0.5, until=200)
        amplitudes.append(np.ptp(trajectory.get_values('neuron.v')[trajectory.times > 100]))
    assert amplitudes[0] < 0.01
    assert amplitudes[1] > 3
