"""Test the `_networks` module of `skcomplex.models.biology.neuroscience`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.neuroscience import (
    delayed_lif_network,
    legion,
)


def test_delayed_lif_network_rates():
    """Test that uncoupled neurons fire at the leaky integrate-and-fire period and that inhibition slows them."""
    period = 20.0 * math.log((25.0 - 10.0) / (25.0 - 20.0))
    counts = []
    for J in (0.0, -2.0):
        simulation = Simulation(delayed_lif_network(5, J=J, probability=0.5), dt=1.0).run(until=60)
        events = simulation.trajectory_.events
        counts.append(len(events))
        if J == 0.0:
            index = list(simulation.problem_.states.names).index('neuron_0.v')
            times = [event.time for event in events if event.before[index] != event.after[index]]
            assert np.diff(times) == pytest.approx(period, rel=1e-3)
    assert counts[1] < counts[0]


def test_legion_segments_objects():
    """Test that LEGION synchronizes each object, alternates between objects, and keeps the background silent."""
    trajectory = Simulation(legion(epsilon=0.1), dt=0.5).run(until=120).trajectory_
    active = trajectory['oscillators.x'].reshape(len(trajectory.times), -1) > 0
    first, second = active[:, [0, 1, 4]], active[:, [11, 15]]
    assert first.mean() > 0.1
    assert second.mean() > 0.1
    assert (first[:, 0] == first[:, 1]).mean() > 0.97
    assert (second[:, 0] == second[:, 1]).mean() > 0.97
    after = trajectory.times > 20
    assert not (first.any(axis=1) & second.any(axis=1))[after].any()
    assert first[after].any()
    assert second[after].any()
    assert active[:, [2, 3, 5, 6, 7]].mean() < 0.02


def test_legion_defaults_fire_both_objects_in_a_short_run():
    """Test that both objects of the default LEGION network start an active phase within 100 time units."""
    trajectory = Simulation(legion(), dt=0.5).run(until=100).trajectory_
    active = trajectory['oscillators.x'].reshape(len(trajectory.times), -1) > 0
    for sites in ([0, 1, 4], [11, 15]):
        on = active[:, sites].any(axis=1)
        assert (on[1:] & ~on[:-1]).any()
