"""Test the `_echo_state` module of `skcomplex.models.computer_science.neural_networks`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.computer_science.neural_networks import echo_state_network


def test_echo_state_network_forgets_its_initial_state():
    """Test that two runs of the reservoir from different initial states converge, the echo state property."""
    system = echo_state_network(100, spectral_radius=0.9)
    first = Simulation(system).check().prepare().run(until=200).trajectory_.values('units.r')
    other = np.random.default_rng(1).uniform(-1.0, 1.0, 100)
    second = (
        Simulation(system, initial_state={'units.r': other})
        .check()
        .prepare()
        .run(until=200)
        .trajectory_.values('units.r')
    )
    distance = np.linalg.norm(first - second, axis=1)
    assert distance[-1] < 1e-6 * distance[0]
    assert np.all(np.diff(distance[::20]) < 0)
