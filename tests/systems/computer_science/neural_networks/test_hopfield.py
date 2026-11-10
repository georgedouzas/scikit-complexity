"""Test the `_hopfield` module of `skcomplex.systems.computer_science.neural_networks`."""

import numpy as np

from skcomplex.systems.computer_science.neural_networks import HopfieldSystem

from ..conftest import _run


def test_hopfield_recalls_a_stored_pattern_and_lowers_its_energy():
    """Test that the network recalls the first pattern from a corrupted copy, with an energy that never increases."""
    rng = np.random.default_rng(1)
    patterns = rng.choice([-1.0, 1.0], size=(3, 64))
    trajectory = _run(HopfieldSystem(patterns, noise=0.2, random_state=1), 5)
    snapshots = [trajectory.get_agents('neurons', frame) for frame in range(6)]
    weights = snapshots[0].values['weights']
    spins = [snapshot.values['spin'] for snapshot in snapshots]
    energies = [-spin @ weights @ spin / 2 for spin in spins]
    assert np.all(np.diff(energies) <= 1e-12)
    assert spins[0] @ patterns[0] < 64
    assert spins[-1] @ patterns[0] == 64
