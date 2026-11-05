"""Test the `_automata` module of `skcomplex.models.computer_science.automata`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.computer_science.automata import langtons_ant


def test_langtons_ant_moves_and_flips():
    """Test that Langton's ant keeps one ant, flips one cell per step, and builds its highway after 10,000 steps."""
    trajectory = Simulation(langtons_ant((100, 100)), dt=100).prepare().run(until=11_000).trajectory_
    ants = trajectory.lattice('cells.ant')
    assert np.all((ants >= 0).sum(axis=(1, 2)) == 1)
    black = trajectory.lattice('cells.color').sum(axis=(1, 2))
    assert black[1] <= 100
    position = np.argwhere(ants[-1] >= 0)[0]
    assert np.abs(position - 50).max() > 15
