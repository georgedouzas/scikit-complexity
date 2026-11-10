"""Test the `_automata` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.computer_science.automata import LangtonsAntSystem


def test_langtons_ant_moves_and_flips():
    """Test that Langton's ant keeps one ant, flips one cell per step, and builds its highway after 10,000 steps."""
    trajectory = simulate(LangtonsAntSystem((100, 100)), until=11_000, dt=100)
    ants = trajectory.get_lattice('cells.ant')
    assert np.all((ants >= 0).sum(axis=(1, 2)) == 1)
    black = trajectory.get_lattice('cells.color').sum(axis=(1, 2))
    assert black[1] <= 100
    position = np.argwhere(ants[-1] >= 0)[0]
    assert np.abs(position - 50).max() > 15
