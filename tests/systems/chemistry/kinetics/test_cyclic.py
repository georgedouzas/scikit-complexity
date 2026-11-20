"""Test the `_cyclic` module of `skcomplex.systems.chemistry.kinetics`."""

import numpy as np

from skcomplex.systems.chemistry.kinetics import CyclicCellularAutomatonSystem

from .conftest import _run


def test_cyclic_cellular_automaton_advances_around_the_cycle():
    """Test that every cell of the cyclic automaton stays or moves to the following level."""
    trajectory = _run(CyclicCellularAutomatonSystem((20, 20), n=8), until=30, dt=1)
    levels = trajectory.get_lattice('cells.level')
    steps = np.mod(levels[1:] - levels[:-1], 8)
    assert set(np.unique(steps)) <= {0.0, 1.0}
    assert steps[-1].sum() > 0
