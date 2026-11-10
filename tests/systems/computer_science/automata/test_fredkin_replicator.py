"""Test the `_fredkin_replicator` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.computer_science.automata import FredkinReplicatorSystem


def test_fredkin_replicator_copies_the_pattern_four_times_at_powers_of_two():
    """Test that at the times 16 and 32 the state is the sum modulo 2 of the pattern shifted by that many cells."""
    trajectory = simulate(FredkinReplicatorSystem(), random_state=0)
    cells = trajectory.get_lattice('cells.alive')
    for time in (16, 32):
        shifted = [np.roll(cells[0], shift, axis) for shift in (time, -time) for axis in (0, 1)]
        assert np.array_equal(cells[time], sum(shifted) % 2)
        assert cells[time].sum() == 4 * cells[0].sum()
