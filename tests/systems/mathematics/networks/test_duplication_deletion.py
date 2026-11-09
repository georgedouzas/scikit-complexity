"""Test the `_duplication_deletion` module of `skcomplex.systems.mathematics.networks`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.mathematics.networks import DuplicationDeletionSystem


def test_duplication_deletion_settles_at_the_connectance_of_jensen():
    """Test that the mean connectance of the steady state is Eq.

    8.84 of Jensen and that the size stays fixed.
    """
    n_nodes, inherit, parent, new = 100, 0.5, 0.5, 0.01
    system = DuplicationDeletionSystem(n_nodes, inherit=inherit, parent=parent, new=new)
    trajectory = simulate(system, until=4000, random_state=0)
    frames = range(2000, 4001, 10)
    connectance = [len(trajectory.get_network(frame).sources) / (n_nodes * (n_nodes - 1) / 2) for frame in frames]
    expected = (new * (n_nodes - 2) + parent) / (n_nodes - 1 + (new - inherit) * (n_nodes - 2))
    assert {len(trajectory.get_network(frame).nodes) for frame in frames} == {n_nodes}
    assert np.mean(connectance) == pytest.approx(expected, rel=0.15)
