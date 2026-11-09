"""Test the `_forest_size_structure` module of `skcomplex.systems.biology.ecology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.biology.ecology import ForestSizeStructureSystem


def test_forest_size_structure_edges_deplete_small_trees():
    """Test that the strip edges hold fewer trees than the bulk, and deplete small trees more than tall ones."""
    system = ForestSizeStructureSystem(20, 40, top=10.0).prepare().system_
    trajectory = simulate(system, until=60.0, dt=60.0, solver=RK4Solver(dt=0.1))
    density = trajectory.values('forest.x')[-1].reshape(20, 40)
    edge, bulk = density[0], density[10]
    assert edge.sum() < 0.8 * bulk.sum()
    assert edge[0] / bulk[0] < edge[-10] / bulk[-10]
    assert np.all(density >= 0)


def test_forest_size_structure_reaches_a_stationary_density():
    """Test that the density of the strip settles, changing by less than one percent over the last ten time units."""
    system = ForestSizeStructureSystem(10, 30, top=10.0, width=4.0).prepare().system_
    trajectory = simulate(system, until=120.0, dt=10.0, solver=RK4Solver(dt=0.1))
    values = trajectory.values('forest.x')
    assert np.abs(values[-1] - values[-2]).max() < 0.01 * np.abs(values[-1]).max()
