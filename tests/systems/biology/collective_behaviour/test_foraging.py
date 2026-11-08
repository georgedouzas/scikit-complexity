"""Test the `_foraging` module of `skcomplex.systems.biology.collective_behaviour`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import PheromoneForagingSystem


def test_pheromone_trail_marks_shortest_path():
    """Test that the pheromone along the line between nest and food ends higher than elsewhere."""
    trajectory = simulate(PheromoneForagingSystem().prepare().system_, until=600, dt=100, random_state=0)
    pheromone = trajectory.lattice('ground/pheromone.level')[-1]
    elsewhere = np.ones(pheromone.shape, dtype=bool)
    elsewhere[5:35, 9:12] = False
    assert pheromone[5:35, 10].mean() > 2 * pheromone[elsewhere].mean()
    assert pheromone[5:35, 10].min() > 0
