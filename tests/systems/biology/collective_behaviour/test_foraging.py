"""Test the `_foraging` module of `skcomplex.systems.biology.collective_behaviour`."""

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.biology.collective_behaviour import pheromone_foraging


def test_pheromone_trail_marks_shortest_path():
    """Test that the pheromone along the line between nest and food ends higher than elsewhere."""
    simulation = Simulation(pheromone_foraging(), dt=100, random_state=0).prepare().run(until=600)
    pheromone = simulation.trajectory_.lattice('ground/pheromone.level')[-1]
    elsewhere = np.ones(pheromone.shape, dtype=bool)
    elsewhere[5:35, 9:12] = False
    assert pheromone[5:35, 10].mean() > 2 * pheromone[elsewhere].mean()
    assert pheromone[5:35, 10].min() > 0
