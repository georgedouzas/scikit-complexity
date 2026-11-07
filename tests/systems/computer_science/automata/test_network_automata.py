"""Test the `_network_automata` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ParameterError
from skcomplex.systems import create_simulation
from skcomplex.systems.computer_science.automata import game_of_life, small_world_life


def test_small_world_life_without_rewiring_is_the_game_of_life():
    """Test that on the unrewired Moore network the automaton follows the Game of Life on the torus."""
    trajectory = create_simulation('small_world_life').prepare().run(until=40).trajectory_
    alive = trajectory.values('cells.alive')
    lattice = Simulation(game_of_life((50, 50), initial=alive[0].reshape(50, 50)), dt=1).prepare().run(until=40)
    assert np.array_equal(lattice.trajectory_.lattice('cells.alive')[-1].ravel(), alive[-1])


def test_small_world_life_keeps_eight_neighbours_and_changes_the_density_it_settles_at():
    """Test that the rewiring keeps every degree at 8 and that a random network sustains far more life."""
    assert {degree for _, degree in small_world_life((20, 20), p=0.5).elements[0].space.graph.degree()} == {8}
    densities = [
        create_simulation('small_world_life', p=p)
        .prepare()
        .run(until=150)
        .trajectory_.values('cells.alive')[-50:]
        .mean()
        for p in (0.0, 1.0)
    ]
    assert densities[0] < 0.15
    assert densities[1] > 0.3


def test_small_world_life_rejects_a_probability_outside_the_unit_interval():
    """Test that a rewiring probability above 1 raises a parameter error."""
    with pytest.raises(ParameterError, match='rewiring probability'):
        small_world_life(p=1.5)
