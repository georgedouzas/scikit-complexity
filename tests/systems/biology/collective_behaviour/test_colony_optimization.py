"""Test the `_colony_optimization` module of `skcomplex.systems.biology.collective_behaviour`."""

import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import AntColonyOptimizationSystem


def _shortest(interaction, is_open=None) -> float:
    """Return the length of the shortest path between the start and the target through the open segments."""
    graph = nx.Graph()
    for index, ((first, second), length) in enumerate(zip(interaction.edges, interaction.lengths, strict=True)):
        if is_open is None or is_open[index]:
            graph.add_edge(first, second, weight=length)
    return nx.shortest_path_length(graph, interaction.start, interaction.target, weight='weight')


def test_colony_finds_a_short_path_and_heals_a_disruption():
    """Test exercises 15.7 and 15.12 of Argun et al.: the ants settle near the shortest path, and again after a cut."""
    for seed in (0, 2):
        system = AntColonyOptimizationSystem(cut=50, seed=seed).prepare().system_
        search = system.interactions[0]
        trajectory = simulate(system, random_state=seed, until=100)
        lengths = trajectory.values('search.shortest')
        assert lengths[49] < 1.1 * _shortest(search)
        is_open = trajectory.agents('roads', 100).values['passable']
        assert (is_open == 0).sum() == 2
        assert np.nanmin(lengths[80:]) < 1.2 * _shortest(search, is_open)
        assert np.nanmin(lengths[80:]) >= _shortest(search, is_open) - 1e-9
