"""Test the `_criticality` module of `skcomplex.models.physics.criticality`."""

import networkx as nx
import numpy as np

from skcomplex.core.analysis import histogram
from skcomplex.core.runtime import Simulation
from skcomplex.models.physics.criticality import hexagonal_forest_fire, manna_sandpile, sandpile

from ...conftest import Events


def test_sandpile_avalanches_follow_power_law():
    """Test that the 50 x 50 sandpile stays stable after every step and its sizes follow a power law over two decades.

    The exponent is the slope of the log-binned density of 100,000 avalanche sizes after a transient of 20,000 grains,
    between sizes 1 and 100.
    """
    observer = Events(lambda frame: (frame.lattice('pile.height') < 4).all())
    simulation = Simulation(sandpile((50, 50)), dt=1000, record=False, random_state=0, observers=[observer])
    simulation.run(until=120_000)
    sizes = np.asarray([event.size for event in observer.events if event.time > 20_000])
    assert observer.passed
    assert sizes.max() > 1000
    assert {event.label for event in observer.events} == {'avalanche'}
    edges, density = histogram(sizes, 20, log=True)
    centers = np.sqrt(edges[1:] * edges[:-1])
    decades = (centers >= 1) & (centers <= 100) & (density > 0)
    exponent = -np.polyfit(np.log(centers[decades]), np.log(density[decades]), 1)[0]
    assert 1.0 <= exponent <= 1.3


def test_forest_fire_burns_graph_by_distance():
    """Test that a fire on a forested Watts-Strogatz graph reaches every node at the step of its graph distance."""
    graph = nx.connected_watts_strogatz_graph(500, 4, 0.1, seed=0)
    cover = np.ones(500)
    cover[0] = 2
    system = hexagonal_forest_fire(graph=graph, growth=0.0, lightning=0.0, cover=cover)
    covers = Simulation(system, dt=1).run(until=40).trajectory_['forest.cover']
    distances = nx.single_source_shortest_path_length(graph, 0)
    assert (covers == 2).any(axis=0).all()
    assert np.array_equal(np.argmax(covers == 2, axis=0), [distances[node] for node in graph.nodes])


def test_manna_sandpile_settles_below_two_grains():
    """Test that the Manna pile records avalanches, stays stable between grains, and fills to a critical density."""
    trajectory = Simulation(manna_sandpile((16, 16)), dt=100, random_state=0).run(until=3000).trajectory_
    heights = trajectory.lattice('pile.height')
    assert heights.max() < 2
    assert trajectory.events
    assert 0.55 < heights[-1].mean() < 0.75
