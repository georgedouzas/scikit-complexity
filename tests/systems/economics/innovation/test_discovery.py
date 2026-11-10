"""Test the `_discovery` module of `skcomplex.systems.economics.innovation`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.innovation import InteractingUrnsNetworkSystem


def _heaps(adjacency, rho, nu, steps):
    """Return the novelties of every node after a number of steps of Eq.

    (4) of Iacopini et al. (2020).
    """
    identity = np.eye(len(adjacency))
    initial = nu + 1
    novelties = np.zeros(len(adjacency))
    for step in range(steps):
        numerator = initial * (adjacency + identity).sum(axis=1) + (nu * identity + (nu + 1) * adjacency) @ novelties
        balls = ((adjacency + identity) * (initial + (nu + 1) * novelties)[None, :]).sum(axis=1)
        novelties = novelties + numerator / (rho * step + balls)
    return novelties


def _run(graph, steps, seed=0, **arguments):
    """Return the trajectory of the urns of a graph."""
    system = InteractingUrnsNetworkSystem(graph, **arguments).prepare().system_
    return simulate(system, until=steps, random_state=seed)


def test_isolated_urns_follow_heaps_law_of_the_urn_with_triggering():
    """Test that the mean novelties of isolated urns agree with Eq.

    (4) without neighbours within 5 percent.
    """
    trajectory = _run(nx.empty_graph(30), 200, rho=4, nu=2)
    novelties = trajectory.get_agents('urns', 200).values['novelties']
    expected = _heaps(np.zeros((30, 30)), 4, 2, 200)
    assert novelties.mean() == pytest.approx(expected.mean(), rel=0.05)


def test_urns_keep_their_balls():
    """Test that the balls of every urn are its initial balls, `rho` per draw, and `nu + 1` per novelty."""
    trajectory = _run(nx.path_graph(5), 40, rho=3, nu=1)
    balls = trajectory.get_agents('balls', 40)
    urns = trajectory.get_agents('urns', 40)
    totals = np.bincount(balls.locations.astype(int), weights=balls.values['count'], minlength=5)
    novelties = urns.values['novelties'][np.argsort(urns.locations.astype(int))]
    np.testing.assert_allclose(totals, 2 + 3 * 40 + 2 * novelties)


def test_the_centre_of_a_star_discovers_faster_than_its_leaves():
    """Test that the social urn of the centre of a star makes it discover more than its leaves, as Eq.

    (4) predicts.
    """
    graph = nx.star_graph(6)
    trajectory = _run(graph, 150, rho=6, nu=3)
    urns = trajectory.get_agents('urns', 150)
    novelties = urns.values['novelties'][np.argsort(urns.locations.astype(int))]
    expected = _heaps(nx.to_numpy_array(graph), 6, 3, 150)
    assert novelties[0] > novelties[1:].max()
    assert expected[0] > expected[1:].max()
    assert novelties[0] == pytest.approx(expected[0], rel=0.25)
