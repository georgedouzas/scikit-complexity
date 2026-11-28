"""Test the `_coevolution` module of `skcomplex.systems.biology.evolution`."""

import itertools

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.systems.biology.evolution import BakSneppenSystem


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()])
def test_bak_sneppen_renews_the_least_fit_species_and_its_neighbours(backend):
    """Test that every step on a ring of 10 renews the least fit species and its two neighbours only."""
    system = BakSneppenSystem(10, backend=backend).prepare()
    states = np.asarray(system.simulate(until=5, dt=1, random_state=0).states)
    for before, after in itertools.pairwise(states):
        least = int(np.argmin(before))
        renewed = {(least - 1) % 10, least, (least + 1) % 10}
        assert set(np.flatnonzero(before != after).tolist()) == renewed


def test_bak_sneppen_renews_the_neighbours_of_a_node():
    """Test that a step on a graph renews the least fit node and its neighbours, and leaves an isolated node alone."""
    graph = nx.path_graph(6)
    graph.add_node(6)
    system = BakSneppenSystem(graph=graph).prepare()
    states = np.asarray(system.simulate(until=3, dt=1, random_state=0).states)
    for before, after in itertools.pairwise(states):
        least = int(np.argmin(before))
        renewed = {least, *graph.neighbors(least)}
        assert set(np.flatnonzero(before != after).tolist()) == renewed


@pytest.mark.benchmark
def test_bak_sneppen_self_organises():
    """Test that Bak-Sneppen on a ring of 200 gathers its fitnesses above a threshold near 0.667."""
    system = BakSneppenSystem(200, backend=NumbaBackend()).prepare()
    trajectory = system.simulate(until=2_000_000, dt=100_000, random_state=0)
    states = np.asarray(trajectory.states)
    late = states[11:].ravel()
    assert np.median(states[-1]) > 0.75
    assert 0.6 < np.quantile(late, 0.1) < 0.72


def test_bak_sneppen_random_neighbors_renew_k_species():
    """Test that every step with random neighbours renews the least fit species and two others, three in all."""
    system = BakSneppenSystem(20, neighbors='random', k=3).prepare()
    states = np.asarray(system.simulate(until=10, dt=1, random_state=0).states)
    for before, after in itertools.pairwise(states):
        changed = np.flatnonzero(before != after)
        assert changed.size == 3
        assert int(np.argmin(before)) in changed


def test_bak_sneppen_random_neighbors_gather_above_one_over_k():
    """Test that the fitnesses of the random neighbour model gather above 1 / 3, uniform over the tenths above it."""
    system = BakSneppenSystem(1000, neighbors='random', k=3).prepare()
    states = np.asarray(system.simulate(until=200_000, dt=2_000, random_state=0).states)
    late = states[50:].ravel()
    counts, _ = np.histogram(late, bins=6, range=(0.4, 1.0))
    assert np.mean(late < 1 / 3) < 0.05
    assert np.all(np.abs(counts / counts.mean() - 1) < 0.1)
