"""Test the mixing measures of categorical states on grids and networks."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import (
    dissimilarity_index,
    fields_of,
    interface_density,
    like_neighbor_share,
    snapshots_of,
)
from skcomplex.core.base import AnalysisError
from skcomplex.core.runtime import Simulation
from skcomplex.models import create, run

EMPTY = -1.0


def _schelling_grid(side: int) -> np.ndarray:
    """Return the groups of a relaxed Schelling model with tolerance 0.5 on a grid, `EMPTY` for empty cells."""
    trajectory = run('schelling', width=side, height=side, threshold=0.5, random_state=0, settings={'until': 100})
    snapshot = snapshots_of(trajectory.trajectory_)
    group = next(values for values in snapshot.states.values())[-1]
    grid = np.full((side, side), EMPTY)
    places = snapshot.positions[-1].astype(int)
    grid[places[:, 0], places[:, 1]] = group
    return grid


def _global_like_share(occupied: np.ndarray) -> float:
    """Return the share of like pairs expected when the states of occupied cells are arranged at random."""
    _, counts = np.unique(occupied, return_counts=True)
    return float(np.sum(counts * (counts - 1)) / (occupied.size * (occupied.size - 1)))


def test_like_neighbor_share_same_index_on_two_domains():
    """Test that a relaxed Schelling model and a quenched Ising lattice exceed 0.75, unlike random arrangements."""
    rng = np.random.default_rng(0)
    schelling = _schelling_grid(40)
    ising = fields_of(Simulation(create('ising', size=64, temperature=1.0), dt=1).run(until=200).trajectory_)[-1]
    for grid, empty in ((schelling, EMPTY), (ising, None)):
        assert like_neighbor_share(grid, empty=empty) > 0.75
        occupied = grid != EMPTY if empty is not None else np.ones(grid.shape, dtype=bool)
        shares = []
        for _ in range(20):
            shuffled = grid.copy()
            shuffled[occupied] = rng.permutation(grid[occupied])
            shares.append(like_neighbor_share(shuffled, empty=empty))
        assert np.mean(shares) == pytest.approx(_global_like_share(grid[occupied]), abs=0.02)


def test_interface_density_and_shares_on_networks():
    """Test interface densities of a uniform and a random lattice, and like shares along network edges."""
    assert interface_density(np.ones((8, 8))) == 0
    random = np.random.default_rng(1).choice([-1, 1], (128, 128))
    assert interface_density(random) == pytest.approx(0.5, abs=0.01)
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, {0: 0, 1: 0, 2: 1, 3: 1}, 'state')
    assert like_neighbor_share(graph) == pytest.approx(2 / 3)
    assert interface_density(graph) == pytest.approx(1 / 3)
    with pytest.raises(AnalysisError, match='no attribute'):
        like_neighbor_share(nx.path_graph(3))


def test_dissimilarity_index_of_separated_and_mixed_groups():
    """Test that separated groups give 1, even mixing 0, and more than two groups an error."""
    assert dissimilarity_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    assert dissimilarity_index([0, 1, 0, 1], [0, 0, 1, 1]) == 0
    with pytest.raises(AnalysisError, match='two groups'):
        dissimilarity_index([0, 1, 2], [0, 0, 1])
