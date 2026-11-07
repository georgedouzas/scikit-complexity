"""Test nearest-neighbour distances, Clark-Evans ratios, radial distributions, and like-neighbour shares."""

import math

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import NeighborsResult, neighbors
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import AnalysisError
from skcomplex.systems import create_simulation, create_system

EMPTY = -1.0


def _neighbors(kind: str, source: object, **settings: object) -> NeighborsResult:
    """Return the neighbour analysis of a source for one kind, run."""
    return neighbors(source, kind=kind, **settings)


def _schelling_grid(side: int) -> np.ndarray:
    """Return the groups of a relaxed Schelling system with tolerance 0.5 on a grid, `EMPTY` for empty cells."""
    trajectory = (
        create_simulation('schelling', width=side, height=side, threshold=0.5, random_state=0).prepare().run(until=100)
    )
    residents = trajectory.trajectory_.agents('residents', -1)
    group = next(iter(residents.values.values()))
    grid = np.full((side, side), EMPTY)
    places = np.asarray(residents.locations).astype(int)
    grid[places[:, 0], places[:, 1]] = group
    return grid


def _global_like_share(occupied: np.ndarray) -> float:
    """Return the share of like pairs expected when the states of occupied cells are arranged at random."""
    _, counts = np.unique(occupied, return_counts=True)
    return float(np.sum(counts * (counts - 1)) / (occupied.size * (occupied.size - 1)))


def test_clark_evans_ratio_random_and_ordered_positions():
    """Test that random points give 1 and a triangular lattice gives above 2."""
    points = np.random.default_rng(4).random((2000, 2))
    assert _neighbors('clark-evans', points, box=[1.0, 1.0]).ratio[0] == pytest.approx(1, abs=0.05)
    rows, columns = np.meshgrid(np.arange(40), np.arange(46), indexing='ij')
    lattice = np.column_stack([(columns + 0.5 * (rows % 2)).ravel(), (rows * math.sqrt(3) / 2).ravel()])
    assert _neighbors('clark-evans', lattice, box=[46.0, 40 * math.sqrt(3) / 2]).ratio[0] > 2
    assert _neighbors('distances', lattice[:3]).distances[0].tolist() == pytest.approx([1.0, 1.0, 1.0])


@pytest.mark.slow
def test_radial_distribution_of_a_gas_and_a_fluid():
    """Test that an ideal gas gives g = 1 and a dense Lennard-Jones fluid peaks at 2^(1/6) sigma."""
    rng = np.random.default_rng(5)
    gas = rng.random((20, 4000, 2)) * 20
    density = _neighbors('radial-distribution', gas, box=[20.0, 20.0], bins=20).distribution
    assert np.all(np.abs(density - 1) < 0.05)
    trajectory = create_simulation('lennard_jones_fluid').prepare().run().trajectory_
    box = np.asarray(trajectory.problem.system.space.extent_, dtype=float)
    positions = np.mod(np.asarray(trajectory.values('fluid.x'), dtype=float)[-10:], box)
    fluid = _neighbors('radial-distribution', positions, box=box, bins=150, r_max=3.0)
    assert fluid.radii[fluid.distribution.argmax()] == pytest.approx(2 ** (1 / 6), rel=0.05)


def test_neighbor_distances_of_every_frame_of_a_trajectory():
    """Test that a flock in a periodic box gives one distance per agent and frame and one ratio per frame."""
    trajectory = create_simulation('boids').prepare().run(until=3).trajectory_
    distances = _neighbors('distances', trajectory).distances
    assert distances.shape == (len(trajectory.times), trajectory.agents('flock', 0).ids.size)
    assert np.all(distances > 0)
    if trajectory.problem.agent_space.periodic_:
        assert _neighbors('clark-evans', trajectory).ratio.shape == (len(trajectory.times),)


def test_like_neighbor_share_same_index_on_two_domains():
    """Test that a relaxed Schelling system and a quenched Ising lattice exceed 0.75, unlike random arrangements."""
    rng = np.random.default_rng(0)
    schelling = _schelling_grid(40)
    quench = Simulation(create_system('ising', size=64, temperature=1.0), dt=1).prepare().run(until=200)
    ising = quench.trajectory_.lattice('spins.spin')[-1]
    for grid, empty in ((schelling, EMPTY), (ising, None)):
        assert _neighbors('like-share', grid, empty=empty).share[0] > 0.75
        occupied = grid != EMPTY if empty is not None else np.ones(grid.shape, dtype=bool)
        shares = []
        for _ in range(20):
            shuffled = grid.copy()
            shuffled[occupied] = rng.permutation(grid[occupied])
            shares.append(_neighbors('like-share', shuffled, empty=empty).share[0])
        assert np.mean(shares) == pytest.approx(_global_like_share(grid[occupied]), abs=0.02)


def test_like_neighbor_share_on_networks():
    """Test the like share along the edges of a network, and the error of a node without a state."""
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, {0: 0, 1: 0, 2: 1, 3: 1}, 'state')
    assert _neighbors('like-share', graph).share[0] == pytest.approx(2 / 3)
    with pytest.raises(AnalysisError, match='no attribute'):
        _neighbors('like-share', nx.path_graph(3))


def test_neighbor_analysis_checks_its_settings():
    """Test the errors of an unknown kind, an ignored setting, and a Clark-Evans ratio without a box."""
    with pytest.raises(AnalysisError, match='Use one of distances'):
        neighbors(np.zeros(10), kind='voronoi')
    with pytest.raises(AnalysisError, match='does not read bins'):
        neighbors(np.zeros(10), kind='distances', bins=10)
    with pytest.raises(AnalysisError, match='needs a periodic box'):
        _neighbors('clark-evans', np.random.default_rng(0).random((10, 2)))
