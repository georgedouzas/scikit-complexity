"""Test cluster sizes of grids and graphs, percolation strength, and spanning probability."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import cluster_sizes, fit_power_law, label_clusters, percolation, spanning_probability

THRESHOLD = 0.5927


def _burnt_forest(rng: np.random.Generator, density: float, side: int) -> np.ndarray:
    """Return the trees burnt by a fire lit along the first row of a random forest of a density."""
    trees = rng.random((side, side)) < density
    clusters = label_clusters(trees)
    lit = np.unique(clusters.labels[0][clusters.labels[0] > 0])
    return np.isin(clusters.labels, lit)


def test_cluster_sizes_at_the_threshold_on_two_kinds_of_data():
    """Test the cluster size exponents of site percolation at threshold and of critical Erdős-Rényi components."""
    rng = np.random.default_rng(0)
    grids = [rng.random((1024, 1024)) < THRESHOLD for _ in range(6)]
    sizes = np.concatenate([cluster_sizes(grid, periodic=True)[1:] for grid in grids])
    assert fit_power_law(sizes, xmin=100, xmax=100_000, discrete=True, bootstrap=0).alpha == pytest.approx(
        187 / 91,
        abs=0.1,
    )
    components = np.concatenate(
        [cluster_sizes(nx.fast_gnp_random_graph(100_000, 1 / 100_000, seed=seed)) for seed in range(5)],
    )
    assert fit_power_law(components, xmin=5, discrete=True, bootstrap=0).alpha == pytest.approx(2.5, abs=0.1)


def test_percolation_forest_fire_clusters():
    """Test that the burnt cluster grows from below 0.05 to above 0.5 of the forest, fastest near the threshold."""
    rng = np.random.default_rng(1)
    densities = np.linspace(0.4, 0.8, 41)
    strengths = np.asarray(
        [np.mean([percolation(_burnt_forest(rng, density, 256)).strength for _ in range(5)]) for density in densities],
    )
    assert strengths[0] < 0.05
    assert strengths[-1] > 0.5
    steepest = densities[np.argmax(np.gradient(strengths, densities))]
    assert steepest == pytest.approx(THRESHOLD, abs=0.03)


def test_spanning_probability_and_observables_of_graphs():
    """Test that spanning rises across the threshold and graphs give their largest component."""
    rng = np.random.default_rng(2)
    below = spanning_probability([rng.random((64, 64)) < 0.5 for _ in range(50)])
    above = spanning_probability([rng.random((64, 64)) < 0.7 for _ in range(50)])
    assert below < 0.2
    assert above > 0.8
    graph = percolation(nx.disjoint_union(nx.path_graph(3), nx.path_graph(1)))
    assert graph.strength == 0.75
    assert not graph.spanning
