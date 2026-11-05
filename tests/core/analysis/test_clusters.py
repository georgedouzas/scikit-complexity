"""Test cluster labels and sizes of grids and graphs, percolation strength, and spanning probability."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import ClusterAnalysis, FitAnalysis
from skcomplex.core.errors import AnalysisError

THRESHOLD = 0.5927


def _clusters(kind: str, source: object, **settings: object) -> ClusterAnalysis:
    """Return the run cluster analysis of a source for one kind."""
    return ClusterAnalysis(kind, **settings).prepare(source).run()


def _burnt_forest(rng: np.random.Generator, density: float, side: int) -> np.ndarray:
    """Return the trees burnt by a fire lit along the first row of a random forest of a density."""
    labels = _clusters('labels', rng.random((side, side)) < density).labels_
    lit = np.unique(labels[0][labels[0] > 0])
    return np.isin(labels, lit)


def test_cluster_sizes_at_the_threshold_on_two_kinds_of_data():
    """Test the cluster size exponents of site percolation at threshold and of critical Erdős-Rényi components."""
    rng = np.random.default_rng(0)
    grids = [rng.random((1024, 1024)) < THRESHOLD for _ in range(6)]
    sizes = np.concatenate([_clusters('sizes', grid, periodic=True).sizes_[1:] for grid in grids])
    fit = FitAnalysis(minimum=100, maximum=100_000, discrete=True, bootstrap=0).prepare(sizes).run()
    assert fit.exponent_ == pytest.approx(187 / 91, abs=0.1)
    components = np.concatenate(
        [_clusters('sizes', nx.fast_gnp_random_graph(100_000, 1 / 100_000, seed=seed)).sizes_ for seed in range(5)],
    )
    fit = FitAnalysis(minimum=5, discrete=True, bootstrap=0).prepare(components).run()
    assert fit.exponent_ == pytest.approx(2.5, abs=0.1)


def test_percolation_forest_fire_clusters():
    """Test that the burnt cluster grows from below 0.05 to above 0.5 of the forest, fastest near the threshold."""
    rng = np.random.default_rng(1)
    densities = np.linspace(0.4, 0.8, 41)
    strengths = np.asarray(
        [
            np.mean([_clusters('percolation', _burnt_forest(rng, density, 256)).strength_ for _ in range(5)])
            for density in densities
        ],
    )
    assert strengths[0] < 0.05
    assert strengths[-1] > 0.5
    steepest = densities[np.argmax(np.gradient(strengths, densities))]
    assert steepest == pytest.approx(THRESHOLD, abs=0.03)


def test_spanning_probability_and_observables_of_graphs():
    """Test that spanning rises across the threshold and graphs give their largest component."""
    rng = np.random.default_rng(2)
    below = _clusters('spanning', [rng.random((64, 64)) < 0.5 for _ in range(50)]).probability_
    above = _clusters('spanning', [rng.random((64, 64)) < 0.7 for _ in range(50)]).probability_
    assert below < 0.2
    assert above > 0.8
    graph = _clusters('percolation', nx.disjoint_union(nx.path_graph(3), nx.path_graph(1)))
    assert graph.strength_ == 0.75
    assert not graph.spanning_


def test_percolation_threshold():
    """Test that the probability of a spanning cluster on a 200 by 200 grid crosses one half within 0.02 of 0.5927."""
    rng = np.random.default_rng(0)
    probabilities = np.linspace(0.56, 0.62, 13)
    spanning = [
        np.mean([_clusters('labels', rng.random((200, 200)) < p).spanning_ for _ in range(60)]) for p in probabilities
    ]
    crossing = np.interp(0.5, spanning, probabilities)
    assert crossing == pytest.approx(THRESHOLD, abs=0.02)


def test_periodic_clusters_merge_across_edges():
    """Test that cells on opposite edges join one cluster only with periodic edges."""
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, 0] = grid[2, 4] = True
    assert _clusters('labels', grid).sizes_.tolist() == [1, 1]
    assert _clusters('labels', grid, periodic=True).sizes_.tolist() == [2]
    assert _clusters('labels', grid, periodic=True, axis=1).spanning_
    assert not _clusters('labels', grid).spanning_


def test_periodic_clusters_merge_diagonally_across_edges():
    """Test that cells touching only diagonally across a periodic edge join one cluster only with diagonal links."""
    grid = np.zeros((4, 4), dtype=bool)
    grid[0, 1] = grid[3, 2] = True
    assert _clusters('labels', grid, periodic=True, diagonal=True).sizes_.tolist() == [2]
    assert _clusters('labels', grid, periodic=True).sizes_.tolist() == [1, 1]
    assert _clusters('labels', grid, diagonal=True).sizes_.tolist() == [1, 1]


def test_periodic_clusters_merge_across_corners_in_three_dimensions():
    """Test that opposite corners of a cube join through the wrapped corner only with diagonal links."""
    grid = np.zeros((3, 3, 3), dtype=bool)
    grid[0, 0, 0] = grid[2, 2, 2] = True
    assert _clusters('sizes', grid, periodic=True, diagonal=True).sizes_.tolist() == [2]
    assert _clusters('sizes', grid, periodic=True).sizes_.tolist() == [1, 1]


def test_periodic_clusters_merge_through_faces_in_three_dimensions():
    """Test that cells facing each other across every periodic face join, and labels keep empty cells at zero."""
    grid = np.zeros((4, 4, 4), dtype=bool)
    grid[0, 1, 1] = grid[3, 1, 1] = grid[1, 0, 2] = grid[1, 3, 2] = grid[2, 2, 0] = grid[2, 2, 3] = True
    result = _clusters('labels', grid, periodic=True)
    assert result.sizes_.tolist() == [2, 2, 2]
    assert np.count_nonzero(result.labels_) == 6


def test_invalid_settings_and_sources_are_rejected():
    """Test an unknown kind, an ignored setting, and a graph for labels."""
    with pytest.raises(AnalysisError, match='labels, sizes'):
        ClusterAnalysis('domains').prepare()
    with pytest.raises(AnalysisError, match='does not read periodic'):
        ClusterAnalysis('percolation', periodic=True).prepare()
    with pytest.raises(AnalysisError, match='not graphs'):
        _clusters('labels', nx.path_graph(3))


def _disks(filling: float, seed: int, side: float = 60.0) -> np.ndarray:
    """Return the centres of disks of radius 1 at a filling factor in a square of a side."""
    count = round(filling * side**2 / np.pi)
    return np.random.default_rng(seed).uniform(0.0, side, (count, 2))


def test_run_spanning_probability_of_disks() -> None:
    """Test that disks of radius 1 span a square of side 60 at filling 1.25 but rarely at filling 1.0."""
    shares = {
        filling: ClusterAnalysis('spanning', radius=1.0)
        .prepare(np.stack([_disks(filling, seed) for seed in range(50)]))
        .run()
        .probability_
        for filling in (1.0, 1.25)
    }
    assert shares[1.0] < 0.1
    assert shares[1.25] > 0.9


def test_run_isolated_points() -> None:
    """Test that points on a grid of spacing 1 with radius 0.1 are clusters of size 1."""
    points = np.stack(np.meshgrid(np.arange(5.0), np.arange(5.0)), axis=-1).reshape(-1, 2)
    sizes = ClusterAnalysis(kind='sizes', radius=0.1).prepare(points).run().sizes_
    assert sizes.tolist() == [1] * 25


def test_run_percolation_of_points() -> None:
    """Test the strength, mean size, and spanning of a chain of points and one point apart."""
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 5.0]])
    result = ClusterAnalysis('percolation', axis=0, radius=0.6).prepare(points).run()
    assert result.sizes_.tolist() == [3, 1]
    assert result.strength_ == 0.75
    assert result.mean_size_ == 1.0
    assert result.spanning_


def test_check_radius_rejects_grid_settings() -> None:
    """Test that a radius takes neither a periodic nor a diagonal setting."""
    with pytest.raises(AnalysisError, match='radius'):
        ClusterAnalysis('sizes', radius=1.0, diagonal=True).prepare()
