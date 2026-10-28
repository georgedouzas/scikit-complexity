"""Test the `_spatial` module of `skcomplex.analysis`."""

import numpy as np
import pytest

from skcomplex.analysis import label_clusters

PERCOLATION_THRESHOLD = 0.5927


def test_percolation_threshold():
    """Test that the probability of a spanning cluster on a 200 by 200 grid crosses one half within 0.02 of 0.5927."""
    rng = np.random.default_rng(0)
    probabilities = np.linspace(0.56, 0.62, 13)
    spanning = [
        np.mean([label_clusters(rng.random((200, 200)) < p).spans(0) for _ in range(60)]) for p in probabilities
    ]
    crossing = np.interp(0.5, spanning, probabilities)
    assert crossing == pytest.approx(PERCOLATION_THRESHOLD, abs=0.02)


def test_periodic_clusters_merge_across_edges():
    """Test that cells on opposite edges join one cluster only with periodic edges."""
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, 0] = grid[2, 4] = True
    assert label_clusters(grid).sizes.tolist() == [1, 1]
    assert label_clusters(grid, periodic=True).sizes.tolist() == [2]
