"""Test the `_spatial_triadic` module of `skcomplex.systems.mathematics.networks`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.mathematics.networks import SpatialTriadicPercolationSystem

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Pearson correlation of two activity vectors, or NaN when one of them is constant."""
    first, second = first - first.mean(), second - second.mean()
    scale = np.sqrt((first**2).sum() * (second**2).sum())
    return float((first * second).sum() / scale) if scale > 0 else np.nan


def test_spatial_triadic_percolation_lists_signed_regulators_after_the_ends_of_every_link():
    """Test that every hyperedge starts with its link and holds as many regulators as the link counts."""
    system = SpatialTriadicPercolationSystem(1.5, random_state=1).prepare().system_
    nodes, links = system.elements
    assert isinstance(nodes.space, NetworkSpace)
    space = nodes.space.prepare()
    counts = np.asarray(links.values['negatives']) + np.asarray(links.values['positives'])
    assert [hyperedge[:2] for hyperedge in space.hyperedges_] == list(nodes.space.graph.edges)
    assert len({len(hyperedge) for hyperedge in space.hyperedges_}) == 1
    assert len(space.hyperedges_[0]) == 2 + counts.max()
    assert np.mean(counts) == pytest.approx(2 * np.pi * 100 * 0.4 * 0.25**2, rel=0.15)


@BACKENDS
def test_spatial_triadic_percolation_regulates_links_by_their_signed_regulators(backend):
    """Test that a link is active when no negative and some positive regulator is active, on both backends."""
    system = SpatialTriadicPercolationSystem(1.0, p=1.0).prepare().system_
    nodes, links = system.elements
    space = nodes.space.prepare()
    trajectory = simulate(system, until=2, backend=backend, random_state=0)
    before, after = trajectory.values('nodes.active')[1], trajectory.values('links.active')[2]
    negatives, positives = (np.asarray(links.values[name], dtype=int) for name in ('negatives', 'positives'))
    expected = [
        not before[list(hyperedge[2 : 2 + negative])].any()
        and (positive == 0 or before[list(hyperedge[2 + negative : 2 + negative + positive])].any())
        for hyperedge, negative, positive in zip(space.hyperedges_, negatives, positives, strict=True)
    ]
    assert after.astype(bool).tolist() == expected


def test_spatial_triadic_percolation_forms_blinking_patterns():
    """Test that the giant component keeps changing, blinks between steps, and is local in space."""
    system = SpatialTriadicPercolationSystem(4.0).prepare().system_
    trajectory = simulate(system, until=80, random_state=0)
    active = trajectory.values('nodes.active').astype(float)[20:]
    space = system.elements[0].space.prepare()
    offsets = np.abs(space.positions_[:, None, :] - space.positions_[None, :, :])
    distances = np.hypot(*np.moveaxis(np.minimum(offsets, 4.0 - offsets), -1, 0))
    near = (distances < 0.25) & (distances > 0)
    near = near / np.maximum(near.sum(axis=1, keepdims=True), 1)
    assert active.mean(axis=1).std() > 0.1
    assert np.nanmean([_correlation(active[step], active[step + 1]) for step in range(len(active) - 1)]) < -0.2
    assert np.nanmean([_correlation(active[step], active[step + 2]) for step in range(len(active) - 2)]) > 0.2
    assert np.nanmean([_correlation(values, near @ values) for values in active]) > 0.5
