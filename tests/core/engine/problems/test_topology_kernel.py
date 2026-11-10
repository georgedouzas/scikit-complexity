"""Test the `topology_kernel` module."""

import numpy as np

from skcomplex.core.engine.problems._topology_kernel import TopologyKernel
from skcomplex.core.specs import TopologyRequestSpec


def test_topology_kernel_get_independent():
    """Test `TopologyKernel._get_independent` on a small case."""
    weights, allowed = np.array([1.0, 0.0, 2.0, 1.0]), np.array([True, True, True, False])
    assert TopologyKernel._get_independent(weights, allowed, np.random.default_rng(0)) == [0, 2]


def test_topology_kernel_changes_edges():
    """Test `TopologyKernel._changes_edges` on a small case."""
    rewire, removal = TopologyRequestSpec('rewire'), TopologyRequestSpec('remove-node')
    assert (TopologyKernel._changes_edges(rewire), TopologyKernel._changes_edges(removal)) == (True, False)
