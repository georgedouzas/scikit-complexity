"""Test the `_explosive` module of `skcomplex.systems.mathematics.networks`."""

import numpy as np
import pytest
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.mathematics.networks import ExplosivePercolationSystem

N_NODES = 2000


def _largest(candidates: int) -> np.ndarray:
    """Return the share of nodes in the largest component after every edge added to isolated nodes."""
    system = ExplosivePercolationSystem(N_NODES, candidates=candidates)
    trajectory = simulate(system, until=int(0.96 * N_NODES), random_state=0)
    return trajectory.values('nodes.size').max(axis=1) / N_NODES


def test_explosive_percolation_jumps_abruptly_under_the_product_rule():
    """Test that the product rule keeps the largest component small up to 0.8 edges per node and large at 0.95."""
    largest = _largest(2)
    assert largest[int(0.8 * N_NODES)] < 0.1
    assert largest[int(0.95 * N_NODES)] > 0.5


def test_explosive_percolation_of_one_candidate_is_an_erdos_renyi_graph():
    """Test that one candidate per step gives the giant component `S = 1 - exp(-1.6 S)` at 0.8 edges per node."""
    giant = brentq(lambda size: size - 1 + np.exp(-1.6 * size), 1e-6, 1.0)
    assert _largest(1)[int(0.8 * N_NODES)] == pytest.approx(giant, abs=0.06)
