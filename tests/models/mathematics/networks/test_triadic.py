"""Test the `_triadic` module of `skcomplex.models.mathematics.networks`."""

import numpy as np
import pytest
from scipy.optimize import brentq

from skcomplex.core.spaces import NetworkSpace
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.models.mathematics.networks import triadic_percolation

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])


def _giant_fraction(mean_degree):
    """Return the giant component of an Erdős-Rényi graph of a mean degree, the root of `S = 1 - exp(-c S)`."""
    if mean_degree <= 1:
        return 0.0
    return brentq(lambda size: size - 1 + np.exp(-mean_degree * size), 1e-9, 1.0)


def _giant(system, backend, until):
    """Return the fraction of nodes in the giant component after every time step of Millán et al., one step each."""
    simulation = Simulation(system, backend=backend, random_state=0).prepare().run(until=until)
    return simulation.trajectory_.values('nodes.active').mean(axis=1)


def test_triadic_percolation_gives_every_link_one_regulator_among_the_other_nodes():
    """Test that every link forms one hyperedge with a third node and takes a negative sign at the given rate."""
    system = triadic_percolation(500, negative=0.3, random_state=1)
    nodes, links = system.elements
    space = nodes.space.prepare()
    assert isinstance(nodes.space, NetworkSpace)
    edges = list(nodes.space.graph.edges)
    assert [hyperedge[:2] for hyperedge in space.hyperedges_] == edges
    assert all(regulator not in ends for *ends, regulator in space.hyperedges_)
    assert np.mean(np.asarray(links.values['sign']) < 0) == pytest.approx(0.3, abs=0.05)


@BACKENDS
def test_triadic_percolation_shows_period_two_above_the_percolation_point(backend):
    """Test that negative regulation above `p = 1 / c` makes the giant component alternate with period two."""
    giant = _giant(triadic_percolation(2000, mean_degree=4.0, p=0.5), backend, until=30)[10:]
    high, low = (giant[::2], giant[1::2]) if giant[0] > giant[1] else (giant[1::2], giant[::2])
    assert np.ptp(high) < 0.1
    assert np.ptp(low) < 0.1
    assert high.mean() == pytest.approx(_giant_fraction(4.0 * 0.5), abs=0.05)
    assert low.mean() < 0.05


def test_triadic_percolation_loses_the_giant_component_below_the_percolation_point():
    """Test that negative regulation below `p = 1 / c` leaves no giant component and no cycle."""
    giant = _giant(triadic_percolation(2000, mean_degree=4.0, p=0.15), NumpyBackend(), until=30)[10:]
    assert giant.max() < 0.05


def test_triadic_percolation_of_positive_regulators_settles():
    """Test that positive regulation alone settles on the fixed point `R = S(c p R)` of its increasing map."""
    giant = _giant(triadic_percolation(2000, mean_degree=4.0, p=0.8, negative=0.0), NumpyBackend(), until=30)[10:]
    fixed = brentq(lambda size: size - _giant_fraction(4.0 * 0.8 * size), 0.5, 1.0)
    assert np.ptp(giant) < 0.1
    assert giant.mean() == pytest.approx(fixed, abs=0.05)
