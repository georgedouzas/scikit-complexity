"""Test the `_assembly` module of `skcomplex.systems.biology.evolution`."""

import math

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.biology.evolution import assembly_growth, polymer_assembly_pool

from ..conftest import _run_precise


def test_undirected_assembly_follows_the_poisson_expansion():
    """Test that without selection `N_a(t) = (k t)^a / a!`, and that selection slows the deeper levels."""
    free = _run_precise(assembly_growth(6), 5, dt=0.5).values('levels.N')[-1]
    np.testing.assert_allclose(free, [5**a / math.factorial(a) for a in range(1, 7)], rtol=1e-6)
    selected = _run_precise(assembly_growth(6, alpha=0.5), 5, dt=0.5).values('levels.N')[-1]
    assert np.all(selected[1:] < free[1:])


def test_directed_joining_builds_longer_polymers():
    """Test that reusing the newest polymer builds far longer chains than joining random polymers."""
    lengths = {}
    for mode in ('undirected', 'directed'):
        trajectory = Simulation(polymer_assembly_pool(mode), random_state=0).prepare().run(until=100).trajectory_
        pool = trajectory.agents('pool', 100).values['length']
        assert pool.size == 101
        lengths[mode] = np.log2(pool.max())
    assert lengths['directed'] > 2 * lengths['undirected']
