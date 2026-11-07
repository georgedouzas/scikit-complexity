"""Test the `_driven_particles` module of `skcomplex.systems.physics.statistical_mechanics`."""

import itertools

import numpy as np
import pytest

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import (
    ASEPSystem,
    DissipativeZeroRangeSystem,
    ZeroRangeProcessSystem,
)


def tree_occupation_probabilities(*, n_levels, branches, nu, alpha, beta_ext, dissipation):
    """Return the probabilities `z_j` that a node of level `j` is occupied, from the balance of currents of Bertin."""
    rates = nu * (branches ** np.arange(n_levels - 1)) ** alpha
    matrix, source = np.zeros((n_levels, n_levels)), np.zeros(n_levels)
    source[0] = nu * np.exp(-beta_ext)
    matrix[0, 0] = nu
    for level in range(n_levels - 1):
        down = branches * rates[level]
        matrix[level, level] += down
        matrix[level, level + 1] -= down
        matrix[level + 1, level + 1] += rates[level]
        matrix[level + 1, level] -= rates[level]
    matrix[-1, -1] += dissipation
    return np.linalg.solve(matrix, source)


def test_zero_range_process_condensation():
    """Test that above the critical density `1 / (b - 2)` one site holds the excess particles and below it none does."""
    for density in (3.0, 0.25):
        system = ZeroRangeProcessSystem(50, density=density, b=4.0)
        trajectory = simulate(system, until=20000, dt=100.0, random_state=0, backend=NumbaBackend())
        occupations = trajectory.values('sites.n')[50:]
        assert (occupations.sum(axis=1) == round(density * 50)).all()
        largest = occupations.max(axis=1).mean()
        if density > 0.5:
            assert abs(largest - 50 * (density - 0.5)) < 20
        else:
            assert largest < 8


def test_dissipative_zero_range_level_profile():
    """Test that the occupation of every level matches the balance of currents of the product stationary state."""
    settings = {'n_levels': 4, 'branches': 2, 'nu': 1.0, 'alpha': 0.5, 'beta_ext': 1.0, 'dissipation': 0.5}
    system = DissipativeZeroRangeSystem(**settings)
    trajectory = simulate(system, until=6000, dt=1.0, random_state=0, backend=NumbaBackend())
    quanta = trajectory.values('nodes.n')[100:]
    starts = [0, 1, 3, 7, 15]
    occupied = [(quanta[:, start:stop] > 0).mean() for start, stop in itertools.pairwise(starts)]
    expected = tree_occupation_probabilities(**settings)
    np.testing.assert_allclose(occupied, expected, atol=0.02)
    assert np.all(np.diff(expected) < 0)


@pytest.mark.parametrize(
    ('alpha', 'gamma', 'current', 'bulk'),
    [(0.2, 0.7, 0.16, 0.2), (0.7, 0.2, 0.16, 0.8), (0.8, 0.8, 0.25, 0.5)],
)
def test_asep_phases(alpha, gamma, current, bulk):
    """Test the current and the bulk density of the low-density, high-density, and maximal-current phases."""
    system = ASEPSystem(100, alpha=alpha, gamma=gamma)
    trajectory = simulate(system, until=4000, dt=2.0, random_state=0, backend=NumbaBackend())
    occupations = trajectory.values('lane.n')[250:]
    assert abs(gamma * occupations[:, -1].mean() - current) < 0.02
    assert abs(occupations[:, 40:60].mean() - bulk) < 0.05
