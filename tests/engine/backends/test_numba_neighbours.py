"""Test that compiled pair kernels find neighbours in the bounded cell grids of the neighbour search."""

import numpy as np
import pytest

from skcomplex.core.base import NumericDomain, parameter
from skcomplex.core.elements import GroupElement, ParticleElement
from skcomplex.core.interactions import PairPotentialInteraction
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend
from skcomplex.engine.compile import compile_system

CUTOFF = 2.5


class SoftSphereInteraction(PairPotentialInteraction):
    """Repulsive pair potential `epsilon * (sigma / r) ** 12`."""

    epsilon: float = parameter(1.0, domain=NumericDomain.POSITIVE)
    sigma: float = parameter(1.0, domain=NumericDomain.POSITIVE)

    def pair_energy(self, s, r):
        """Return `epsilon * (sigma / r) ** 12`."""
        return s.epsilon * (s.sigma / r) ** 12


@pytest.fixture(scope='module')
def numba_backend(tmp_path_factory):
    """Return a Numba backend with a test cache directory."""
    return NumbaBackend(cache_dir=tmp_path_factory.mktemp('kernels'))


def _accelerations(system, numba_backend):
    """Return the accelerations of a system at its initial state on the NumPy and the Numba backends."""
    accelerations = []
    for backend in ('numpy', numba_backend):
        problem = compile_system(system, backend=backend)
        drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
        accelerations.append(problem.states.read(drift, 'gas.v'))
    return accelerations


def test_pair_kernel_bounds_grid_of_distant_particles(numba_backend):
    """Test that two particles 1000 cutoffs apart in three dimensions bin into a small grid and feel no force."""
    system = System(
        elements=[
            GroupElement('gas', element=ParticleElement, size=2, values={'x': [[0.0, 0.0, 0.0], [1000 * CUTOFF] * 3]}),
        ],
        interactions=[SoftSphereInteraction('soft', first='gas', second='gas', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=3),
    )
    reference, compiled = _accelerations(system, numba_backend)
    np.testing.assert_array_equal(compiled, reference)
    assert not np.any(compiled)


@pytest.mark.parametrize('period', [2.0, 6.0, 9.0])
def test_pair_kernel_visits_every_cell_of_narrow_periodic_box_once(numba_backend, period):
    """Test that periodic boxes of one, two, and three cells per side count every pair once on both backends."""
    rng = np.random.default_rng(0)
    system = System(
        elements=[
            GroupElement('gas', element=ParticleElement, size=30, values={'x': rng.uniform(0, period, (30, 2))}),
        ],
        interactions=[
            SoftSphereInteraction('soft', first='gas', second='gas', cutoff=CUTOFF, sigma=0.3, period=period),
        ],
        space=ContinuousSpace(extent=(period, period), boundary='periodic'),
    )
    reference, compiled = _accelerations(system, numba_backend)
    assert np.any(reference)
    np.testing.assert_allclose(compiled, reference, rtol=1e-12, atol=1e-12)
