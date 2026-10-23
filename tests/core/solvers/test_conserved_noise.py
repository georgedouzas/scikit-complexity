"""Test that SDE steppers keep the total of a field under conserved noise and reach its equilibrium fluctuations."""

import numpy as np
import pytest

from skcomplex.core.analysis import structure_factor
from skcomplex.core.components import Element, Field
from skcomplex.core.laws import Noise, Rate, laplacian
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama, Heun
from skcomplex.core.space import LatticeSpace
from skcomplex.core.system import System

Conserved = Element.define(
    'Conserved',
    states={'rho': 0.0},
    parameters={'a': 0.5},
    laws=lambda s: [Noise(s.rho, s.a, conserved=True)],
)
Diffusing = Element.define(
    'Diffusing',
    states={'rho': 0.0},
    parameters={'D': 1.0, 'T': 0.5},
    laws=lambda s: [Rate(s.rho, s.D * laplacian(s.rho)), Noise(s.rho, (2 * s.T) ** 0.5, conserved=True)],
)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
@pytest.mark.parametrize('stepper', [EulerMaruyama, Heun])
@pytest.mark.parametrize('boundary', ['periodic', 'zero-flux'])
def test_conserved_noise_keeps_total(backend, stepper, boundary):
    """Test that a field with only conserved noise keeps its total within 1e-9 relative over 1000 steps."""
    values = np.random.default_rng(0).random((8, 6))
    field = Field('f', Conserved, LatticeSpace((8, 6)), values={'rho': values}, boundary={'rho': boundary})
    trajectory = (
        Simulation(System(components=[field]), backend=backend, solver=stepper(dt=0.01), random_state=0, dt=1.0)
        .run(until=10)
        .trajectory_
    )
    sites = trajectory['f.rho'].reshape(len(trajectory.times), -1)
    np.testing.assert_allclose(sites.sum(axis=1), values.sum(), rtol=1e-9)
    assert np.abs(sites[-1] - values.ravel()).min() > 0


def test_conserved_noise_has_site_variance_of_links():
    """Test that every site of a periodic lattice gains the variance `2 d (a / h)^2` per unit time."""
    field = Field('f', Conserved, LatticeSpace((16, 16), spacing=0.5))
    trajectory = Simulation(System(components=[field]), solver=EulerMaruyama(dt=0.05), random_state=0, dt=4.0)
    sites = trajectory.run(until=4).trajectory_['f.rho'][-1]
    assert sites.var() == pytest.approx(4 * (0.5 / 0.5) ** 2 * 4, rel=0.15)


def test_conserved_noise_gives_flat_structure_factor():
    """Test that diffusion with conserved noise has the flat structure factor `T / D` and a fixed mean."""
    field = Field('f', Diffusing, LatticeSpace((16, 16)))
    trajectory = (
        Simulation(System(components=[field]), solver=EulerMaruyama(dt=0.01), random_state=0, dt=1.0)
        .run(until=1000)
        .trajectory_
    )
    frames = trajectory['f.rho'][50:].reshape(-1, 16, 16)
    factor = structure_factor(frames, frames=True)
    np.testing.assert_allclose(factor.power, 0.5, rtol=0.1)
    np.testing.assert_allclose(np.abs(np.fft.fftn(frames, axes=(1, 2))[:, 0, 0]) ** 2, 0.0, atol=1e-18)
