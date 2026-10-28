"""Test the `_ordering` module of `skcomplex.models.physics.pattern_formation`."""

import numpy as np
import pytest

from skcomplex.analysis import SpectrumAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ETDRK4Solver, EulerMaruyamaSolver
from skcomplex.models.physics.pattern_formation import (
    allen_cahn,
    cahn_hilliard,
)


def _mean_domain_size(field):
    """Return the domain size `2 pi / <k>` of a field from the mean wavenumber of its structure factor."""
    factor = SpectrumAnalysis('structure-factor').check().prepare(field - field.mean()).run()
    return 2 * np.pi * factor.power_.sum() / (factor.wavenumbers_ * factor.power_).sum()


def test_allen_cahn_does_not_conserve_the_order_parameter():
    """Test that Model A from an off-critical mean relaxes to the uniform phase of its sign."""
    system = allen_cahn((32, 32), length=32.0, mean=0.1)
    phi = Simulation(system, dt=20.0, solver=ETDRK4Solver(dt=0.2)).run(until=100).trajectory_.lattice('mixture.phi')
    np.testing.assert_allclose(phi[-1], 1.0, atol=1e-3)


def test_cahn_hilliard_conserves_the_mean_and_coarsens_as_t_one_third():
    """Test that Model B keeps the mean of the order parameter and its domains grow as t^(1/3)."""
    system = cahn_hilliard((64, 64), length=64.0, mean=0.1)
    trajectory = Simulation(system, dt=100.0, solver=ETDRK4Solver(dt=0.25)).run(until=1000).trajectory_
    phi = trajectory.lattice('mixture.phi')
    np.testing.assert_allclose(phi.mean(axis=(1, 2)), phi[0].mean(), atol=1e-10)
    assert np.abs(phi[-1]).max() > 0.9
    sizes = [_mean_domain_size(field) for field in phi[1:]]
    exponent = np.polyfit(np.log(trajectory.times[1:]), np.log(sizes), 1)[0]
    assert exponent == pytest.approx(1 / 3, abs=0.08)


def test_cahn_hilliard_with_temperature_keeps_the_mean_under_conserved_noise():
    """Test that thermal Model B keeps the mean of the order parameter while its noise roughens the field."""
    system = cahn_hilliard((16, 16), length=16.0, mean=0.2, temperature=0.05)
    trajectory = (
        Simulation(system, dt=5.0, solver=EulerMaruyamaSolver(dt=0.02), random_state=0).run(until=20).trajectory_
    )
    phi = trajectory.lattice('mixture.phi')
    np.testing.assert_allclose(phi.mean(axis=(1, 2)), phi[0].mean(), rtol=1e-9)
    deterministic = cahn_hilliard((16, 16), length=16.0, mean=0.2)
    assert deterministic.elements[0].discretization == 'spectral'
    assert np.abs(phi[-1] - phi[0]).max() > 0.1
