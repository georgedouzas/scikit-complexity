"""Test the `_ordering` module of `skcomplex.systems.physics.pattern_formation`."""

import numpy as np
import pytest

from skcomplex.core.analysis import spectrum
from skcomplex.core.components.elements import FieldElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ETDRK4Solver, EulerMaruyamaSolver
from skcomplex.systems.physics.pattern_formation import allen_cahn, cahn_hilliard
from skcomplex.systems.physics.pattern_formation._ordering import LandauEnergyInteraction, PhaseFieldElement


def test_phase_field_holds_its_order_parameter_without_interactions():
    """Test that a phase field without interactions keeps its order parameter."""
    values = {'phi': np.linspace(-1.0, 1.0, 8)}
    field = FieldElement('mixture', element=PhaseFieldElement, space=LatticeSpace((8,)), values=values)
    trajectory = Simulation(System(elements=[field]), dt=0.5).prepare().run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory.values('mixture.phi')[-1], values['phi'])


def _run(conserved):
    """Return the order parameter of a periodic line relaxing in the Landau free energy."""
    values = {'phi': 0.1 * np.sin(2 * np.pi * np.arange(32) / 32) + 0.2}
    field = FieldElement('mixture', element=PhaseFieldElement, space=LatticeSpace((32,), spacing=0.5), values=values)
    energy = LandauEnergyInteraction('energy', target='mixture', kappa=0.5, conserved=conserved)
    system = System(elements=[field], interactions=[energy])
    return Simulation(system, dt=1.0).prepare().run(until=40.0).trajectory_.values('mixture.phi')


def test_non_conserved_field_falls_into_the_nearer_well():
    """Test that a non-conserved field everywhere positive relaxes to the well at 1."""
    np.testing.assert_allclose(_run(conserved=False)[-1], 1.0, atol=1e-3)


def test_conserved_field_keeps_its_mean():
    """Test that a conserved field keeps the mean of the order parameter while it relaxes."""
    phi = _run(conserved=True)
    assert phi[-1].mean() == pytest.approx(phi[0].mean(), rel=1e-9)


def _mean_domain_size(field):
    """Return the domain size `2 pi / <k>` of a field from the mean wavenumber of its structure factor."""
    factor = spectrum(field - field.mean(), kind='structure-factor')
    return 2 * np.pi * factor.power.sum() / (factor.wavenumbers * factor.power).sum()


def test_allen_cahn_does_not_conserve_the_order_parameter():
    """Test that System A from an off-critical mean relaxes to the uniform phase of its sign."""
    system = allen_cahn((32, 32), length=32.0, mean=0.1)
    phi = (
        Simulation(system, dt=20.0, solver=ETDRK4Solver(dt=0.2))
        .prepare()
        .run(until=100)
        .trajectory_.lattice('mixture.phi')
    )
    np.testing.assert_allclose(phi[-1], 1.0, atol=1e-3)


def test_cahn_hilliard_conserves_the_mean_and_coarsens_as_t_one_third():
    """Test that System B keeps the mean of the order parameter and its domains grow as t^(1/3)."""
    system = cahn_hilliard((64, 64), length=64.0, mean=0.1)
    trajectory = Simulation(system, dt=100.0, solver=ETDRK4Solver(dt=0.25)).prepare().run(until=1000).trajectory_
    phi = trajectory.lattice('mixture.phi')
    np.testing.assert_allclose(phi.mean(axis=(1, 2)), phi[0].mean(), atol=1e-10)
    assert np.abs(phi[-1]).max() > 0.9
    sizes = [_mean_domain_size(field) for field in phi[1:]]
    exponent = np.polyfit(np.log(trajectory.times[1:]), np.log(sizes), 1)[0]
    assert exponent == pytest.approx(1 / 3, abs=0.08)


def test_cahn_hilliard_with_temperature_keeps_the_mean_under_conserved_noise():
    """Test that thermal System B keeps the mean of the order parameter while its noise roughens the field."""
    system = cahn_hilliard((16, 16), length=16.0, mean=0.2, temperature=0.05)
    trajectory = (
        Simulation(system, dt=5.0, solver=EulerMaruyamaSolver(dt=0.02), random_state=0)
        .prepare()
        .run(until=20)
        .trajectory_
    )
    phi = trajectory.lattice('mixture.phi')
    np.testing.assert_allclose(phi.mean(axis=(1, 2)), phi[0].mean(), rtol=1e-9)
    deterministic = cahn_hilliard((16, 16), length=16.0, mean=0.2)
    assert deterministic.elements[0].discretization == 'spectral'
    assert np.abs(phi[-1] - phi[0]).max() > 0.1
