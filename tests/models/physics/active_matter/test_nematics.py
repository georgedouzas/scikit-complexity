"""Test the `_nematics` module of `skcomplex.models.physics.active_matter`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, matrix, var

from skcomplex.analysis import AlignmentAnalysis, DefectAnalysis
from skcomplex.core.laws.utils import laplacian
from skcomplex.engine.backends import NumbaBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.models import describe
from skcomplex.models.physics.active_matter import (
    LandauDeGennesEnergyInteraction,
    active_gels,
    dry_active_nematics,
)


def _run_gel(system, until, dt, step):
    """Return the trajectory of a gel, with its velocity as the observables `u_x` and `u_y`."""
    observables = describe('active_gels').observables_of(system)
    simulation = Simulation(system, dt=dt, solver=RK4Solver(dt=step), observables=observables, random_state=0)
    return simulation.check().prepare().run(until=until).trajectory_


def _kinetic_energy(trajectory):
    """Return the mean kinetic energy per site at every frame."""
    return 0.5 * (np.asarray(trajectory.values('u_x')) ** 2 + np.asarray(trajectory.values('u_y')) ** 2).mean(axis=1)


def test_molecular_field_is_the_negative_derivative_of_the_energy():
    """The molecular field of a symmetric traceless tensor is `-a Q - c tr(Q^2) Q + K laplacian(Q)`."""
    q, r, a, c, k = var('q, r, a, c, k')
    tensor = matrix(SR, [[q, r], [r, -q]])
    field = LandauDeGennesEnergyInteraction._molecular_field(tensor, a, c, k)
    square = (tensor * tensor).trace()
    expected = -a * q - c * square * q + k * laplacian(q)
    assert bool((field[0, 0] - expected).expand() == 0)
    assert bool(field[1, 0] == field[0, 1])
    assert bool((field[0, 0] + field[1, 1]).expand() == 0)


@pytest.mark.slow
def test_dry_active_nematics_conserves_the_density():
    """The total density stays constant within 1e-10 relative error over 100 time units."""
    system = dry_active_nematics((32, 32))
    settings = {'solver': RK4Solver(dt=0.02), 'backend': NumbaBackend()}
    trajectory = Simulation(system, dt=20.0, **settings).check().prepare().run(until=100.0).trajectory_
    totals = np.asarray(trajectory.values('nematic.rho')).sum(axis=1)
    assert np.max(np.abs(totals - totals[0])) < 1e-10 * totals[0]
    assert np.abs(np.asarray(trajectory.values('nematic.Q'))).max() > 0.1


@pytest.mark.slow
def test_active_gels_passive_nematic_relaxes_to_the_scalar_order():
    """A uniform passive nematic has no flow, and its scalar order relaxes to `sqrt(-2 a / c)`."""
    a, c = -0.5, 2.0
    system = active_gels((8, 8), a=a, c=c, zeta=0.0, perturbation=0.0)
    gel = system.elements[0]
    gel.values['Q'] = np.tile([0.2, 0.0], (64, 1))
    trajectory = _run_gel(system, until=30.0, dt=10.0, step=0.05)
    assert np.abs(np.asarray(trajectory.values('u_x'))).max() < 1e-8
    assert np.abs(np.asarray(trajectory.values('u_y'))).max() < 1e-8
    frames = np.asarray(trajectory.lattice('gel.Q'))
    order = AlignmentAnalysis('principal-axis').check().prepare(frames).run().order_
    assert np.abs(order[-1] - np.sqrt(-2 * a / c)).max() < 1e-6


def test_active_gels_polar_defaults_differ_from_the_nematic():
    """The turbulent defaults belong to the nematic, and the polar gel keeps a small lattice and a long run."""
    polar = active_gels(order='polar').elements[0]
    assert polar.space.shape == (32, 32)
    assert polar.space.spacing == 1.0
    assert np.abs(polar.values['p'][:, 1]).max() < 0.1
    assert active_gels().elements[0].space.shape == (128, 128)
    assert describe('active_gels').settings_of(order='polar')['dt'] == 1.0
    assert describe('active_gels').settings_of(order='polar')['solver'].dt == 0.05
    assert describe('active_gels').settings_of()['solver'].dt == 5e-4


@pytest.mark.slow
def test_active_gels_extensile_nematic_turbulence_conserves_the_charge():
    """Well above the bend threshold defects appear in pairs, the charge stays 0, and the kinetic energy settles."""
    system = active_gels()
    values = system.elements[0].values
    assert values['zeta'] > 10 * 2 * values['K'] * (2 * np.pi / 10.0) ** 2
    trajectory = _run_gel(system, until=0.5, dt=0.025, step=5e-4)
    axes = AlignmentAnalysis('principal-axis').check().prepare(np.asarray(trajectory.lattice('gel.Q'))).run().axes_
    angles = np.arctan2(axes[..., 1], axes[..., 0])
    defects = DefectAnalysis(period=np.pi).check().prepare(angles).run()
    charges = [item.sum() for item in defects.charges_]
    assert np.max(np.abs(charges)) < 1e-6
    late = len(charges) // 2
    assert max(len(item) for item in defects.charges_[late:]) >= 2
    assert any(0.5 in item.tolist() and -0.5 in item.tolist() for item in defects.charges_[late:])
    energy = _kinetic_energy(trajectory)
    first, second = energy[late : late + (len(energy) - late) // 2], energy[late + (len(energy) - late) // 2 :]
    assert second.mean() > 1e-4
    assert abs(first.mean() - second.mean()) < 0.5 * second.mean()


def _polar_film_threshold(width, *, K, gamma, eta, length, xi):
    """Return the threshold activity of the lowest mode of a polar film from the linear stability analysis."""
    wave = np.pi / width
    half = (1 - xi) / 2
    return -K * wave**2 * (eta / (gamma * half * length**2) + half)


def _polar_film(zeta):
    """Return the kinetic energy of a polar film at an activity at every frame, with the threshold kept apart."""
    sites, width = 16, 16.0
    system = active_gels(
        (4, sites),
        order='polar',
        spacing=2 * width / sites,
        film=True,
        zeta=zeta,
        perturbation=0.0,
    )
    gel = system.elements[0]
    rows = np.tile(np.arange(sites) * 2 * width / sites, 4)
    gel.values['p'] = np.column_stack([np.ones(4 * sites), 0.01 * np.sin(np.pi * rows / width)])
    return _kinetic_energy(_run_gel(system, until=1000.0, dt=50.0, step=0.5))


@pytest.mark.slow
def test_active_gels_polar_film_flows_above_the_threshold():
    """The flow decays 20% below the threshold and grows to a steady nonzero flow 20% above it."""
    threshold = _polar_film_threshold(16.0, K=1.0, gamma=1.0, eta=1.0, length=1.0, xi=0.0)
    below = _polar_film(0.8 * threshold)
    above = _polar_film(1.2 * threshold)
    assert below[-1] < 1e-3 * below[1]
    assert above[-1] > 10 * above[1]
    assert abs(above[-1] - above[-2]) < 1e-3 * above[-1]
