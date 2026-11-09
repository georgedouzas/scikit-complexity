"""Test the `_collective` module of `skcomplex.systems.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, LammpsSolver, VelocityVerletSolver
from skcomplex.systems.physics.active_matter import (
    CuckerSmaleFlockingSystem,
    LivingCrystalsSystem,
    MotilityInducedPhaseSeparationSystem,
    TonerTuSystem,
)
from skcomplex.systems.physics.active_matter._collective import SoftSphereInteraction
from skcomplex.systems.physics.common.particles import LAMMPS_FIXES, LAMMPS_PAIRS


def _pair(interaction, distance):
    """Return a system of two particles at a distance with one pair interaction."""
    return CompositeSystem(
        elements=[GroupElement('pair', element=PointElement, size=2, values={'x': [[0.0, 0.0], [distance, 0.0]]})],
        interactions=[interaction],
        space=ContinuousSpace(n_dim=2),
    )


@pytest.mark.parametrize(
    'solver',
    [VelocityVerletSolver(dt=0.001), LammpsSolver(dt=0.001, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES)],
    ids=['native', 'lammps'],
)
def test_soft_spheres_repel_inside_contact(solver):
    """Test that overlapping soft spheres move apart and stop interacting beyond contact."""
    system = _pair(SoftSphereInteraction('soft', first='pair', second='pair', cutoff=1.0, k=50.0), 0.8)
    trajectory = simulate(system, until=1, solver=solver)
    separation = np.linalg.norm(trajectory.values('pair.x')[:, 1] - trajectory.values('pair.x')[:, 0], axis=1)
    assert separation[-1] > 1.0
    assert np.allclose(np.diff(separation[-3:]), np.diff(separation[-3:])[0])


def _mean_neighbors(positions, size, radius):
    """Return the mean number of neighbours closer than a radius in a periodic box."""
    separations = positions[:, None] - positions[None]
    separations -= size * np.rint(separations / size)
    distances = np.linalg.norm(separations, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return (distances < radius).sum(axis=1).mean()


def test_motility_induced_phase_separation_gathers_disks():
    """Test that fast repulsive swimmers gather into clusters without passing through each other."""
    system = MotilityInducedPhaseSeparationSystem(200).prepare().system_
    size = system.space.prepare().extent_[0]
    trajectory = simulate(system, until=5, dt=0.5, solver=EulerMaruyamaSolver(dt=0.005), random_state=0)
    positions = trajectory.values('swimmers.x')
    assert _mean_neighbors(positions[-1], size, 1.1) > 3 * _mean_neighbors(positions[0], size, 1.1)
    distances = np.linalg.norm(positions[-1][:, None] - positions[-1][None], axis=-1)
    assert distances[np.triu_indices(len(distances), 1)].min() > 0.8


def test_living_crystals_form_only_with_phoretic_attraction():
    """Test that active colloids gather into clusters with phoretic attraction and stay apart without it."""
    neighbors = {}
    for v0 in (0.0, 50.0):
        system = LivingCrystalsSystem(50, size=30.0, v0=v0)
        solver = EulerMaruyamaSolver(dt=0.002)
        trajectory = simulate(system, until=5, dt=0.5, solver=solver, random_state=0)
        neighbors[v0] = _mean_neighbors(trajectory.values('swimmers.x')[-1], 30.0, 2.5)
    assert neighbors[50.0] > 4 * neighbors[0.0]


def test_cucker_smale_flock_aligns_and_keeps_momentum():
    """Test that Cucker-Smale velocities converge to their conserved mean."""
    trajectory = simulate(CuckerSmaleFlockingSystem(), dt=1.0, random_state=0)
    velocities = trajectory.values('birds.x_d1')
    assert np.allclose(velocities[-1].mean(axis=0), velocities[0].mean(axis=0), atol=1e-6)
    assert velocities[-1].std(axis=0).max() < 1e-2 * velocities[0].std(axis=0).min()


def test_toner_tu_flock_orders_at_the_mean_field_speed():
    """Test that the polar fluid conserves its mass and moves together at `sqrt(a (rho0 - rho_c) / beta)`."""
    system = TonerTuSystem((16, 16))
    trajectory = simulate(system, until=30, dt=1.0, solver=EulerMaruyamaSolver(dt=0.02), random_state=0)
    vx, vy, rho = trajectory.values('flock.vx')[-1], trajectory.values('flock.vy')[-1], trajectory.values('flock.rho')
    assert rho[-1].sum() == pytest.approx(rho[0].sum())
    assert np.hypot(vx.mean(), vy.mean()) == pytest.approx(np.sqrt(0.5), rel=0.02)
