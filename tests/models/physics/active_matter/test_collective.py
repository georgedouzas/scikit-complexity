"""Test the `_collective` module of `skcomplex.models.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.models.physics.active_matter import (
    cucker_smale_flocking,
    living_crystals,
    motility_induced_phase_separation,
    toner_tu,
)


def _mean_neighbors(positions, size, radius):
    """Return the mean number of neighbours closer than a radius in a periodic box."""
    separations = positions[:, None] - positions[None]
    separations -= size * np.rint(separations / size)
    distances = np.linalg.norm(separations, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return (distances < radius).sum(axis=1).mean()


def test_motility_induced_phase_separation_gathers_disks():
    """Test that fast repulsive swimmers gather into clusters without passing through each other."""
    model = motility_induced_phase_separation(200)
    size = model.space.check().prepare().extent_[0]
    trajectory = (
        Simulation(model, dt=0.5, solver=EulerMaruyamaSolver(dt=0.005), random_state=0)
        .check()
        .prepare()
        .run(until=5)
        .trajectory_
    )
    positions = trajectory.values('swimmers.x')
    assert _mean_neighbors(positions[-1], size, 1.1) > 3 * _mean_neighbors(positions[0], size, 1.1)
    distances = np.linalg.norm(positions[-1][:, None] - positions[-1][None], axis=-1)
    assert distances[np.triu_indices(len(distances), 1)].min() > 0.8


def test_living_crystals_form_only_with_phoretic_attraction():
    """Test that active colloids gather into clusters with phoretic attraction and stay apart without it."""
    neighbors = {}
    for v0 in (0.0, 50.0):
        model = living_crystals(50, size=30.0, v0=v0)
        solver = EulerMaruyamaSolver(dt=0.002)
        trajectory = Simulation(model, dt=0.5, solver=solver, random_state=0).check().prepare().run(until=5).trajectory_
        neighbors[v0] = _mean_neighbors(trajectory.values('swimmers.x')[-1], 30.0, 2.5)
    assert neighbors[50.0] > 4 * neighbors[0.0]


def test_cucker_smale_flock_aligns_and_keeps_momentum():
    """Test that Cucker-Smale velocities converge to their conserved mean."""
    trajectory = Simulation(cucker_smale_flocking(), dt=1.0, random_state=0).check().prepare().run(until=20).trajectory_
    velocities = trajectory.values('birds.x_d1')
    assert np.allclose(velocities[-1].mean(axis=0), velocities[0].mean(axis=0), atol=1e-6)
    assert velocities[-1].std(axis=0).max() < 1e-2 * velocities[0].std(axis=0).min()


def test_toner_tu_flock_orders_at_the_mean_field_speed():
    """Test that the polar fluid conserves its mass and moves together at `sqrt(a (rho0 - rho_c) / beta)`."""
    model = toner_tu((16, 16))
    trajectory = (
        Simulation(model, dt=1.0, solver=EulerMaruyamaSolver(dt=0.02), random_state=0)
        .check()
        .prepare()
        .run(until=30)
        .trajectory_
    )
    vx, vy, rho = trajectory.values('flock.vx')[-1], trajectory.values('flock.vy')[-1], trajectory.values('flock.rho')
    assert rho[-1].sum() == pytest.approx(rho[0].sum())
    assert np.hypot(vx.mean(), vy.mean()) == pytest.approx(np.sqrt(0.5), rel=0.02)
