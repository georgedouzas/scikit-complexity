"""Test the `_particles` module of `skcomplex.models.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyama
from skcomplex.models.physics.active_matter import (
    active_brownian_particles,
    active_ornstein_uhlenbeck_particles,
    chiral_active_particles,
    depot_brownian_particles,
    run_and_tumble,
)


def _long_time_slope(times, displacements, start):
    """Return the slope of the mean squared displacement against time after a start index."""
    squared = ((displacements - displacements[0]) ** 2).sum(axis=-1).mean(axis=-1)
    return np.polyfit(times[start:], squared[start:], 1)[0]


def test_active_brownian_particles_diffuse_with_the_active_coefficient():
    """Test that active Brownian particles spread as `(4 D_t + 2 v^2 / D_r) t` at long times within 5%."""
    speed, D_r, D_t = 3.0, 1.0, 0.1
    model = active_brownian_particles(4000, speed=speed, D_r=D_r, D_t=D_t)
    trajectory = Simulation(model, dt=1.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=50).trajectory_
    slope = _long_time_slope(trajectory.times, trajectory['swimmers.x'], 10)
    assert slope == pytest.approx(4 * D_t + 2 * speed**2 / D_r, rel=0.05)


def test_chiral_active_particles_diffuse_slower_when_they_turn():
    """Test that chiral particles diffuse with `D_t + v^2 D_r / (2 (D_r^2 + omega^2))`."""
    model = chiral_active_particles(1000, speed=3.0, omega=2.0, D_t=0.1, D_r=0.1)
    trajectory = Simulation(model, dt=1.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=40).trajectory_
    slope = _long_time_slope(trajectory.times, trajectory['swimmers.x'], 10)
    assert slope / 4 == pytest.approx(0.1 + 9 * 0.1 / (2 * (0.01 + 4)), rel=0.1)


def test_active_ornstein_uhlenbeck_particles_diffuse_with_the_active_coefficient():
    """Test that Ornstein-Uhlenbeck propulsion gives the diffusion coefficient `D` at long times."""
    model = active_ornstein_uhlenbeck_particles(1000, tau=1.0, D=1.0)
    trajectory = Simulation(model, dt=1.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=40).trajectory_
    assert _long_time_slope(trajectory.times, trajectory['swimmers.x'], 10) / 4 == pytest.approx(1.0, rel=0.1)


def test_run_and_tumble_diffuses_with_the_persistence_of_runs():
    """Test that run-and-tumble agents diffuse with `v^2 (1 + q) / (4 (1 - q))` per step."""
    trajectory = Simulation(run_and_tumble(200), random_state=0).run(until=200).trajectory_
    positions = np.array([trajectory.agents('bacteria', frame).locations for frame in range(len(trajectory.times))])
    steps = np.diff(positions.astype(float), axis=0)
    steps -= 10.0 * np.rint(steps / 10.0)
    paths = np.concatenate([np.zeros((1, *steps.shape[1:])), np.cumsum(steps, axis=0)])
    persistence = np.exp(-0.1)
    expected = 0.01 * (1 + persistence) / (4 * (1 - persistence))
    assert _long_time_slope(trajectory.times, paths, 50) / 4 == pytest.approx(expected, rel=0.15)


def test_depot_particles_reach_the_pumped_speed():
    """Test that depot-pumped particles start at rest and move at `sqrt(q / gamma - c / d)`."""
    trajectory = Simulation(depot_brownian_particles(), dt=0.5, random_state=0).run(until=10).trajectory_
    speeds = np.linalg.norm(trajectory['walkers.v'][-1], axis=-1)
    assert speeds.mean() == pytest.approx(np.sqrt(10 / 20 - 1 / 10), rel=0.05)
