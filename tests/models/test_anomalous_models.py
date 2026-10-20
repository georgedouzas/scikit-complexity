"""Test the anomalous diffusion and trap models built on renewal and fractional noise laws."""

import numpy as np
import pytest

from skcomplex.core.analysis import hurst_exponent
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, EulerMaruyama, FractionalEulerMaruyama
from skcomplex.models.physics.soft_matter import (
    annealed_transient_time_motion,
    continuous_time_random_walk,
    fractional_brownian_motion,
    levy_flight,
    levy_walk,
)
from skcomplex.models.physics.statistical_mechanics import trap_model


def _exponent(trajectory, name, start):
    """Return the scaling exponent of the ensemble mean squared displacement after a time."""
    times = trajectory.times
    squared = np.mean(np.sum(trajectory[name] ** 2, axis=-1), axis=1)
    late = times > start
    return np.polyfit(np.log(times[late]), np.log(squared[late]), 1)[0]


def test_continuous_time_random_walk_subdiffuses():
    """Test that 10000 walkers with waiting-time exponent 0.5 run to time 10000 spread as t^0.5."""
    system = continuous_time_random_walk(10000, alpha=0.5, n_dim=1)
    trajectory = Simulation(system, dt=100, random_state=0, record=True).run(10000).trajectory_
    assert _exponent(trajectory, 'walkers.x', 100) == pytest.approx(0.5, abs=0.1)


def test_levy_walk_superdiffuses():
    """Test that 2000 walkers with flight-time exponent 1.5 run to time 2000 spread as t^(3 - sigma)."""
    system = levy_walk(2000, sigma=1.5, n_dim=1)
    simulation = Simulation(system, solver=RK4(dt=1.0), dt=20, random_state=0)
    assert _exponent(simulation.run(2000).trajectory_, 'walkers.x', 100) == pytest.approx(1.5, abs=0.1)


def test_annealed_transient_time_motion_subdiffuses():
    """Test that 10000 walkers with sigma 0.5 and gamma 1 spread as t^(sigma / gamma) at long times."""
    system = annealed_transient_time_motion(10000, sigma=0.5, gamma=1.0, n_dim=1)
    simulation = Simulation(system, solver=EulerMaruyama(dt=1.0), dt=20, random_state=0)
    assert _exponent(simulation.run(2000).trajectory_, 'walkers.x', 100) == pytest.approx(0.5, abs=0.1)


def test_fractional_brownian_motion_has_its_hurst_index():
    """Test that 1000 walkers with H = 0.3 spread as t^0.6, with increments of Hurst exponent 0.3."""
    system = fractional_brownian_motion(1000, H=0.3, n_dim=1)
    simulation = Simulation(system, solver=FractionalEulerMaruyama(dt=0.01), dt=0.01, random_state=0)
    trajectory = simulation.run(10).trajectory_
    assert _exponent(trajectory, 'walkers.x', 0.05) == pytest.approx(0.6, abs=0.05)
    exponents = [hurst_exponent(np.diff(trajectory['walkers.x'][:, member, 0])) for member in range(100)]
    assert np.mean(exponents) == pytest.approx(0.3, abs=0.05)


def test_levy_flight_spreads_as_n_to_one_over_mu():
    """Test that the median distance of 2000 walkers with mu = 1.2 grows as n^(1 / mu)."""
    trajectory = Simulation(levy_flight(2000, mu=1.2), random_state=0).run(200).trajectory_
    distance = np.median(np.linalg.norm(trajectory['walkers.x'][10:], axis=-1), axis=1)
    slope = np.polyfit(np.log(trajectory.times[10:]), np.log(distance), 1)[0]
    assert slope == pytest.approx(1 / 1.2, abs=0.05)


def test_trap_model_escapes_grow_as_t_to_the_temperature():
    """Test that the escapes per walker below the glass temperature grow as t^T."""
    trajectory = Simulation(trap_model(5000, T=0.5), dt=100, random_state=0).run(100000).trajectory_
    late = trajectory.times > 1000
    escapes = trajectory['escapes.renewals'][late] / 5000
    slope = np.polyfit(np.log(trajectory.times[late]), np.log(escapes), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.05)
