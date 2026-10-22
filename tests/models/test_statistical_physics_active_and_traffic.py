"""Test the active matter and traffic models of Bertin's statistical physics of complex systems."""

import numpy as np

from skcomplex.core.runtime import Simulation, Trajectory
from skcomplex.core.solvers import RK4, EulerMaruyama
from skcomplex.models import describe
from skcomplex.models.engineering.traffic import langevin_traffic, traffic_density_equation
from skcomplex.models.physics.active_matter import aligning_active_brownian_particles, binary_collision_flocking


def _stationary_polarization(trajectory: Trajectory, start: int) -> float:
    """Return the polarization of a trajectory averaged over its frames from `start` on."""
    return float(np.mean(np.abs(np.asarray(trajectory['polarization'])[start:])))


def test_aligning_active_brownian_particles_alignment_transition() -> None:
    """Polarization stays below 0.2 at weak alignment and exceeds 0.6 at strong alignment, at one density and noise."""
    polarizations = []
    for gamma in (0.02, 1.0):
        system = aligning_active_brownian_particles(200, size=20.0, gamma=gamma, D_r=0.1, random_state=0)
        simulation = Simulation(
            system,
            dt=1.0,
            solver=EulerMaruyama(dt=0.05),
            random_state=0,
            observables=describe('aligning_active_brownian_particles').observables_of(system),
        ).run(until=200)
        polarizations.append(_stationary_polarization(simulation.trajectory_, 100))
    low, high = polarizations
    assert low < 0.2
    assert high > 0.6


def test_binary_collision_flocking_orders_below_noise_threshold() -> None:
    """Particles flock at low noise and stay disordered above the largest threshold noise of about 0.90."""
    polarizations = []
    for sigma in (0.2, 1.0):
        system = binary_collision_flocking(400, size=20.0, sigma=sigma, random_state=0)
        simulation = Simulation(
            system,
            dt=1,
            random_state=0,
            observables=describe('binary_collision_flocking').observables_of(system),
        ).run(until=200)
        polarizations.append(_stationary_polarization(simulation.trajectory_, 100))
    ordered, disordered = polarizations
    assert ordered > 0.6
    assert disordered < 0.2


def test_langevin_traffic_follows_desired_speed_with_density_dependent_noise() -> None:
    """Mean speed equals the desired speed of the density and the speed variance equals `D1 rho / gamma`."""
    gamma, D1, density = 3.0, 0.05, 0.5
    system = langevin_traffic(50, 100.0, gamma=gamma, D1=D1)
    simulation = Simulation(system, dt=1.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=200)
    speeds = simulation.trajectory_['cars.v'][50:]
    assert abs(speeds.mean() - (1 - density)) < 0.02
    assert abs(speeds.var() / (D1 * density / gamma) - 1) < 0.2


def test_traffic_density_equation_jams_above_threshold_density() -> None:
    """Density waves decay below the density `D1 gamma` and grow into jams above it."""
    spreads = []
    for density in (0.1, 0.5):
        system = traffic_density_equation(50, length=25.0, density=density, D1=0.2)
        simulation = Simulation(system, dt=10.0, solver=RK4(dt=0.1)).run(until=300)
        rho = simulation.trajectory_['road.rho']
        spreads.append(rho[-1].std() / rho[0].std())
        assert abs(rho[-1].mean() - rho[0].mean()) < 1e-9
    stable, unstable = spreads
    assert stable < 0.5
    assert unstable > 10
