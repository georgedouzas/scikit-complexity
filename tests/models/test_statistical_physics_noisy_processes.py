"""Test the Bertin driven and noisy particles, intermittency, and queues against their known behaviour."""

import math

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.models import describe
from skcomplex.models.mathematics.stochastic_processes import birth_death_queue, on_off_intermittency
from skcomplex.models.physics.soft_matter import (
    biased_trap_walk,
    colored_noise_langevin,
    flashing_ratchet,
    shaken_spring_block,
    tilted_periodic_potential,
)

STANDARD_ERRORS = 3


def _washboard_velocity(f, *, U0, L, kT, gamma, n_points=400):
    """Return the exact mean velocity of an overdamped particle in the tilted potential `U0 cos(2 pi x / L) - f x`."""
    x = np.linspace(0.0, L, n_points, endpoint=False)
    first, second = np.meshgrid(x, x, indexing='ij')

    def potential(position):
        return U0 * np.cos(2 * np.pi * position / L) - f * position

    integral = np.exp((potential(first) - potential(first - second)) / kT).mean() * L**2
    return L * (1 - math.exp(-f * L / kT)) / integral * kT / gamma


def _laminar_durations(values, threshold, dt):
    """Return the durations of the complete phases below a threshold of every column of a series."""
    durations = []
    for column in values.T:
        edges = np.diff(np.r_[0, (column < threshold).astype(int), 0])
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        durations.extend(((ends - starts) * dt)[1:-1])
    return np.array(durations)


def _elongation_statistics(positions):
    """Return the energy per spring and the correlation length of the elongations of blocked chains."""
    elongations = np.diff(positions, axis=1)
    correlations = np.array(
        [np.mean(elongations[:, : elongations.shape[1] - shift] * elongations[:, shift:]) for shift in range(30)],
    )
    correlations /= correlations[0]
    positive = correlations[: np.argmax(correlations < 0)] if np.any(correlations < 0) else correlations
    return np.mean(elongations**2 / 2), positive.sum()


@pytest.mark.parametrize('tau', [0.1, 2.0])
def test_colored_noise_langevin_velocity_variance(tau):
    """Test that the stationary velocity variance is D / (gamma (1 + gamma tau)) and the noise decays in tau."""
    system = colored_noise_langevin(500, gamma=1.0, D=1.0, tau=tau)
    solver = EulerMaruyama(dt=min(0.01, tau / 50))
    trajectory = Simulation(system, dt=0.5, solver=solver, random_state=0).run(until=40).trajectory_
    velocities, noise = trajectory['particles.v'][20:], trajectory['particles.xi'][20:]
    assert np.mean(velocities**2) == pytest.approx(1 / (1 + tau), rel=0.05)
    assert np.mean(noise**2) == pytest.approx(1 / tau, rel=0.05)
    shift = round(tau / 0.5) or 1
    correlation = np.mean(noise[:-shift] * noise[shift:]) / np.mean(noise**2)
    assert correlation == pytest.approx(math.exp(-shift * 0.5 / tau), abs=0.05)


def test_colored_noise_langevin_white_noise_limit():
    """Test that a short correlation time gives the white noise variance D / gamma."""
    system = colored_noise_langevin(1000, gamma=1.0, D=1.0, tau=0.02)
    simulation = Simulation(system, dt=0.5, solver=EulerMaruyama(dt=0.002), random_state=0).run(until=15)
    assert np.mean(simulation.trajectory_['particles.v'][10:] ** 2) == pytest.approx(1.0, rel=0.05)


def test_on_off_intermittency_laminar_phases():
    """Test that the laminar phases at the bifurcation have durations distributed as tau^(-3/2)."""
    system = on_off_intermittency(50, epsilon=0.0)
    simulation = Simulation(system, dt=0.1, solver=EulerMaruyama(dt=0.02), random_state=0).run(until=500)
    values = simulation.trajectory_['variables.x']
    durations = _laminar_durations(values, 0.05, 0.1)
    edges = np.logspace(0, 2, 9)
    counts, _ = np.histogram(durations, edges)
    slope = np.polyfit(np.log(np.sqrt(edges[1:] * edges[:-1])), np.log(counts / np.diff(edges)), 1)[0]
    assert slope == pytest.approx(-1.5, abs=0.15)
    assert np.mean(values < 0.05) > 0.5


def test_biased_trap_walk_subdiffusion():
    """Test that walkers among traps of exponent 1/2 travel as t^(1/2) with the renewal prefactor."""
    alpha = 0.5
    simulation = Simulation(
        built := biased_trap_walk(100, n_sites=200, alpha=alpha),
        dt=10,
        random_state=0,
        observables=describe('biased_trap_walk').observables_of(built),
    ).run(until=2000)
    times, position = simulation.trajectory_.times, simulation.trajectory_['position']
    late = times >= 50
    slope = np.polyfit(np.log(times[late]), np.log(position[late]), 1)[0]
    assert slope == pytest.approx(alpha, abs=0.1)
    prediction = times[-1] ** alpha * math.sin(math.pi * alpha) / (math.pi * alpha)
    assert position[-1] == pytest.approx(prediction, rel=0.2)


@pytest.mark.parametrize('f', [0.5, 2.0])
def test_tilted_periodic_potential_current(f):
    """Test that the mean velocity matches the exact steady current of the tilted periodic potential."""
    system = tilted_periodic_potential(500, f=f, U0=0.5, L=1.0, kT=0.5)
    simulation = Simulation(system, dt=1.0, solver=EulerMaruyama(dt=0.005), random_state=0).run(until=30)
    positions = simulation.trajectory_['particles.x'][..., 0]
    velocities = (positions[-1] - positions[5]) / 25
    error = velocities.std() / math.sqrt(velocities.size)
    exact = _washboard_velocity(f, U0=0.5, L=1.0, kT=0.5, gamma=1.0)
    assert abs(velocities.mean() - exact) < STANDARD_ERRORS * error + 0.02 * exact


def test_flashing_ratchet_current():
    """Test that an asymmetric sawtooth drives a current of the sign of its asymmetry and a symmetric one none."""
    currents = []
    for asymmetry in (0.1, 0.5):
        system = flashing_ratchet(300, asymmetry=asymmetry, rate=1.0, duration=40.0, random_state=1)
        simulation = Simulation(system, dt=1.0, solver=EulerMaruyama(dt=0.004), random_state=0).run(until=40)
        velocities = simulation.trajectory_['particles.x'][-1, :, 0] / 40
        currents.append((velocities.mean(), velocities.std() / math.sqrt(velocities.size)))
    (asymmetric, asymmetric_error), (symmetric, symmetric_error) = currents
    assert np.sign(asymmetric) == np.sign(0.5 - 0.1)
    assert abs(asymmetric) > STANDARD_ERRORS * asymmetric_error
    assert abs(symmetric) < STANDARD_ERRORS * symmetric_error


def test_shaken_spring_block_blocked_configurations():
    """Test that every cycle ends blocked and that stronger shaking gives more energy and longer correlations."""
    statistics = []
    for amplitude, relax_time in ((1.0, 10.0), (3.0, 38.0)):
        period = 2.0 + relax_time
        system = shaken_spring_block(50, amplitude=amplitude, relax_time=relax_time)
        simulation = Simulation(system, dt=period, solver=EulerMaruyama(dt=0.01), random_state=0)
        trajectory = simulation.run(until=12 * period).trajectory_
        positions, velocities = trajectory['chain.x'][1:], trajectory['chain.v'][1:]
        assert np.all(np.abs(velocities) <= 0.02)
        assert np.all(np.abs(np.diff(positions, n=2, axis=1)) < 1.0)
        statistics.append(_elongation_statistics(positions))
    (weak_energy, weak_length), (strong_energy, strong_length) = statistics
    assert strong_energy > 2 * weak_energy
    assert strong_length > 2 * weak_length


def test_shaken_spring_block_independent_anchored_blocks():
    """Test that independent blocks held by springs stop inside the blocked interval |x| < mu_s / anchor."""
    system = shaken_spring_block(100, k=0.0, anchor=2.0, amplitude=2.0, relax_time=20.0)
    simulation = Simulation(system, dt=22.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=110)
    positions = simulation.trajectory_['chain.x'][1:]
    assert np.all(np.abs(positions) < 0.5)
    assert np.abs(positions).max() > 0.3


def test_birth_death_queue_stationary_distribution():
    """Test that an M/M/1 queue of load 1/2 occupies n customers a fraction (1 - rho) rho^n of the time."""
    rho = 0.5
    system = birth_death_queue(200, arrival_rate=rho, service_rate=1.0)
    simulation = Simulation(system, dt=0.5, random_state=0, backend='numba').run(until=2000)
    lengths = simulation.trajectory_['queues.n'][100:]
    occupation = np.array([np.mean(lengths == n) for n in range(6)])
    assert np.all(np.abs(occupation - (1 - rho) * rho ** np.arange(6)) < 0.01)
