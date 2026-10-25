"""Test the `_populations` module of `skcomplex.models.biology.neuroscience`."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.models.biology.neuroscience import (
    polyhomeostatic_neuron,
    theta_neurons,
    wilson_cowan,
    winners_take_all,
)


def _ott_antonsen(z0, times, *, eta=-0.2, delta=0.1, kappa=1.5, n=2):
    """Return the order parameter of the Ott-Antonsen reduction of Luke, Barreto, and So for theta neurons."""
    normalization = 2**n * math.factorial(n) ** 2 / math.factorial(2 * n)
    grid = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    coefficients = np.fft.fft((1 - np.cos(grid)) ** n).real / grid.size

    def pulses(z):
        return normalization * (coefficients[0] + sum(coefficients[k] * 2 * (z**k).real for k in range(1, n + 1)))

    def rates(t, y):
        z = y[0] + 1j * y[1]
        rate = -1j * (z - 1) ** 2 / 2 + (z + 1) ** 2 / 2 * (-delta + 1j * eta + 1j * kappa * pulses(z))
        return [rate.real, rate.imag]

    solution = solve_ivp(rates, (times[0], times[-1]), [z0.real, z0.imag], t_eval=times, rtol=1e-9, atol=1e-11)
    return solution.y[0] + 1j * solution.y[1]


def test_wilson_cowan_limit_cycle():
    """Test that the Wilson-Cowan populations oscillate with input and decay to rest without."""
    amplitudes = []
    for drive in (0.0, 1.25):
        trajectory = Simulation(wilson_cowan(P=drive), dt=0.2).run(until=100).trajectory_
        amplitudes.append(np.ptp(trajectory['cortex.E'][trajectory.times > 50]))
    assert amplitudes[0] < 1e-6
    assert amplitudes[1] > 0.1


def test_theta_neuron_period():
    """Test that an uncoupled theta neuron with excitability `eta` fires with period `π / sqrt(eta)`."""
    system = theta_neurons(1, eta=1.0, delta=0.0, kappa=0.0)
    trajectory = Simulation(system, dt=0.05).run(until=20).trajectory_
    phase = np.unwrap(trajectory['neurons.theta'][:, 0])
    turns = (phase[-1] - phase[0]) / (2 * math.pi)
    assert abs(20 / turns - math.pi) < 0.05


def test_theta_neurons_follow_the_ott_antonsen_reduction():
    """Test that 2000 theta neurons reading their mean pulse follow the Ott-Antonsen reduction after the transient.

    A fixed RK4 step wraps the phases once per step, whereas an adaptive integrator restarts at every wrap.
    """
    system = theta_neurons(2000)
    trajectory = Simulation(system, dt=1.0, solver=RK4Solver(dt=0.02)).run(until=40).trajectory_
    assert trajectory.problem.states.names == ['neurons.theta']
    order = np.exp(1j * trajectory['neurons.theta']).mean(axis=1)
    reduced = _ott_antonsen(order[0], trajectory.times)
    assert np.abs(order - reduced)[trajectory.times >= 20].max() < 0.05


def test_winners_take_all_with_one_inhibition():
    """Test that the default network has one inhibition reading the total, and only the unit with input 1.1 wins."""
    for inputs in ((1.0, 1.1, 0.9), (1.0, 1.1, 0.9, 1.05, 0.95)):
        assert len(winners_take_all(inputs).interactions) == 1
    state = Simulation(winners_take_all(beta=2.0), dt=1.0).run(until=200).state_
    assert np.argmax(state) == 1
    assert abs(state[1] - 1.1) < 1e-3
    assert np.sort(state)[-2] < 1e-3


def test_winner_takes_all():
    """Test that strong inhibition leaves only the unit with the largest input, and weak inhibition several."""
    inputs = (1.0, 1.1, 0.9)
    strong = Simulation(winners_take_all(inputs, beta=2.0), dt=1.0).run(until=80).state_
    weak = Simulation(winners_take_all(inputs, beta=0.5), dt=1.0).run(until=80).state_
    assert np.argmax(strong) == 1
    assert strong[1] == pytest.approx(1.1, rel=1e-3)
    assert np.sort(strong)[-2] < 1e-3
    assert np.all(weak > 0.1)


def test_polyhomeostatic_neuron_approaches_target_rate():
    """Test that adaptation brings the mean firing rate of the polyhomeostatic neuron to its target."""
    trajectory = Simulation(polyhomeostatic_neuron(mu=0.2), dt=0.5).run(until=600).trajectory_
    x, a, b = trajectory['neuron.x'], trajectory['neuron.a'], trajectory['neuron.b']
    rate = 1 / (1 + np.exp(a * (b - x)))
    assert abs(rate[trajectory.times > 400].mean() - 0.2) < 0.05
    fixed = Simulation(polyhomeostatic_neuron(eps_a=0.0, eps_b=0.0), dt=0.5).run(until=100).trajectory_
    fixed_rate = 1 / (1 + np.exp(-fixed['neuron.x'][fixed.times > 50]))
    assert fixed_rate.mean() == pytest.approx(0.5, abs=0.02)
