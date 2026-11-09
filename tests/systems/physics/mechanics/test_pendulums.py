"""Test the `_pendulums` module of `skcomplex.systems.physics.mechanics`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.physics.mechanics import (
    BallisticPendulumSystem,
    CartPendulumSystem,
    CoupledPendulumsSystem,
    DoublePendulumSystem,
    HuygensClocksSystem,
    MagneticPendulumSystem,
    ParametricPendulumSystem,
)

from ..conftest import _run, _series, _upward_crossings

G = 9.81


def test_double_pendulum_conserves_energy():
    """Test that the double pendulum keeps its energy while it moves chaotically."""
    trajectory = _run(DoublePendulumSystem(), 10.0, 0.05)
    t1, t2, w1, w2 = _series(
        trajectory,
        'pendulum.theta1',
        'pendulum.theta2',
        'pendulum.theta1_d1',
        'pendulum.theta2_d1',
    )
    kinetic = 0.5 * 2 * w1**2 + 0.5 * (w2**2 + 2 * w1 * w2 * np.cos(t1 - t2))
    potential = -2 * G * np.cos(t1) - G * np.cos(t2)
    assert np.ptp(kinetic + potential) < 1e-6


def test_coupled_pendulums_oscillate_in_their_antiphase_mode():
    """Test that identical coupled pendulums started in antiphase oscillate at `sqrt(g / l + 2 k / m)`."""
    trajectory = _run(CoupledPendulumsSystem(0.01, -0.01, k=0.5), 20.0, 0.05)
    left, right = (np.sin(angle) for angle in _series(trajectory, 'left.theta', 'right.theta'))
    crossings = _upward_crossings(trajectory.times, left)
    assert np.allclose(left, -right, atol=1e-9)
    assert np.diff(crossings).mean() == pytest.approx(2 * math.pi / math.sqrt(G + 1.0), rel=1e-4)


def test_cart_pendulum_conserves_horizontal_momentum():
    """Test that the cart and the pendulum keep their horizontal momentum without a force."""
    trajectory = _run(CartPendulumSystem(1.0, m1=2.0, m2=1.0), 10.0, 0.05)
    velocity, theta, omega = _series(trajectory, 'cart.y_d1', 'cart.theta', 'cart.theta_d1')
    momentum = 3.0 * velocity + omega * np.cos(theta)
    assert np.abs(momentum).max() < 1e-8
    assert np.ptp(velocity) > 0.1


def test_parametric_pendulum_resonates_at_twice_its_frequency():
    """Test that small oscillations grow when the pivot oscillates at twice the natural frequency."""
    resonant = _run(ParametricPendulumSystem(0.01, epsilon=0.3), 20.0, 0.05)
    detuned = _run(ParametricPendulumSystem(0.01, epsilon=0.3, Omega=3.3 * math.sqrt(G)), 20.0, 0.05)
    assert np.abs(np.sin(_series(resonant, 'swing.theta')[0])).max() > 0.3
    assert np.abs(np.sin(_series(detuned, 'swing.theta')[0])).max() < 0.03


def test_magnetic_pendulum_comes_to_rest_near_a_magnet():
    """Test that the damped magnetic pendulum stops near one of the magnets."""
    trajectory = _run(MagneticPendulumSystem(), 100.0, 5.0, solver=ScipyODESolver())
    position, velocity = (array[-1] for array in _series(trajectory, 'bob.x', 'bob.x_d1'))
    magnets = np.stack([np.cos(2 * np.pi * np.arange(3) / 3), np.sin(2 * np.pi * np.arange(3) / 3)], axis=1)
    assert np.linalg.norm(velocity) < 1e-3
    assert np.linalg.norm(magnets - position, axis=1).min() < 0.5


def test_ballistic_pendulum_rises_to_the_height_of_momentum_conservation():
    """Test that the pendulum hit by the bullet rises to `(m v / (m + M))² / (2 g)`."""
    trajectory = _run(BallisticPendulumSystem(speed=200.0), 2.0, 0.002)
    theta = _series(trajectory, 'pendulum.theta')[0]
    height = 1 - np.cos(theta).min()
    assert trajectory.events[0].time == pytest.approx(1.0 / 200.0)
    assert height == pytest.approx((0.01 * 200 / 1.01) ** 2 / (2 * G), rel=1e-4)


def test_huygens_clocks_synchronize_in_antiphase():
    """Test that the default clocks run quickly past the escapement switch and lock in antiphase."""
    trajectory = simulate(HuygensClocksSystem(), until=80.0)
    late = trajectory.times >= 60.0
    first, second = trajectory.get_values('clock_0.theta')[late], trajectory.get_values('clock_1.theta')[late]
    assert np.corrcoef(first, second)[0, 1] < -0.95


def test_huygens_pendulums_without_escapement_conserve_energy():
    """Test that pendulums on undamped supports joined by a spring keep their total energy."""
    system = HuygensClocksSystem(2, theta=0.3, k=2.0, M=0.5, rho=0.0, lam=0.0)
    trajectory = _run(system, 10.0, 0.05)
    t1, t2, x1, x2 = _series(trajectory, 'clock_0.theta', 'clock_1.theta', 'clock_0.x', 'clock_1.x')
    w1, w2, v1, v2 = _series(trajectory, 'clock_0.theta_d1', 'clock_1.theta_d1', 'clock_0.x_d1', 'clock_1.x_d1')
    kinetic = sum(0.5 * (w**2 + v**2 + 2 * v * w * np.cos(t)) + 0.25 * v**2 for t, w, v in ((t1, w1, v1), (t2, w2, v2)))
    potential = -9.8 * (np.cos(t1) + np.cos(t2)) + (x2 - x1) ** 2
    assert np.ptp(kinetic + potential) < 1e-6
    assert np.abs(t2).max() > 0.05


def test_parametric_pendulum_driven_horizontally_resonates_linearly_from_rest():
    """Test that a horizontal drive at the natural frequency grows the swing as `ε Ω t / 2`, unlike a vertical one."""
    frequency = math.sqrt(9.81)
    systems = [
        ParametricPendulumSystem(theta=0.0, epsilon=0.002, Omega=frequency, direction=direction).prepare().system_
        for direction in ('horizontal', 'vertical')
    ]
    horizontal, vertical = (_series(_run(system, 20.0, 0.01), 'swing.theta')[0] for system in systems)
    assert np.abs(horizontal[-200:]).max() == pytest.approx(0.002 * frequency * 20.0 / 2, rel=0.05)
    assert np.abs(vertical).max() == 0
