"""Test states and laws of any order: companions, superposition on the highest derivative, and order checks."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import abs_symbolic, sin

from skcomplex.core import Domain, ValidationError, parameter, state
from skcomplex.model import Element, Interaction, Rate, System, derivative
from skcomplex.runtime import Simulation
from skcomplex.solvers import ScipyODE, VelocityVerlet

PRECISE = ScipyODE(rtol=1e-10, atol=1e-12)

Pendulum = Element.define(
    'Pendulum',
    states={'theta': state(default=0.1, domain=Domain.ANGLE, order=2)},
    parameters={
        'g': parameter(default=9.81, domain=Domain.POSITIVE),
        'l': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [Rate(s.theta, -(s.g / s.l) * sin(s.theta))],
)
Mass = Element.define('Mass', states={'x': state(default=1.0, order=2)}, parameters={'m': parameter(default=2.0)})
Spring = Interaction.define(
    'Spring',
    roles=['target'],
    parameters={'k': 8.0},
    laws=lambda s: [Rate(s.target.x, -s.k * s.target.x / s.target.m)],
)
Damper = Interaction.define(
    'Damper',
    roles=['target'],
    parameters={'c': 0.4},
    laws=lambda s: [Rate(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)],
)
Jerk = Element.define(
    'Jerk',
    states={'x': state(default=0.0, order=3, initial=(0.0, 1.0))},
    parameters={'A': 0.6},
    laws=lambda s: [Rate(s.x, -s.A * derivative(s.x, 2) - derivative(s.x, 1) + abs_symbolic(s.x) - 1)],
)
Stock = Element.define('Stock', states={'n': state(default=1.0)}, laws=lambda s: [Rate(s.n, -s.n)])
Mismatched = Interaction.define('Mismatched', roles=['target'], laws=lambda s: [Rate(s.target.x, -s.target.x, order=1)])


def test_initial_angular_velocity():
    """Test that a state of order 2 starts from its angle and its angular velocity, both recorded by path."""
    trajectory = (
        Simulation(System(elements=[Pendulum('bob', theta=0.1, theta_d1=0.5)]), dt=0.1).run(until=1).trajectory_
    )
    assert trajectory['bob.theta'][0] == pytest.approx(0.1)
    assert trajectory['bob.theta_d1'][0] == pytest.approx(0.5)
    assert trajectory['bob.theta'][1] > 0.1


def test_first_order_unchanged():
    """Test that a state without an order compiles without companion states."""
    simulation = Simulation(System(elements=[Stock('a')]), dt=0.5).run(until=1)
    assert simulation.problem_.states.names == ['a.n']


def test_newton_superposition_of_separate_laws():
    """Test that a spring and a damper from separate interactions add up to `m x'' = -k x - c x'`."""
    system = System(
        elements=[Mass('mass')],
        interactions=[Spring('spring', target='mass'), Damper('damper', target='mass')],
    )
    trajectory = Simulation(system, solver=PRECISE, dt=0.1).run(until=10).trajectory_
    gamma = 0.4 / (2 * 2.0)
    omega = np.sqrt(8.0 / 2.0 - gamma**2)
    t = trajectory.times
    expected = np.exp(-gamma * t) * (np.cos(omega * t) + gamma / omega * np.sin(omega * t))
    assert np.allclose(trajectory['mass.x'], expected, atol=1e-7)


def test_third_order_jerk_is_bounded_and_chaotic():
    """Test that Sprott's jerk equation compiles to three equations and gives a bounded, sensitive trajectory."""
    first = Simulation(System(elements=[Jerk('j')]), solver=PRECISE, dt=0.5).run(until=300)
    assert first.problem_.states.names == ['j.x', 'j.x_d1', 'j.x_d2']
    second = Simulation(System(elements=[Jerk('j', x=1e-8)]), solver=PRECISE, dt=0.5).run(until=300)
    x = first.trajectory_['j.x']
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() < 10
    assert np.abs(x - second.trajectory_['j.x']).max() > 0.1


def test_order_mismatch_is_rejected():
    """Test that a law asserting order 1 on a state of order 2 names the law, the state, and both orders."""
    system = System(elements=[Mass('mass')], interactions=[Mismatched('wrong', target='mass')])
    with pytest.raises(
        ValidationError,
        match=r'wrong on mass\.x gives the derivative of order 1, but the state has order 2',
    ):
        Simulation(system).run(until=1)


def test_derivative_at_order_is_rejected_in_rates():
    """Test that a rate cannot read the derivative of a state's own order."""
    Bad = Interaction.define('Bad', roles=['target'], laws=lambda s: [Rate(s.target.x, derivative(s.target.x, 2))])
    with pytest.raises(ValidationError, match='below its declared order'):
        Simulation(System(elements=[Mass('mass')], interactions=[Bad('bad', target='mass')])).run(until=1)


def test_velocity_verlet_runs_order_two_position():
    """Test that velocity Verlet recognises a position of order 2 and conserves the oscillator's energy."""
    system = System(elements=[Mass('mass')], interactions=[Spring('spring', target='mass')])
    trajectory = Simulation(system, solver=VelocityVerlet(dt=0.001), dt=0.1).run(until=10).trajectory_
    energy = 0.5 * 2.0 * trajectory['mass.x_d1'] ** 2 + 0.5 * 8.0 * trajectory['mass.x'] ** 2
    assert np.allclose(energy, energy[0], rtol=1e-4)
