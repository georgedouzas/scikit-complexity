"""Test equation laws, solved jointly for the highest derivatives of the states they relate."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, sin

from skcomplex.core import Domain, ValidationError, parameter, state
from skcomplex.model import Element, Equation, Interaction, Rate, System, derivative
from skcomplex.runtime import Simulation
from skcomplex.solvers import ScipyODE

PRECISE = ScipyODE(rtol=1e-10, atol=1e-12)
POSITIVE = Domain.POSITIVE

Written = Element.define(
    'Written',
    states={'theta': state(default=0.3, order=2)},
    parameters={'m': parameter(default=2.0, domain=POSITIVE), 'g': 9.81, 'l': 1.5},
    laws=lambda s: [Equation(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))],
)
Solved = Element.define(
    'Solved',
    states={'theta': state(default=0.3, order=2)},
    parameters={'g': 9.81, 'l': 1.5},
    laws=lambda s: [Rate(s.theta, -(s.g / s.l) * sin(s.theta))],
)
DoublePendulum = Element.define(
    'DoublePendulum',
    states={'a': state(default=1.0, order=2), 'b': state(default=0.5, order=2)},
    parameters={'g': 9.81},
    laws=lambda s: [
        Equation(
            2 * derivative(s.a, 2) + cos(s.a - s.b) * derivative(s.b, 2)
            == -(derivative(s.b, 1) ** 2) * sin(s.a - s.b) - 2 * s.g * sin(s.a),
        ),
        Equation(
            derivative(s.b, 2) + cos(s.a - s.b) * derivative(s.a, 2)
            == derivative(s.a, 1) ** 2 * sin(s.a - s.b) - s.g * sin(s.b),
        ),
    ],
)
Squared = Element.define(
    'Squared',
    states={'x': state(default=1.0, order=2)},
    laws=lambda s: [Equation(derivative(s.x, 2) ** 2, s.x)],
)
Oscillators = Element.define(
    'Oscillators',
    states={f'x{index}': state(default=1.0, order=2) for index in range(7)},
    laws=lambda s: [
        Equation(2 * derivative(getattr(s, f'x{index}'), 2), -getattr(s, f'x{index}')) for index in range(7)
    ],
)
Push = Interaction.define('Push', roles=['target'], parameters={'f': 1.0}, laws=lambda s: [Rate(s.target.theta, s.f)])


def test_pendulum_as_written_matches_solved_form():
    """Test that `m l θ'' = -m g sin θ` runs exactly like the rate law `θ'' = -(g / l) sin θ`."""
    written = Simulation(System(elements=[Written('bob')]), solver=PRECISE, dt=0.1).run(until=5).trajectory_
    solved = Simulation(System(elements=[Solved('bob')]), solver=PRECISE, dt=0.1).run(until=5).trajectory_
    assert np.array_equal(written['bob.theta'], solved['bob.theta'])


def test_coupled_equations_conserve_energy():
    """Test that the double pendulum's coupled equations are solved jointly and conserve its energy."""
    trajectory = Simulation(System(elements=[DoublePendulum('dp')]), solver=PRECISE, dt=0.05).run(until=10).trajectory_
    a, b = trajectory['dp.a'], trajectory['dp.b']
    a_d1, b_d1 = trajectory['dp.a_d1'], trajectory['dp.b_d1']
    energy = a_d1**2 + 0.5 * b_d1**2 + a_d1 * b_d1 * np.cos(a - b) - 2 * 9.81 * np.cos(a) - 9.81 * np.cos(b)
    assert np.abs(energy - energy[0]).max() < 1e-7 * abs(energy[0])


def test_rates_add_to_equation():
    """Test that a rate law on a state with an equation adds to the solved highest derivative."""
    pushed = System(elements=[Written('bob', theta=0.0)], interactions=[Push('push', target='bob')])
    trajectory = Simulation(pushed, solver=PRECISE, dt=0.01).run(until=0.1).trajectory_
    assert trajectory['bob.theta'][-1] == pytest.approx(0.5 * 0.1**2, rel=1e-2)


def test_nonlinear_highest_derivative_is_rejected():
    """Test that an equation with the square of a highest derivative names the equation and the derivative."""
    with pytest.raises(ValidationError, match=r'equation .* of s is not linear in the highest derivative x_d1'):
        Simulation(System(elements=[Squared('s')])).run(until=1)


def test_many_equations_are_solved_numerically():
    """Test that seven equations, above the symbolic size, are solved per evaluation with the same result."""
    trajectory = Simulation(System(elements=[Oscillators('o')]), solver=PRECISE, dt=0.1).run(until=3).trajectory_
    expected = np.cos(trajectory.times / np.sqrt(2))
    for index in range(7):
        assert np.allclose(trajectory[f'o.x{index}'], expected, atol=1e-7)


def test_equation_takes_two_sides():
    """Test that an equation takes either a relation or two sides."""
    assert Equation(1, 2).rhs == 2
    with pytest.raises(ValueError, match='left and a right side'):
        Equation(1)
