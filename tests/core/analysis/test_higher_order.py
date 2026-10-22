"""Test the analysis of equations of any order: written order, exact solutions, no false solutions, and signs."""

import pytest
from sage.all__sagemath_symbolics import SR, cos, diff, function, sin, sqrt

from skcomplex.core.analysis import equations_of, solve_symbolically
from skcomplex.core.base import AnalysisError, state
from skcomplex.core.components import Element
from skcomplex.core.laws import Rate, Update, derivative, lag
from skcomplex.core.system import System

Pendulum = Element.define(
    'Pendulum',
    states={'theta': state(default=0.1, order=2)},
    parameters={'g': 9.81, 'l': 1.0},
    laws=lambda s: [Rate(s.theta, -(s.g / s.l) * sin(s.theta))],
)
SmallAngle = Element.define(
    'SmallAngle',
    states={'theta': state(default=0.1, order=2)},
    parameters={'g': 9.81, 'l': 1.0},
    laws=lambda s: [Rate(s.theta, -(s.g / s.l) * s.theta)],
)
ThirdOrder = Element.define(
    'ThirdOrder',
    states={'x': state(default=0.0, order=3, initial=(1.0, 0.0))},
    laws=lambda s: [Rate(s.x, -derivative(s.x, 1))],
)
FirstOrderPendulum = Element.define(
    'FirstOrderPendulum',
    states={'theta': 0.1, 'omega': 0.0},
    parameters={'g': 9.81, 'l': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega), Rate(s.omega, -(s.g / s.l) * sin(s.theta))],
)
Fibonacci = Element.define(
    'Fibonacci',
    states={'x': state(default=1.0, lags=1, initial=(1.0,))},
    laws=lambda s: [Update(s.x, s.x + lag(s.x, 1))],
)


def test_pendulum_equation_keeps_its_order():
    """Test that a pendulum of order 2 gives one second-order equation and no separate velocity equation."""
    equations = equations_of(System(components=[Pendulum('bob')]))
    assert list(equations) == ['bob.theta']
    theta = function('theta_bob')(SR.var('t'))
    g, l = SR.var('g_bob'), SR.var('l_bob')
    equation = equations['bob.theta']
    assert bool(equation.lhs() == diff(theta, SR.var('t'), 2))
    assert (equation.rhs() + g * sin(theta) / l).simplify_full() == 0


def test_lagged_equation_reads_past_values():
    """Test that a difference equation with a lag reads the value one step back."""
    equation = equations_of(System(components=[Fibonacci('f')]))['f.x']
    t = SR.var('t')
    x = function('x_f')
    assert bool(equation.lhs() == x(t + 1))
    assert (equation.rhs() - x(t) - x(t - 1)).simplify_full() == 0


def test_small_angle_pendulum_solved_with_plain_numbers():
    """Test that `θ'' = -(g / l) θ` with plain positive numbers solves to `0.1 cos(sqrt(g / l) t)`."""
    solution = solve_symbolically(System(components=[SmallAngle('bob')]), use_initial_state=True)['bob.theta']
    values = {SR.var('g_bob'): 9.81, SR.var('l_bob'): 2.0}
    for time in (0.3, 1.0, 2.5):
        expected = 0.1 * float(cos(sqrt(9.81 / 2.0) * time))
        assert float(solution.rhs().subs(values).subs({SR.var('t'): time})) == pytest.approx(expected)


def test_third_order_equation_solved():
    """Test that `x''' = -x'` from `x = 0`, `x' = 1`, `x'' = 0` solves to `sin(t)`."""
    solution = solve_symbolically(System(components=[ThirdOrder('a')]), use_initial_state=True)['a.x']
    t = SR.var('t')
    for time in (0.3, 1.0, 2.5):
        assert float(solution.rhs().subs({t: time})) == pytest.approx(float(sin(time)))


@pytest.mark.parametrize('element', [Pendulum('bob'), FirstOrderPendulum('bob')], ids=['second-order', 'first-order'])
def test_nonlinear_pendulum_has_no_false_solution(element):
    """Test that the nonlinear pendulum raises the no-closed-form error instead of returning unevaluated transforms."""
    with pytest.raises(AnalysisError, match='No closed form was found'):
        solve_symbolically(System(components=[element]), use_initial_state=True)


def test_remaining_sign_question_names_parameter():
    """Test that a sign Maxima still needs raises an error that names the parameter and the domain to declare."""
    Spring = Element.define(
        'Spring',
        states={'x': state(default=0.1, order=2)},
        parameters={'k': 0.0},
        laws=lambda s: [Rate(s.x, -s.k * s.x)],
    )
    with pytest.raises(AnalysisError, match=r'sign of k_a.*domain=Domain\.POSITIVE'):
        solve_symbolically(System(components=[Spring('a')]), use_initial_state=True)
