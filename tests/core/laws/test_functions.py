"""Test the symbolic functions for conditions and remainders."""

import warnings

import numpy as np
import pytest
from sage.all__sagemath_symbolics import abs_symbolic, cos, sin

from skcomplex.core.base import BaseInteraction, ValidationError, state, symbol
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import EquationLaw, RateLaw, UpdateLaw, both, derivative, either, lag, mod, series, where
from skcomplex.core.problems import NUMPY_DIALECT, SCALAR_DIALECT, numpy_function, print_expressions
from skcomplex.core.space import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver, ScipyODESolver
from skcomplex.formalisms.particles import ParticleComponent

from .conftest import PRECISE, MassComponent


def test_functions_stay_unevaluated_on_symbols():
    """Test that conditions and remainders of symbols stay symbolic."""
    x = symbol('x')
    assert str(where(x < 0, -x, x)) == 'where(x < 0, -x, x)'
    assert str(mod(x, 2)) == 'mod(x, 2)'
    assert str(either(x < 0, x > 1)) == 'either(x < 0, x > 1)'


@pytest.mark.parametrize(
    ('expression', 'value', 'expected'),
    [
        (lambda x: where(x < 0, -x, x), -2, 2),
        (lambda x: where(either(x < 0, x > 1), 1, 0), 2, 1),
        (lambda x: where(both(x > 0, x < 1), 1, 0), 2, 0),
        (lambda x: mod(x, 1), -0.25, 0.75),
    ],
    ids=['where', 'either', 'both', 'mod'],
)
def test_functions_evaluate_on_numbers(expression, value, expected):
    """Test that conditions and remainders evaluate once their arguments are numbers."""
    x = symbol('x')
    assert float(expression(x).subs({x: value})) == pytest.approx(expected)


def test_functions_differentiate_by_branch():
    """Test that a condition differentiates each branch and keeps the condition."""
    x = symbol('x')
    assert str(where(x < 0, -(x**2), x**3).diff(x)) == 'where(x < 0, -2*x, 3*x^2)'


def test_functions_lower_to_numpy_and_numba():
    """Test that NumPy code evaluates conditions and remainders like the Python operators."""
    x = symbol('x')
    expression = where(either(x < 0, x > 2), -x, mod(x, 2))
    values = np.array([-1.5, 0.5, 3.5])
    result = numpy_function(symbol('t'), [x], [expression])(0.0, values)[0]
    assert np.allclose(result, [1.5, 0.5, -3.5])
    scalar = print_expressions([expression], SCALAR_DIALECT)[1][0]
    assert eval(scalar, {'x': 0.5}) == 0.5  # noqa: S307  # nosec B307: evaluates printed code of a fixed expression
    assert 'np.where' in print_expressions([expression], NUMPY_DIALECT)[1][0]


def test_reflecting_box_reverses_crossing_particle():
    """Test that a particle crossing a wall comes back into the box with its velocity reversed."""
    system = System(
        components=[ParticleComponent('p', x=[0.9], v=[1.0])],
        space=ContinuousSpace(1.0, boundary='reflecting'),
    )
    trajectory = Simulation(system, dt=0.2).run(until=0.2).trajectory_
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.9)
    assert trajectory.values('p.v')[-1, 0] == pytest.approx(-1.0)


def test_periodic_box_wraps_leaving_particle():
    """Test that a particle leaving through one side enters through the opposite side."""
    system = System(
        components=[ParticleComponent('p', x=[0.9], v=[1.0])],
        space=ContinuousSpace(1.0, boundary='periodic'),
    )
    trajectory = Simulation(system, dt=0.2).run(until=0.2).trajectory_
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.1)
    assert trajectory.values('p.v')[-1, 0] == pytest.approx(1.0)


def test_where_with_equal_branches_simplifies():
    """Test that the second derivative of a condition times a symbol is exactly zero."""
    x, y = symbol('x'), symbol('y')
    assert where(x < 0, 3, 3) == 3
    assert (where(x < 0, 1, 2) * y).diff(y, 2).is_trivial_zero()


def test_where_branch_not_chosen_raises_no_warning():
    """Test that NumPy evaluates a condition with a division in the discarded branch without warnings."""
    t, x = symbol('t'), symbol('x')
    function = numpy_function(t, [x], [where(x > 0, 1 / x, 0)])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert function(0.0, np.array([0.0, 2.0]))[0].tolist() == [0.0, 0.5]


@pytest.mark.parametrize('solver', [RK4Solver(), ScipyODESolver()], ids=['rk4', 'scipy'])
def test_long_frame_keeps_particle_in_reflecting_box(solver):
    """Test that a frame of duration 5 reflects a particle four times and ends at 0.5 with velocity -1."""
    system = System(
        components=[ParticleComponent('p', x=[0.5], v=[1.0])],
        space=ContinuousSpace(1.0, boundary='reflecting'),
    )
    trajectory = Simulation(system, solver=solver, dt=5.0).run(until=5).trajectory_
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.5, abs=1e-6)
    assert trajectory.values('p.v')[-1, 0] == pytest.approx(-1.0, abs=1e-6)


def _driven_by(data):
    """Return a system whose one rate reads a series input."""
    driven = ElementComponent.define('DrivenComponent', states={'y': 0.0}, laws=lambda s: [RateLaw(s.y, data(s.t))])
    return System(components=[driven('d')])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_series_compiles_long_held_data(backend):
    """Test that a held series of 1000 points compiles and holds each value until the next time."""
    times, values = np.arange(1000.0), np.sin(np.arange(1000.0))
    problem = compile_system(_driven_by(lambda t: series(t, times, values, 'previous')), backend=backend)
    rates = [problem.drift(k + 0.5, problem.initial_state, problem.initial_parameters)[0] for k in range(1000)]
    np.testing.assert_allclose(rates, values, rtol=0, atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_series_interpolates_long_linear_data(backend):
    """Test that a linear series of 1000 points matches linear interpolation and breaks at every data time."""
    times, values = np.arange(1000.0), np.sin(np.arange(1000.0))
    problem = compile_system(_driven_by(lambda t: series(t, times, values)), backend=backend)
    probes = np.linspace(-3.0, 1003.0, 2501)
    rates = [problem.drift(time, problem.initial_state, problem.initial_parameters)[0] for time in probes]
    np.testing.assert_allclose(rates, np.interp(probes, times, values), rtol=0, atol=1e-12)
    assert problem.breakpoints.between(-1.0, 1000.0) == list(times)


def test_condition_on_a_nested_modulo_of_time_compiles():
    """Test that a condition whose side holds a modulo of time inside another function compiles and switches."""
    nested = ElementComponent.define(
        'NestedComponent',
        states={'y': 0.0},
        laws=lambda s: [RateLaw(s.y, where(s.y < sin(where(mod(s.t, 2) < 1, 1, 0)), 1, 0))],
    )
    problem = compile_system(System(components=[nested('n')]))
    assert problem.drift(0.5, problem.initial_state, problem.initial_parameters)[0] == 1.0
    assert problem.drift(1.5, problem.initial_state, problem.initial_parameters)[0] == 0.0


DoublePendulumComponent = ElementComponent.define(
    'DoublePendulumComponent',
    states={'a': state(default=1.0, order=2), 'b': state(default=0.5, order=2)},
    parameters={'g': 9.81},
    laws=lambda s: [
        EquationLaw(
            2 * derivative(s.a, 2) + cos(s.a - s.b) * derivative(s.b, 2)
            == -(derivative(s.b, 1) ** 2) * sin(s.a - s.b) - 2 * s.g * sin(s.a),
        ),
        EquationLaw(
            derivative(s.b, 2) + cos(s.a - s.b) * derivative(s.a, 2)
            == derivative(s.a, 1) ** 2 * sin(s.a - s.b) - s.g * sin(s.b),
        ),
    ],
)


TooFarComponent = ElementComponent.define(
    'TooFarComponent',
    states={'x': state(default=1.0, lags=1)},
    laws=lambda s: [UpdateLaw(s.x, lag(s.x, 2))],
)


JerkComponent = ElementComponent.define(
    'JerkComponent',
    states={'x': state(default=0.0, order=3, initial=(0.0, 1.0))},
    parameters={'A': 0.6},
    laws=lambda s: [RateLaw(s.x, -s.A * derivative(s.x, 2) - derivative(s.x, 1) + abs_symbolic(s.x) - 1)],
)


def test_coupled_equations_conserve_energy():
    """Test that the double pendulum's coupled equations are solved jointly and conserve its energy."""
    trajectory = (
        Simulation(System(components=[DoublePendulumComponent('dp')]), solver=PRECISE, dt=0.05)
        .run(until=10)
        .trajectory_
    )
    a, b = trajectory.values('dp.a'), trajectory.values('dp.b')
    a_d1, b_d1 = trajectory.values('dp.a_d1'), trajectory.values('dp.b_d1')
    energy = a_d1**2 + 0.5 * b_d1**2 + a_d1 * b_d1 * np.cos(a - b) - 2 * 9.81 * np.cos(a) - 9.81 * np.cos(b)
    assert np.abs(energy - energy[0]).max() < 1e-7 * abs(energy[0])


def test_lag_beyond_kept_values_is_rejected():
    """Test that reading a lag the state does not keep raises an error naming the lag."""
    with pytest.raises(ValidationError, match=r'lag\(.*, 2\)'):
        Simulation(System(components=[TooFarComponent('t')])).run(until=1)


def test_third_order_jerk_is_bounded_and_chaotic():
    """Test that Sprott's jerk equation compiles to three equations and gives a bounded, sensitive trajectory."""
    first = Simulation(System(components=[JerkComponent('j')]), solver=PRECISE, dt=0.5).run(until=300)
    assert first.problem_.states.names == ['j.x', 'j.x_d1', 'j.x_d2']
    second = Simulation(System(components=[JerkComponent('j', x=1e-8)]), solver=PRECISE, dt=0.5).run(until=300)
    x = first.trajectory_.values('j.x')
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() < 10
    assert np.abs(x - second.trajectory_.values('j.x')).max() > 0.1


def test_derivative_at_order_is_rejected_in_rates():
    """Test that a rate cannot read the derivative of a state's own order."""
    BadInteraction = BaseInteraction.define(
        'BadInteraction',
        roles=['target'],
        laws=lambda s: [RateLaw(s.target.x, derivative(s.target.x, 2))],
    )
    with pytest.raises(ValidationError, match='below its declared order'):
        Simulation(System(components=[MassComponent('mass')], interactions=[BadInteraction('bad', target='mass')])).run(
            until=1,
        )
