"""Test the symbolic functions for conditions and remainders."""

import warnings

import numpy as np
import pytest

from skcomplex.core.base import symbol
from skcomplex.core.compile import NUMPY_DIALECT, SCALAR_DIALECT, compile_system, numpy_function, print_expressions
from skcomplex.core.model import (
    Element,
    EuclideanSpace,
    Particle,
    PeriodicBox,
    Rate,
    ReflectingBox,
    System,
    both,
    either,
    mod,
    series,
    where,
)
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, ScipyODE


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
        components=[Particle('p', x=[0.9], v=[1.0])],
        interactions=[ReflectingBox('box', target='p', size=1.0)],
        space=EuclideanSpace(1),
    )
    trajectory = Simulation(system, dt=0.2).run(until=0.2).trajectory_
    assert trajectory['p.x'][-1, 0] == pytest.approx(0.9)
    assert trajectory['p.v'][-1, 0] == pytest.approx(-1.0)


def test_periodic_box_wraps_leaving_particle():
    """Test that a particle leaving through one side enters through the opposite side."""
    system = System(
        components=[Particle('p', x=[0.9], v=[1.0])],
        interactions=[PeriodicBox('box', target='p', size=1.0)],
        space=EuclideanSpace(1),
    )
    trajectory = Simulation(system, dt=0.2).run(until=0.2).trajectory_
    assert trajectory['p.x'][-1, 0] == pytest.approx(0.1)
    assert trajectory['p.v'][-1, 0] == pytest.approx(1.0)


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


@pytest.mark.parametrize('solver', [RK4(), ScipyODE()], ids=['rk4', 'scipy'])
def test_long_frame_keeps_particle_in_reflecting_box(solver):
    """Test that a frame of duration 5 reflects a particle four times and ends at 0.5 with velocity -1."""
    system = System(
        components=[Particle('p', x=[0.5], v=[1.0])],
        interactions=[ReflectingBox('box', target='p', size=1.0)],
        space=EuclideanSpace(1),
    )
    trajectory = Simulation(system, solver=solver, dt=5.0).run(until=5).trajectory_
    assert trajectory['p.x'][-1, 0] == pytest.approx(0.5, abs=1e-6)
    assert trajectory['p.v'][-1, 0] == pytest.approx(-1.0, abs=1e-6)


def _driven_by(data):
    """Return a system whose one rate reads a series input."""
    driven = Element.define('Driven', states={'y': 0.0}, laws=lambda s: [Rate(s.y, data(s.t))])
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
