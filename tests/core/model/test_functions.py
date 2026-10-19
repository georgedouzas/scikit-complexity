"""Test the symbolic functions for conditions and remainders."""

import numpy as np
import pytest

from skcomplex.core.base import symbol
from skcomplex.core.compile import NUMPY_DIALECT, SCALAR_DIALECT, numpy_function, print_expressions
from skcomplex.core.model import EuclideanSpace, Particle, PeriodicBox, ReflectingBox, System, both, either, mod, where
from skcomplex.core.runtime import Simulation


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
        elements=[Particle('p', x=[0.9], v=[1.0])],
        interactions=[ReflectingBox('box', target='p', size=1.0)],
        space=EuclideanSpace(1),
    )
    trajectory = Simulation(system, dt=0.2).run(until=0.2).trajectory_
    assert trajectory['p.x'][-1, 0] == pytest.approx(0.9)
    assert trajectory['p.v'][-1, 0] == pytest.approx(-1.0)


def test_periodic_box_wraps_leaving_particle():
    """Test that a particle leaving through one side enters through the opposite side."""
    system = System(
        elements=[Particle('p', x=[0.9], v=[1.0])],
        interactions=[PeriodicBox('box', target='p', size=1.0)],
        space=EuclideanSpace(1),
    )
    trajectory = Simulation(system, dt=0.2).run(until=0.2).trajectory_
    assert trajectory['p.x'][-1, 0] == pytest.approx(0.1)
    assert trajectory['p.v'][-1, 0] == pytest.approx(1.0)
