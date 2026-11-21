"""Test the `_constraint` module of `skcomplex.core.components.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, sin, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, RateLaw
from skcomplex.core.components.laws.utils import step, where
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

from .conftest import PRECISE

POSITIVE = RealSet.open(0, oo)
PENDULUM = {'theta': 0.3, 'm': 2.0, 'g': 9.81, 'l': 1.5}
SPRING = {'x': 1.0, 'm': 2.0, 'k': 3.0}


def _run(label, values, laws, until, dt, *, interactions=()):
    """Return the run of one element whose laws are a function of the namespace of its target."""
    rule = Interaction('laws', roles={'target': label}, laws=laws)
    system = System(elements=[Element(label, values=values)], interactions=[rule, *interactions], solver=PRECISE)
    return system.prepare().simulate(until=until, dt=dt)


def _written(s):
    """Return the pendulum written as Newton's equation."""
    b = s.target
    return [ConstraintLaw(b.m * b.l * b.theta.diff(s.t, 2) == -b.m * b.g * sin(b.theta))]


def _solved(s):
    """Return the pendulum written as the rate of its angular velocity."""
    b = s.target
    return [RateLaw(b.theta.diff(s.t, 2), -(b.g / b.l) * sin(b.theta))]


def _driven_equation(s):
    """Return a mass on a spring pushed by a step at time 1, as an equation."""
    m = s.target
    return [ConstraintLaw(m.m * m.x.diff(s.t, 2) == -m.k * m.x + step(s.t, 1))]


def _driven_rate(s):
    """Return a mass on a spring pushed by a step at time 1, as a rate law."""
    m = s.target
    return [RateLaw(m.x.diff(s.t, 2), (-m.k * m.x + step(s.t, 1)) / m.m)]


def _switched_equation(s):
    """Return a spring that stiffens on the positive side, as an equation."""
    m = s.target
    return [ConstraintLaw(m.m * m.x.diff(s.t, 2) == where(m.x < 0, -m.k * m.x, -2 * m.k * m.x))]


def _switched_rate(s):
    """Return a spring that stiffens on the positive side, as a rate law."""
    m = s.target
    return [RateLaw(m.x.diff(s.t, 2), where(m.x < 0, -m.k * m.x, -2 * m.k * m.x) / m.m)]


def test_equation_takes_two_sides_or_a_relation():
    """Test that an equation takes either a relation or two sides, and splits a relation into its sides."""
    x, h = var('x h')
    law = ConstraintLaw(x == h)
    assert law._get_sides() == (x, h)
    assert ConstraintLaw(1, 2).rhs == 2
    with pytest.raises(ValidationError, match='left and a right side'):
        ConstraintLaw(1)._check()


def test_pendulum_as_written_matches_solved_form():
    """Test that `m l theta'' = -m g sin theta` runs exactly like the rate law `theta'' = -(g / l) sin theta`."""
    written = _run('bob', PENDULUM, _written, 5, 0.1)
    solved = _run('bob', PENDULUM, _solved, 5, 0.1)
    assert np.array_equal(written.get_values('bob.theta'), solved.get_values('bob.theta'))


def test_rates_add_to_equation():
    """Test that a rate law on a state with an equation adds to the solved highest derivative."""
    push = Interaction('push', roles={'target': 'bob'}, laws=lambda s: [RateLaw(s.target.theta.diff(s.t, 2), 1.0)])
    trajectory = _run('bob', PENDULUM | {'theta': 0.0}, _written, 0.1, 0.01, interactions=[push])
    assert trajectory.get_values('bob.theta')[-1] == pytest.approx(0.5 * 0.1**2, rel=1e-2)


def test_nonlinear_highest_derivative_is_rejected():
    """Test that an equation with the square of a highest derivative names the equation and the derivative."""
    with pytest.raises(ValidationError, match=r'equation .* of laws is not linear in the highest derivative x_d1'):
        _run('s', {'x': 1.0}, lambda s: [ConstraintLaw(s.target.x.diff(s.t, 2) ** 2, s.target.x)], 1, None)


def test_many_equations_are_solved_numerically():
    """Test that seven equations, above the symbolic size, are solved per evaluation with the same result."""
    values = {f'x{index}': 1.0 for index in range(7)}

    def laws(s):
        return [ConstraintLaw(2 * getattr(s.target, name).diff(s.t, 2), -getattr(s.target, name)) for name in values]

    trajectory = _run('o', values, laws, 3, 0.1)
    expected = np.cos(trajectory.times / np.sqrt(2))
    for index in range(7):
        assert np.allclose(trajectory.get_values(f'o.x{index}'), expected, atol=1e-7)


@pytest.mark.parametrize(
    ('equation', 'rate'),
    [(_driven_equation, _driven_rate), (_switched_equation, _switched_rate)],
    ids=['driver', 'condition'],
)
def test_equation_with_driver_or_condition_matches_rate(equation, rate):
    """Test that equations with a driver or a condition on the right side run like their rate laws."""
    first = _run('mass', SPRING, equation, 3, 0.1)
    second = _run('mass', SPRING, rate, 3, 0.1)
    assert first.get_values('mass.x') == pytest.approx(second.get_values('mass.x'), abs=1e-8)


def test_check_names_equations_as_symbolic_only():
    """Test that a function as a side of an equation raises an error pointing to a rate law."""
    with pytest.raises(ValidationError, match=r'solved symbolically.*`RateLaw`'):
        ConstraintLaw(var('x'), print)._check()
