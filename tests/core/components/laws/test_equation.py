"""Test the `_equation` module of `skcomplex.core.components.laws`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, sin, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EquationLaw, RateLaw
from skcomplex.core.components.laws.utils import derivative, step, where
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

from .conftest import PRECISE

POSITIVE = RealSet.open(0, oo)


class WrittenElement(Element):
    """Pendulum written as Newton's equation."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.3,
        theta_d1: float = 0.0,
        m: Annotated[object, ParameterSpec(domain=POSITIVE)] = 2.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        l: Annotated[object, ParameterSpec()] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m = m
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the equation of motion."""
        return [EquationLaw(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))]


class SolvedElement(Element):
    """Pendulum written as the rate of its angular velocity."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.3,
        theta_d1: float = 0.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        l: Annotated[object, ParameterSpec()] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the rate of the angular velocity."""
        return [RateLaw(s.theta, -(s.g / s.l) * sin(s.theta))]


class SquaredElement(Element):
    """Element whose equation holds the square of its highest derivative."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1

    def _get_laws(self, s):
        """Return the nonlinear equation."""
        return [EquationLaw(derivative(s.x, 2) ** 2, s.x)]


class OscillatorsElement(Element):
    """Seven oscillators, one equation each, more than the equations solved symbolically."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x0: Annotated[object, StateSpec(order=2)] = 1.0,
        x1: Annotated[object, StateSpec(order=2)] = 1.0,
        x2: Annotated[object, StateSpec(order=2)] = 1.0,
        x3: Annotated[object, StateSpec(order=2)] = 1.0,
        x4: Annotated[object, StateSpec(order=2)] = 1.0,
        x5: Annotated[object, StateSpec(order=2)] = 1.0,
        x6: Annotated[object, StateSpec(order=2)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6

    def _get_laws(self, s):
        """Return one equation per oscillator."""
        states = (s.x0, s.x1, s.x2, s.x3, s.x4, s.x5, s.x6)
        return [EquationLaw(2 * derivative(x, 2), -x) for x in states]


class SpringElement(Element):
    """Mass on a spring, whose laws the subclasses write."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
        m: Annotated[object, ParameterSpec()] = 2.0,
        k: Annotated[object, ParameterSpec()] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.m = m
        self.k = k


class DrivenEquationElement(SpringElement):
    """Mass on a spring pushed by a step at time 1, as an equation."""

    def _get_laws(self, s):
        """Return the equation of motion."""
        return [EquationLaw(s.m * derivative(s.x, 2) == -s.k * s.x + step(s.t, 1))]


class DrivenRateElement(SpringElement):
    """Mass on a spring pushed by a step at time 1, as a rate law."""

    def _get_laws(self, s):
        """Return the rate of the velocity."""
        return [RateLaw(s.x, (-s.k * s.x + step(s.t, 1)) / s.m)]


class SwitchedEquationElement(SpringElement):
    """Spring that stiffens on the positive side, as an equation."""

    def _get_laws(self, s):
        """Return the equation of motion."""
        return [EquationLaw(s.m * derivative(s.x, 2) == where(s.x < 0, -s.k * s.x, -2 * s.k * s.x))]


class SwitchedRateElement(SpringElement):
    """Spring that stiffens on the positive side, as a rate law."""

    def _get_laws(self, s):
        """Return the rate of the velocity."""
        return [RateLaw(s.x, where(s.x < 0, -s.k * s.x, -2 * s.k * s.x) / s.m)]


class PushInteraction(Interaction):
    """Constant push on the angle of its target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        f: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.f = f

    def _get_laws(self, s):
        """Return the push."""
        return [RateLaw(s.target.theta, s.f)]


def test_equation_takes_two_sides_or_a_relation():
    """Test that an equation takes either a relation or two sides, and prepares both sides."""
    x, h = var('x h')
    law = EquationLaw(x == h).prepare()
    assert (law.lhs_, law.rhs_) == (x, h)
    assert EquationLaw(1, 2).rhs == 2
    with pytest.raises(ValidationError, match='left and a right side'):
        EquationLaw(1).prepare()


def test_pendulum_as_written_matches_solved_form():
    """Test that `m l theta'' = -m g sin theta` runs exactly like the rate law `theta'' = -(g / l) sin theta`."""
    written = simulate(System(elements=[WrittenElement('bob')]), until=5, solver=PRECISE, dt=0.1)
    solved = simulate(System(elements=[SolvedElement('bob')]), until=5, solver=PRECISE, dt=0.1)
    assert np.array_equal(written.values('bob.theta'), solved.values('bob.theta'))


def test_rates_add_to_equation():
    """Test that a rate law on a state with an equation adds to the solved highest derivative."""
    pushed = System(elements=[WrittenElement('bob', theta=0.0)], interactions=[PushInteraction('push', target='bob')])
    trajectory = simulate(pushed, until=0.1, solver=PRECISE, dt=0.01)
    assert trajectory.values('bob.theta')[-1] == pytest.approx(0.5 * 0.1**2, rel=1e-2)


def test_nonlinear_highest_derivative_is_rejected():
    """Test that an equation with the square of a highest derivative names the equation and the derivative."""
    with pytest.raises(ValidationError, match=r'equation .* of s is not linear in the highest derivative x_d1'):
        simulate(System(elements=[SquaredElement('s')]), until=1)


def test_many_equations_are_solved_numerically():
    """Test that seven equations, above the symbolic size, are solved per evaluation with the same result."""
    trajectory = simulate(System(elements=[OscillatorsElement('o')]), until=3, solver=PRECISE, dt=0.1)
    expected = np.cos(trajectory.times / np.sqrt(2))
    for index in range(7):
        assert np.allclose(trajectory.values(f'o.x{index}'), expected, atol=1e-7)


@pytest.mark.parametrize(
    ('equation', 'rate'),
    [(DrivenEquationElement, DrivenRateElement), (SwitchedEquationElement, SwitchedRateElement)],
    ids=['driver', 'condition'],
)
def test_equation_with_driver_or_condition_matches_rate(equation, rate):
    """Test that equations with a driver or a condition on the right side run like their rate laws."""
    first = simulate(System(elements=[equation('mass')]), until=3, solver=PRECISE, dt=0.1)
    second = simulate(System(elements=[rate('mass')]), until=3, solver=PRECISE, dt=0.1)
    assert first.values('mass.x') == pytest.approx(second.values('mass.x'), abs=1e-8)


def test_check_names_equations_as_symbolic_only():
    """Test that a function as a side of an equation raises an error pointing to a rate law."""
    with pytest.raises(ValidationError, match=r'solved symbolically.*`RateLaw`'):
        EquationLaw(var('x'), print).prepare()
