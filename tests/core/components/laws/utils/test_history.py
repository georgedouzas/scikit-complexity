"""Test the `_history` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import abs_symbolic, cos, sin

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EquationLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import derivative, lag, residence, where
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

from ..conftest import PRECISE, MassElement


class DoublePendulumElement(Element):
    """Double pendulum element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec(order=2)] = 1.0,
        a_d1: float = 0.0,
        b: Annotated[object, StateSpec(order=2)] = 0.5,
        b_d1: float = 0.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.a_d1 = a_d1
        self.b = b
        self.b_d1 = b_d1
        self.g = g

    def _get_laws(self, s):
        """Return the laws of the double pendulum element."""
        return [
            EquationLaw(
                2 * derivative(s.a, 2) + cos(s.a - s.b) * derivative(s.b, 2)
                == -(derivative(s.b, 1) ** 2) * sin(s.a - s.b) - 2 * s.g * sin(s.a),
            ),
            EquationLaw(
                derivative(s.b, 2) + cos(s.a - s.b) * derivative(s.a, 2)
                == derivative(s.a, 1) ** 2 * sin(s.a - s.b) - s.g * sin(s.b),
            ),
        ]


class TooFarElement(Element):
    """Too far element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(lags=1)] = 1.0,
        x_lag1: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_lag1 = x_lag1

    def _get_laws(self, s):
        """Return the laws of the too far element."""
        return [UpdateLaw(s.x, lag(s.x, 2))]


class JerkElement(Element):
    """Jerk element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=3, initial=(0.0, 1.0))] = 0.0,
        x_d2: float = 1.0,
        x_d1: float = 0.0,
        A: Annotated[object, ParameterSpec()] = 0.6,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d2 = x_d2
        self.x_d1 = x_d1
        self.A = A

    def _get_laws(self, s):
        """Return the laws of the jerk element."""
        return [RateLaw(s.x, -s.A * derivative(s.x, 2) - derivative(s.x, 1) + abs_symbolic(s.x) - 1)]


def test_coupled_equations_conserve_energy():
    """Test that the double pendulum's coupled equations are solved jointly and conserve its energy."""
    trajectory = simulate(System(elements=[DoublePendulumElement('dp')]), until=10, solver=PRECISE, dt=0.05)
    a, b = trajectory.values('dp.a'), trajectory.values('dp.b')
    a_d1, b_d1 = trajectory.values('dp.a_d1'), trajectory.values('dp.b_d1')
    energy = a_d1**2 + 0.5 * b_d1**2 + a_d1 * b_d1 * np.cos(a - b) - 2 * 9.81 * np.cos(a) - 9.81 * np.cos(b)
    assert np.abs(energy - energy[0]).max() < 1e-7 * abs(energy[0])


def test_lag_beyond_kept_values_is_rejected():
    """Test that reading a lag the state does not keep raises an error naming the lag."""
    with pytest.raises(ValidationError, match=r'lag\(.*, 2\)'):
        simulate(System(elements=[TooFarElement('t')]), until=1)


def test_third_order_jerk_is_bounded_and_chaotic():
    """Test that Sprott's jerk equation compiles to three equations and gives a bounded, sensitive trajectory."""
    first = simulate(System(elements=[JerkElement('j')]), until=300, solver=PRECISE, dt=0.5)
    assert first.problem.states.names == ['j.x', 'j.x_d1', 'j.x_d2']
    second = simulate(System(elements=[JerkElement('j', x=1e-8)]), until=300, solver=PRECISE, dt=0.5)
    x = first.values('j.x')
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() < 10
    assert np.abs(x - second.values('j.x')).max() > 0.1


def test_derivative_at_order_is_rejected_in_rates():
    """Test that a rate cannot read the derivative of a state's own order."""

    class BadInteraction(Interaction):
        """Bad interaction."""

        def __init__(
            self,
            label: str,
            *,
            target: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.target = target

        def _get_laws(self, s):
            """Return the laws of the bad interaction."""
            return [RateLaw(s.target.x, derivative(s.target.x, 2))]

    with pytest.raises(ValidationError, match='below its declared order'):
        simulate(System(elements=[MassElement('mass')], interactions=[BadInteraction('bad', target='mass')]), until=1)


class ToggleElement(Element):
    """Value that switches between 0 and 1 once it has held its value for a given number of steps."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        held: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.held = held

    def _get_laws(self, s):
        """Return the switch of the value after it has held it for `held` steps."""
        return [UpdateLaw(s.x, where(residence(s.x) >= s.held, 1 - s.x, s.x))]


def test_residence_counts_the_steps_a_state_holds_its_value():
    """Test that the residence is 0 in the step a state takes a value and grows by 1 every step it keeps it."""
    group = GroupElement('toggles', element=ToggleElement, size=2, values={'held': [2.0, 1.0]})
    trajectory = simulate(System(elements=[group]), until=6)
    np.testing.assert_array_equal(
        trajectory.values('toggles.x').T,
        [[0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 1]],
    )


def test_residence_outside_update_laws_is_rejected():
    """Test that a law other than an update law cannot read the residence of a state."""

    class ResidenceRateInteraction(Interaction):
        """Rate that reads a residence."""

        def __init__(
            self,
            label: str,
            *,
            target: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.target = target

        def _get_laws(self, s):
            """Return a rate that reads the residence of the position."""
            return [RateLaw(s.target.x, residence(s.target.x))]

    system = System(elements=[MassElement('mass')], interactions=[ResidenceRateInteraction('bad', target='mass')])
    with pytest.raises(ValidationError, match='only update laws'):
        simulate(system, until=1)
