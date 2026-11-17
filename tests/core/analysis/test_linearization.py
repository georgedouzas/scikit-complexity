"""Test the linearisation of systems and the analysis of their controllability, observability, and regulators."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.analysis import linearization
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import AnalysisError, ValidationError
from skcomplex.core.specs import ParameterSpec, Role, StateSpec
from skcomplex.core.systems import System


class DrivenElement(Element):
    """Driven element."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        force: Annotated[object, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
        w: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1
        self.force = force
        self.w = w

    def _get_laws(self, s):
        """Return the laws of the driven element."""
        return [RateLaw(s.x, -(s.w**2) * s.x + s.force)]


class FreeElement(Element):
    """Free element."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        w: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1
        self.w = w

    def _get_laws(self, s):
        """Return the laws of the free element."""
        return [RateLaw(s.x, -(s.w**2) * s.x)]


class PendulumElement(Element):
    """Pendulum element."""

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.0,
        theta_d1: float = 0.0,
        torque: Annotated[object, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.torque = torque

    def _get_laws(self, s):
        """Return the laws of the pendulum element."""
        return [RateLaw(s.theta, -sin(s.theta) + s.torque)]


class CounterElement(Element):
    """Counter element."""

    def __init__(
        self,
        label: str,
        *,
        k: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.k = k

    def _get_laws(self, s):
        """Return the laws of the counter element."""
        return [UpdateLaw(s.k, s.k + 1)]


class CartElement(Element):
    """Cart element, a double integrator of its force."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        force: Annotated[object, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1
        self.force = force

    def _get_laws(self, s):
        """Return the laws of the cart element."""
        return [RateLaw(s.x, s.force)]


def test_linearize_uncontrollable_mode():
    """Test that two oscillators of which only one receives the control have controllability rank 2 of 4."""
    system = linearization(CompositeSystem(elements=[DrivenElement('a'), FreeElement('b')]))
    assert system.states == ('a.x', "a.x'", 'b.x', "b.x'")
    assert system.controls == ('a.force',)
    assert system.A.shape == (4, 4)
    assert system.controllability == 2
    assert system.observability == 4
    assert system.gain is None


def test_linearize_at_an_operating_point():
    """Test that the pendulum linearised upright has the unstable entry `cos(pi) = -1` negated in its state matrix."""
    system = linearization(
        CompositeSystem(elements=[PendulumElement('p')]),
        outputs=['p.theta', 'p.torque'],
        at={'p.theta': np.pi},
    )
    np.testing.assert_allclose(system.A, [[0, 1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(system.B, [[0], [1]])
    np.testing.assert_allclose(system.C, [[1, 0], [0, 0]])
    np.testing.assert_allclose(system.D, [[0], [1]])
    observed = linearization(
        CompositeSystem(elements=[PendulumElement('p')]),
        outputs=['p.theta'],
        at={'p.theta': np.pi},
    )
    assert observed.observability == 2


def test_linearize_rejects_unknown_controls():
    """Test that a control that is not a parameter is rejected."""
    with pytest.raises(AnalysisError, match='not parameters or adaptive parameters'):
        linearization(CompositeSystem(elements=[PendulumElement('p')]), controls=['p.theta'])


def test_linearize_rejects_discrete_time():
    """Test that a system that is not under continuous time is rejected as invalid."""
    with pytest.raises(ValidationError, match='continuous time'):
        linearization(CompositeSystem(elements=[CounterElement('c')], time=DiscreteTime()))


def test_lqr_stabilises_a_double_integrator():
    """Test that the regulator gain of a double integrator is `[1, sqrt(3)]` and makes the closed loop stable."""
    system = linearization(
        CompositeSystem(elements=[CartElement('cart')]),
        state_weights=np.eye(2),
        control_weights=[[1.0]],
    )
    np.testing.assert_allclose(system.A, [[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(system.gain, [[1.0, np.sqrt(3.0)]])
    assert np.all(np.linalg.eigvals(system.A - system.B @ system.gain).real < 0)


def test_regulator_needs_both_weights():
    """Test that state weights without control weights are rejected."""
    with pytest.raises(AnalysisError, match='both state_weights and control_weights'):
        linearization(CompositeSystem(elements=[CartElement('cart')]), state_weights=np.eye(2))


class PushInteraction(Interaction):
    """Push `x'' = force` of the element it binds."""

    def __init__(self, label, *, target: Role):
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            cart, t = context.target, context.t
            self.laws_ = [RateLaw(cart.x.diff(t, 2), cart.force)]


def test_linearize_a_system_with_a_parameter_as_control():
    """Test that a plain parameter no law changes is a control of a `System`, and that derivatives carry primes."""
    cart = Element('cart', values={'x': 0.0, 'force': 0.0})
    system = System(elements=[cart], interactions=[PushInteraction('push', target='cart')])
    linear = linearization(system, controls=['cart.force'], state_weights=np.eye(2), control_weights=[[1.0]])
    assert (linear.states, linear.controls) == (('cart.x', "cart.x'"), ('cart.force',))
    np.testing.assert_allclose(linear.B, [[0.0], [1.0]])
    np.testing.assert_allclose(linear.gain, [[1.0, np.sqrt(3.0)]])
    assert linearization(system, controls=['cart.force'], outputs=["cart.x'"]).outputs == ("cart.x'",)
