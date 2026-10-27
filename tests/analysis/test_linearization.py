"""Test the linearisation of systems and the analysis of their controllability, observability, and regulators."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.analysis import controllability, linearize, lqr, observability
from skcomplex.core.base import AnalysisError
from skcomplex.core.elements import Element
from skcomplex.core.errors import ValidationError
from skcomplex.core.laws import RateLaw, UpdateLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.core.system import System


class DrivenElement(Element):
    """Driven element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        force: Annotated[object, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
        w: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
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
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        w: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
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
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.0,
        theta_d1: float = 0.0,
        torque: Annotated[object, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
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
        visual: VisualSpec | None = None,
        *,
        k: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k

    def _get_laws(self, s):
        """Return the laws of the counter element."""
        return [UpdateLaw(s.k, s.k + 1)]


def test_linearize_uncontrollable_mode():
    """Test that two oscillators of which only one receives the control have controllability rank 2 of 4."""
    model = linearize(System(elements=[DrivenElement('a'), FreeElement('b')]))
    assert model.states == ('a.x', 'a.x_d1', 'b.x', 'b.x_d1')
    assert model.controls == ('a.force',)
    assert model.A.shape == (4, 4)
    assert controllability(model.A, model.B) == 2
    assert observability(model.A, model.C) == 4


def test_linearize_at_an_operating_point():
    """Test that the pendulum linearised upright has the unstable entry `cos(pi) = -1` negated in its state matrix."""
    model = linearize(
        System(elements=[PendulumElement('p')]),
        outputs=['p.theta', 'p.torque'],
        at={'p.theta': np.pi},
    )
    np.testing.assert_allclose(model.A, [[0, 1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(model.B, [[0], [1]])
    np.testing.assert_allclose(model.C, [[1, 0], [0, 0]])
    np.testing.assert_allclose(model.D, [[0], [1]])
    assert observability(model.A, model.C[:1]) == 2


def test_linearize_rejects_unknown_controls():
    """Test that a control that is not an adaptive parameter is rejected."""
    with pytest.raises(AnalysisError, match='not adaptive parameters'):
        linearize(System(elements=[PendulumElement('p')]), controls=['p.theta'])


def test_linearize_rejects_discrete_time():
    """Test that a system that is not under continuous time is rejected as invalid."""
    with pytest.raises(ValidationError, match='continuous time'):
        linearize(System(elements=[CounterElement('c')], time=DiscreteTime()))


def test_lqr_stabilises_a_double_integrator():
    """Test that the regulator gain of a double integrator is `[1, sqrt(3)]` and makes the closed loop stable."""
    a, b = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])
    gain = lqr(a, b, np.eye(2), [[1.0]])
    np.testing.assert_allclose(gain, [[1.0, np.sqrt(3.0)]])
    assert np.all(np.linalg.eigvals(a - b @ gain).real < 0)
