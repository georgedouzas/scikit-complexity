"""Test the linearisation of systems and the analysis of their controllability, observability, and regulators."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.analysis import LinearizationAnalysis
from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw, UpdateLaw
from skcomplex.core.system import System
from skcomplex.core.times import DiscreteTime
from skcomplex.errors import AnalysisError, ValidationError
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec


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


class CartElement(Element):
    """Cart element, a double integrator of its force."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        force: Annotated[object, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.force = force

    def _get_laws(self, s):
        """Return the laws of the cart element."""
        return [RateLaw(s.x, s.force)]


def _linearization(system, **settings):
    """Return the linearization analysis of a system, run."""
    return LinearizationAnalysis(**settings).check().prepare(system).run()


def test_linearize_uncontrollable_mode():
    """Test that two oscillators of which only one receives the control have controllability rank 2 of 4."""
    model = _linearization(System(elements=[DrivenElement('a'), FreeElement('b')]))
    assert model.states_ == ('a.x', 'a.x_d1', 'b.x', 'b.x_d1')
    assert model.controls_ == ('a.force',)
    assert model.A_.shape == (4, 4)
    assert model.controllability_ == 2
    assert model.observability_ == 4
    assert model.gain_ is None


def test_linearize_at_an_operating_point():
    """Test that the pendulum linearised upright has the unstable entry `cos(pi) = -1` negated in its state matrix."""
    model = _linearization(
        System(elements=[PendulumElement('p')]),
        outputs=['p.theta', 'p.torque'],
        at={'p.theta': np.pi},
    )
    np.testing.assert_allclose(model.A_, [[0, 1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(model.B_, [[0], [1]])
    np.testing.assert_allclose(model.C_, [[1, 0], [0, 0]])
    np.testing.assert_allclose(model.D_, [[0], [1]])
    observed = _linearization(System(elements=[PendulumElement('p')]), outputs=['p.theta'], at={'p.theta': np.pi})
    assert observed.observability_ == 2


def test_linearize_rejects_unknown_controls():
    """Test that a control that is not an adaptive parameter is rejected."""
    with pytest.raises(AnalysisError, match='not adaptive parameters'):
        _linearization(System(elements=[PendulumElement('p')]), controls=['p.theta'])


def test_linearize_rejects_discrete_time():
    """Test that a system that is not under continuous time is rejected as invalid."""
    with pytest.raises(ValidationError, match='continuous time'):
        _linearization(System(elements=[CounterElement('c')], time=DiscreteTime()))


def test_lqr_stabilises_a_double_integrator():
    """Test that the regulator gain of a double integrator is `[1, sqrt(3)]` and makes the closed loop stable."""
    model = _linearization(System(elements=[CartElement('cart')]), state_weights=np.eye(2), control_weights=[[1.0]])
    np.testing.assert_allclose(model.A_, [[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(model.gain_, [[1.0, np.sqrt(3.0)]])
    assert np.all(np.linalg.eigvals(model.A_ - model.B_ @ model.gain_).real < 0)


def test_regulator_needs_both_weights():
    """Test that state weights without control weights are rejected at check."""
    with pytest.raises(AnalysisError, match='both state_weights and control_weights'):
        LinearizationAnalysis(state_weights=np.eye(2)).check()
