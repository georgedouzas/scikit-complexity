"""Test the linearisation of systems and the analysis of their controllability, observability, and regulators."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.analysis import controllability, linearize, lqr, observability
from skcomplex.core.base import AnalysisError, ValidationError, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import Rate, Update
from skcomplex.core.space import DiscreteTime
from skcomplex.core.system import System

Driven = Element.define(
    'Driven',
    states={'x': state(0.0, order=2)},
    parameters={'force': parameter(0.0, adaptive=True, bounds=(-1, 1)), 'w': 1.0},
    laws=lambda s: [Rate(s.x, -(s.w**2) * s.x + s.force)],
)
Free = Element.define(
    'Free',
    states={'x': state(0.0, order=2)},
    parameters={'w': 2.0},
    laws=lambda s: [Rate(s.x, -(s.w**2) * s.x)],
)
Pendulum = Element.define(
    'Pendulum',
    states={'theta': state(0.0, order=2)},
    parameters={'torque': parameter(0.0, adaptive=True, bounds=(-1, 1))},
    laws=lambda s: [Rate(s.theta, -sin(s.theta) + s.torque)],
)
Counter = Element.define('Counter', states={'k': 0.0}, laws=lambda s: [Update(s.k, s.k + 1)])


def test_linearize_uncontrollable_mode():
    """Test that two oscillators of which only one receives the control have controllability rank 2 of 4."""
    model = linearize(System(components=[Driven('a'), Free('b')]))
    assert model.states == ('a.x', 'a.x_d1', 'b.x', 'b.x_d1')
    assert model.controls == ('a.force',)
    assert model.A.shape == (4, 4)
    assert controllability(model.A, model.B) == 2
    assert observability(model.A, model.C) == 4


def test_linearize_at_an_operating_point():
    """Test that the pendulum linearised upright has the unstable entry `cos(pi) = -1` negated in its state matrix."""
    model = linearize(System(components=[Pendulum('p')]), outputs=['p.theta', 'p.torque'], at={'p.theta': np.pi})
    np.testing.assert_allclose(model.A, [[0, 1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(model.B, [[0], [1]])
    np.testing.assert_allclose(model.C, [[1, 0], [0, 0]])
    np.testing.assert_allclose(model.D, [[0], [1]])
    assert observability(model.A, model.C[:1]) == 2


def test_linearize_rejects_unknown_controls():
    """Test that a control that is not an adaptive parameter is rejected."""
    with pytest.raises(AnalysisError, match='not adaptive parameters'):
        linearize(System(components=[Pendulum('p')]), controls=['p.theta'])


def test_linearize_rejects_discrete_time():
    """Test that a system that is not under continuous time is rejected as invalid."""
    with pytest.raises(ValidationError, match='continuous time'):
        linearize(System(components=[Counter('c')], time=DiscreteTime()))


def test_lqr_stabilises_a_double_integrator():
    """Test that the regulator gain of a double integrator is `[1, sqrt(3)]` and makes the closed loop stable."""
    a, b = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])
    gain = lqr(a, b, np.eye(2), [[1.0]])
    np.testing.assert_allclose(gain, [[1.0, np.sqrt(3.0)]])
    assert np.all(np.linalg.eigvals(a - b @ gain).real < 0)
