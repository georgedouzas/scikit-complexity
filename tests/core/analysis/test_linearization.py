"""Test the linearisation of systems and the analysis of their controllability, observability, and regulators."""

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.analysis import linearization
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.errors import AnalysisError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role


def _system(label, laws, **values):
    """Return one element of values whose interaction `laws` sets laws of the namespace."""
    interaction = Interaction('laws', roles={'target': label}, laws=laws)
    return System(elements=[Element(label, values=values)], interactions=[interaction])


def _oscillators():
    """Return an oscillator `a` that a force drives and a free oscillator `b`."""
    driven = Interaction(
        'driven',
        roles={'target': 'a'},
        laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -(s.target.w**2) * s.target.x + s.target.force)],
    )
    free = Interaction(
        'free',
        roles={'target': 'b'},
        laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -(s.target.w**2) * s.target.x)],
    )
    elements = [Element('a', values={'x': 0.0, 'force': 0.0, 'w': 1.0}), Element('b', values={'x': 0.0, 'w': 2.0})]
    return System(elements=elements, interactions=[driven, free])


def _pendulum():
    """Return a pendulum that a torque drives."""
    return _system(
        'p',
        lambda s: [RateLaw(s.target.theta.diff(s.t, 2), -sin(s.target.theta) + s.target.torque)],
        theta=0.0,
        torque=0.0,
    )


def _cart():
    """Return a cart, a double integrator of its force."""
    return _system('cart', lambda s: [RateLaw(s.target.x.diff(s.t, 2), s.target.force)], x=0.0, force=0.0)


def test_linearize_uncontrollable_mode():
    """Test that two oscillators of which only one receives the control have controllability rank 2 of 4."""
    system = linearization(_oscillators(), controls=['a.force'])
    assert system.states == ('a.x', "a.x'", 'b.x', "b.x'")
    assert system.controls == ('a.force',)
    assert system.A.shape == (4, 4)
    assert system.controllability == 2
    assert system.observability == 4
    assert system.gain is None


def test_linearize_at_an_operating_point():
    """Test that the pendulum linearised upright has the unstable entry `cos(pi) = -1` negated in its state matrix."""
    system = linearization(
        _pendulum(),
        controls=['p.torque'],
        outputs=['p.theta', 'p.torque'],
        at={'p.theta': np.pi},
    )
    np.testing.assert_allclose(system.A, [[0, 1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(system.B, [[0], [1]])
    np.testing.assert_allclose(system.C, [[1, 0], [0, 0]])
    np.testing.assert_allclose(system.D, [[0], [1]])
    observed = linearization(
        _pendulum(),
        controls=['p.torque'],
        outputs=['p.theta'],
        at={'p.theta': np.pi},
    )
    assert observed.observability == 2


def test_linearize_rejects_unknown_controls():
    """Test that a control that is not a parameter is rejected."""
    with pytest.raises(AnalysisError, match='not parameters or adaptive parameters'):
        linearization(_pendulum(), controls=['p.theta'])


def test_linearize_rejects_discrete_time():
    """Test that a system that is not under continuous time is rejected as invalid."""
    with pytest.raises(ValidationError, match='continuous time'):
        linearization(_system('c', lambda s: [UpdateLaw(s.target.k, s.target.k + 1)], k=0.0))


def test_lqr_stabilises_a_double_integrator():
    """Test that the regulator gain of a double integrator is `[1, sqrt(3)]` and makes the closed loop stable."""
    system = linearization(
        _cart(),
        controls=['cart.force'],
        state_weights=np.eye(2),
        control_weights=[[1.0]],
    )
    np.testing.assert_allclose(system.A, [[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(system.gain, [[1.0, np.sqrt(3.0)]])
    assert np.all(np.linalg.eigvals(system.A - system.B @ system.gain).real < 0)


def test_regulator_needs_both_weights():
    """Test that state weights without control weights are rejected."""
    with pytest.raises(AnalysisError, match='both state_weights and control_weights'):
        linearization(_cart(), controls=['cart.force'], state_weights=np.eye(2))


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
