"""Test the `_rate` module of `skcomplex.core.components.laws`."""

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, sin, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, Period, Quantity
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.solvers import VelocityVerletSolver
from skcomplex.core.errors import ConservationError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

from .conftest import PRECISE, MassElement


class PendulumElement(Element):
    """Pendulum whose angle wraps around a turn, with its angular velocity the value `theta'`."""

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[float, Period(2 * math.pi)] = 0.1,
        values: dict[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.theta = theta


class SwingInteraction(Interaction):
    """Swing of a pendulum under gravity."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        g: Annotated[float, RealSet.open(0, oo)] = 9.81,
        l: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.g = g
        self.l = l

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the angular acceleration."""
        super()._prepare(context)
        s = context
        self.laws_ = [RateLaw(s.target.theta.diff(s.t, 2), -(s.g / s.l) * sin(s.target.theta))]


class SpringInteraction(Interaction):
    """Spring on the mass of its target."""

    def __init__(self, label: str, *, target: Role, k: float = 8.0) -> None:
        super().__init__(label)
        self.target = target
        self.k = k

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the force of the spring."""
        super()._prepare(context)
        s = context
        self.laws_ = [RateLaw(s.target.x.diff(s.t, 2), -s.k * s.target.x / s.target.m)]


class DamperInteraction(Interaction):
    """Damper on the mass of its target."""

    def __init__(self, label: str, *, target: Role, c: float = 0.4) -> None:
        super().__init__(label)
        self.target = target
        self.c = c

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the force of the damper."""
        super()._prepare(context)
        s = context
        self.laws_ = [RateLaw(s.target.x.diff(s.t, 2), -s.c * s.target.x.diff(s.t) / s.target.m)]


class MismatchedInteraction(Interaction):
    """Rate that gives the first derivative of a state of order 2."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the mismatched rate."""
        super()._prepare(context)
        s = context
        self.laws_ = [RateLaw(s.target.x.diff(s.t), -s.target.x)]


class LeakyElement(Element):
    """Money that a rate law changes outside any flow."""

    def __init__(self, label: str, *, money: Annotated[float, Quantity('money')] = 1.0) -> None:
        super().__init__(label)
        self.money = money


def _decay(label: str, state: str, rate: float) -> Interaction:
    """Return the decay of a state of an element at a rate."""
    return Interaction(
        f'{label}_decay',
        roles={'target': label},
        laws=lambda s: [RateLaw(getattr(s.target, state), rate * getattr(s.target, state))],
    )


def test_rate_gives_its_value():
    """Test that a rate law targets its state and gives its value at given values of the symbols."""
    n, r = var('n r')
    law = RateLaw(n, r * n)
    assert law._get_targets() == (n,)
    assert law._get_value({'n': 3.0, 'r': 2.0}) == 6.0


def test_initial_angular_velocity():
    """Test that a state of order 2 starts from its angle and its angular velocity, both recorded by path."""
    system = System(
        elements=[PendulumElement('bob', theta=0.1, values={"theta'": 0.5})],
        interactions=[SwingInteraction('swing', target='bob')],
    )
    trajectory = system.prepare().simulate(until=1, dt=0.1)
    assert trajectory.get_values('bob.theta')[0] == pytest.approx(0.1)
    assert trajectory.get_values("bob.theta'")[0] == pytest.approx(0.5)
    assert trajectory.get_values('bob.theta')[1] > 0.1


def test_first_order_unchanged():
    """Test that a state without a higher derivative compiles without companion states."""
    system = System(elements=[Element('a', values={'n': 1.0})], interactions=[_decay('a', 'n', -1.0)])
    trajectory = system.prepare().simulate(until=1, dt=0.5)
    assert trajectory.problem.states.names == ['a.n']


def test_newton_superposition_of_separate_laws():
    """Test that a spring and a damper from separate interactions add up to `m x'' = -k x - c x'`."""
    system = System(
        elements=[MassElement('mass')],
        interactions=[SpringInteraction('spring', target='mass'), DamperInteraction('damper', target='mass')],
        solver=PRECISE,
    )
    trajectory = system.prepare().simulate(until=10, dt=0.1)
    gamma = 0.4 / (2 * 2.0)
    omega = np.sqrt(8.0 / 2.0 - gamma**2)
    t = trajectory.times
    expected = np.exp(-gamma * t) * (np.cos(omega * t) + gamma / omega * np.sin(omega * t))
    assert np.allclose(trajectory.get_values('mass.x'), expected, atol=1e-7)


def test_order_mismatch_is_rejected():
    """Test that rate laws of a state on its first and second derivatives name the state and both orders."""
    system = System(
        elements=[MassElement('mass')],
        interactions=[SpringInteraction('spring', target='mass'), MismatchedInteraction('wrong', target='mass')],
    )
    with pytest.raises(ValidationError, match=r'mass\.x has rate laws of order 1 and of order 2'):
        system.prepare()


def test_velocity_verlet_runs_order_two_position():
    """Test that velocity Verlet recognises a position of order 2 and conserves the oscillator's energy."""
    system = System(
        elements=[MassElement('mass')],
        interactions=[SpringInteraction('spring', target='mass')],
        solver=VelocityVerletSolver(dt=0.001),
    )
    trajectory = system.prepare().simulate(until=10, dt=0.1)
    energy = 0.5 * 2.0 * trajectory.get_values("mass.x'") ** 2 + 0.5 * 8.0 * trajectory.get_values('mass.x') ** 2
    assert np.allclose(energy, energy[0], rtol=1e-4)


def test_undeclared_change_names_the_state_and_the_quantity():
    """Test that a rate law on a state tagged money raises a conservation error naming both."""
    system = System(elements=[LeakyElement('leaky')], interactions=[_decay('leaky', 'money', -0.1)])
    with pytest.raises(ConservationError, match=r"leaky\.money.*'money'"):
        system.prepare()
