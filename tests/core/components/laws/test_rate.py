"""Test the `_rate` module of `skcomplex.core.components.laws`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, pi, sin, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import VelocityVerletSolver
from skcomplex.core.errors import ConservationError, ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

from .conftest import PRECISE, MassElement


class PendulumElement(Element):
    """Pendulum whose angle wraps around a turn."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(period=2 * pi, order=2)] = 0.1,
        theta_d1: float = 0.0,
        g: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        l: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the rate of the angular velocity."""
        return [RateLaw(s.theta, -(s.g / s.l) * sin(s.theta))]


class StockElement(Element):
    """Stock that decays."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n

    def _get_laws(self, s):
        """Return the decay."""
        return [RateLaw(s.n, -s.n)]


class LeakyElement(Element):
    """Money that a rate law changes outside any flow."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        money: Annotated[object, StateSpec(quantity='money')] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.money = money

    def _get_laws(self, s):
        """Return the leak of money."""
        return [RateLaw(s.money, -0.1 * s.money)]


class SpringInteraction(Interaction):
    """Spring on the mass of its target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        k: Annotated[object, ParameterSpec()] = 8.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k

    def _get_laws(self, s):
        """Return the force of the spring."""
        return [RateLaw(s.target.x, -s.k * s.target.x / s.target.m)]


class DamperInteraction(Interaction):
    """Damper on the mass of its target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        c: Annotated[object, ParameterSpec()] = 0.4,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.c = c

    def _get_laws(self, s):
        """Return the force of the damper."""
        return [RateLaw(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)]


class MismatchedInteraction(Interaction):
    """Rate that asserts the first derivative on a state of order 2."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the mismatched rate."""
        return [RateLaw(s.target.x, -s.target.x, order=1)]


def test_rate_runs_into_its_value():
    """Test that a rate law targets its state and runs into its value at given values of the symbols."""
    n, r = var('n r')
    law = RateLaw(n, r * n, order=1).prepare()
    assert (law.targets_, law.order) == ((n,), 1)
    assert law.run({'n': 3.0, 'r': 2.0}).value_ == 6.0


def test_initial_angular_velocity():
    """Test that a state of order 2 starts from its angle and its angular velocity, both recorded by path."""
    system = System(elements=[PendulumElement('bob', theta=0.1, theta_d1=0.5)])
    trajectory = Simulation(system, dt=0.1).prepare().run(until=1).trajectory_
    assert trajectory.values('bob.theta')[0] == pytest.approx(0.1)
    assert trajectory.values('bob.theta_d1')[0] == pytest.approx(0.5)
    assert trajectory.values('bob.theta')[1] > 0.1


def test_first_order_unchanged():
    """Test that a state without an order compiles without companion states."""
    simulation = Simulation(System(elements=[StockElement('a')]), dt=0.5).prepare().run(until=1)
    assert simulation.problem_.states.names == ['a.n']


def test_newton_superposition_of_separate_laws():
    """Test that a spring and a damper from separate interactions add up to `m x'' = -k x - c x'`."""
    system = System(
        elements=[MassElement('mass')],
        interactions=[SpringInteraction('spring', target='mass'), DamperInteraction('damper', target='mass')],
    )
    trajectory = Simulation(system, solver=PRECISE, dt=0.1).prepare().run(until=10).trajectory_
    gamma = 0.4 / (2 * 2.0)
    omega = np.sqrt(8.0 / 2.0 - gamma**2)
    t = trajectory.times
    expected = np.exp(-gamma * t) * (np.cos(omega * t) + gamma / omega * np.sin(omega * t))
    assert np.allclose(trajectory.values('mass.x'), expected, atol=1e-7)


def test_order_mismatch_is_rejected():
    """Test that a law asserting order 1 on a state of order 2 names the law, the state, and both orders."""
    system = System(elements=[MassElement('mass')], interactions=[MismatchedInteraction('wrong', target='mass')])
    with pytest.raises(ValidationError, match=r'mass\.x gives the derivative of order 1, but the state has order 2'):
        Simulation(system).prepare().run(until=1)


def test_velocity_verlet_runs_order_two_position():
    """Test that velocity Verlet recognises a position of order 2 and conserves the oscillator's energy."""
    system = System(elements=[MassElement('mass')], interactions=[SpringInteraction('spring', target='mass')])
    trajectory = Simulation(system, solver=VelocityVerletSolver(dt=0.001), dt=0.1).prepare().run(until=10).trajectory_
    energy = 0.5 * 2.0 * trajectory.values('mass.x_d1') ** 2 + 0.5 * 8.0 * trajectory.values('mass.x') ** 2
    assert np.allclose(energy, energy[0], rtol=1e-4)


def test_undeclared_change_names_the_state_and_the_quantity():
    """Test that a rate law on a state tagged money raises a conservation error naming both."""
    with pytest.raises(ConservationError, match=r"leaky\.money.*'money'"):
        compile_system(System(elements=[LeakyElement('leaky')]))
