"""Test the `_update` module of `skcomplex.core.laws`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.elements import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw, UpdateLaw
from skcomplex.core.laws.utils import lag, where
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.specs import ParameterSpec, RoleSpec, StateSpec, TopologyRequestSpec, VisualSpec


class EconomyElement(Element):
    """Prices that grow every step, with an adaptive policy rate."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        price: Annotated[float, StateSpec()] = 1.0,
        rate: Annotated[float, ParameterSpec(adaptive=True)] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price
        self.rate = rate

    def _get_laws(self, s):
        """Return the growth of prices."""
        return [UpdateLaw(s.price, s.price * 1.03)]


class PolicyInteraction(Interaction):
    """Rule that raises the policy rate whenever prices exceed a target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the update of the adaptive rate."""
        return [UpdateLaw(s.target.rate, where(s.target.price > 1.05, s.target.rate + 0.01, s.target.rate))]


class FibonacciElement(Element):
    """Fibonacci sequence, a state with one lag."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(lags=1, initial=(1.0,))] = 1.0,
        x_lag1: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_lag1 = x_lag1

    def _get_laws(self, s):
        """Return the recurrence."""
        return [UpdateLaw(s.x, s.x + lag(s.x, 1))]


class IncomeElement(Element):
    """Samuelson's multiplier-accelerator model of income."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Y: Annotated[object, StateSpec(lags=1, initial=(100.0,))] = 110.0,
        Y_lag1: float = 100.0,
        c: Annotated[object, ParameterSpec()] = 0.8,
        v: Annotated[object, ParameterSpec()] = 1.5,
        G: Annotated[object, ParameterSpec()] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Y = Y
        self.Y_lag1 = Y_lag1
        self.c = c
        self.v = v
        self.G = G

    def _get_laws(self, s):
        """Return the recurrence of income."""
        return [UpdateLaw(s.Y, s.c * (1 + s.v) * s.Y - s.c * s.v * lag(s.Y, 1) + s.G)]


class PumpElement(Element):
    """Tank filled by a bounded control input."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        u: Annotated[object, ParameterSpec(adaptive=True, bounds=(0, 1))] = 0.0,
        weight: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.u = u
        self.weight = weight

    def _get_laws(self, s):
        """Return the filling of the tank."""
        return [RateLaw(s.x, -s.x + s.u)]


def test_update_runs_into_its_value_and_targets_a_topology_request():
    """Test that an update runs into its new value, and targets a topology request that it fires with a probability."""
    x = var('x')
    assert UpdateLaw(x, 2 * x).check().prepare().run({'x': 1.5}).value_ == 3.0
    request = TopologyRequestSpec('rewire', keep='target')
    law = UpdateLaw(request, 0.1).check().prepare()
    assert law.targets_ == (request,)


def test_adaptive_parameter_changes_only_when_the_policy_fires():
    """Test that the adaptive rate changes only at steps where prices exceed the target, and starts from its value."""
    system = System(
        elements=[EconomyElement('economy', rate=0.03)],
        interactions=[PolicyInteraction('bank', target='economy')],
    )
    trajectory = Simulation(system).check().prepare().run(until=4).trajectory_
    rates, prices = trajectory.values('economy.rate'), trajectory.values('economy.price')
    assert rates[0] == pytest.approx(0.03)
    np.testing.assert_array_equal(np.diff(rates) > 0, prices[:-1] > 1.05)


def test_fibonacci_recurrence():
    """Test that `x_{t+1} = x_t + x_{t-1}` from 1 and 1 gives the Fibonacci sequence."""
    trajectory = Simulation(System(elements=[FibonacciElement('f')])).check().prepare().run(until=8).trajectory_
    assert trajectory.values('f.x').tolist() == [1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert trajectory.values('f.x_lag1').tolist()[:3] == [1, 1, 2]


def test_multiplier_accelerator_oscillates_with_growing_amplitude():
    """Test that Samuelson's model with c v > 1 oscillates around G / (1 - c) with growing amplitude."""
    income = (
        Simulation(System(elements=[IncomeElement('economy')]))
        .check()
        .prepare()
        .run(until=40)
        .trajectory_.values('economy.Y')
    )
    deviation = income - 20.0 / (1 - 0.8)
    assert np.any(deviation[1:] * deviation[:-1] < 0)
    assert np.abs(deviation[-10:]).max() > 10 * np.abs(deviation[:10]).max()


def test_adaptive_parameters_compile_with_their_bounds():
    """Test that compiling keeps the bounded adaptive parameters, the controls, and leaves the form an ODE."""
    problem = compile_system(System(elements=[PumpElement('pump')]))
    assert problem.form.name == 'ODE'
    assert set(problem.adaptive_parameters) == {'pump.u'}
    assert problem.adaptive_parameters['pump.u'].bounds == (0.0, 1.0)
