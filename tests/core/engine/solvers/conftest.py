"""Fixtures of the solver tests: optimal control problems with known solutions."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import BaseLaw, RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec

HORIZON = 1.0

type Control = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
type Cost = Callable[[SimpleNamespace], object]


class IntegratorElement(Element):
    """Scalar state driven by its control, with a quadratic running cost."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(-10, 10))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.u = u

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the state."""
        return [RateLaw(s.x, s.u)]


class FisheryElement(Element):
    """Logistic resource harvested with bounded effort for a discounted profit."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        effort: Annotated[float, ParameterSpec(adaptive=True, bounds=(0, 1))] = 0.0,
        cost: Annotated[float, ParameterSpec()] = 0.2,
        discount: Annotated[float, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.effort = effort
        self.cost = cost
        self.discount = discount

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the logistic regrowth less the harvest."""
        return [RateLaw(s.x, s.x * (1 - s.x) - s.effort * s.x)]


def _quadratic_cost(s: SimpleNamespace) -> object:
    """Return the running cost `(x^2 + u^2) / 2` of the plant."""
    return (s.plant.x**2 + s.plant.u**2) / 2


def _profit(s: SimpleNamespace) -> object:
    """Return the discounted profit of the fishery."""
    fishery = s.fishery
    return exp(-fishery.discount * s.t) * (fishery.x - fishery.cost) * fishery.effort


def _riccati_control(times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the optimal control of the scalar linear-quadratic problem, `-sinh(1 - t) / cosh(1)`."""
    return -np.sinh(HORIZON - times) / np.cosh(HORIZON)


@pytest.fixture
def riccati_control() -> Control:
    """Return the analytic optimal control of the scalar linear-quadratic problem, a function of time."""
    return _riccati_control


@pytest.fixture
def linear_quadratic() -> CompositeSystem:
    """Return the scalar linear-quadratic problem `x' = u`, `x(0) = 1`, with cost `∫ (x^2 + u^2) / 2 dt`."""
    return CompositeSystem(elements=[IntegratorElement('plant')])


@pytest.fixture
def quadratic_cost() -> Cost:
    """Return the running cost of the scalar linear-quadratic problem, a function of the namespace."""
    return _quadratic_cost


@pytest.fixture
def profit() -> Cost:
    """Return the discounted profit of the harvest to maximise, a function of the namespace."""
    return _profit


@pytest.fixture
def harvesting() -> CompositeSystem:
    """Return the discounted harvest of a logistic resource with bounded effort."""
    return CompositeSystem(elements=[FisheryElement('fishery')])
