"""Fixtures of the solver tests: optimal control problems with known solutions."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import exp

from skcomplex.core.base import BaseLaw, NumericDomain, parameter, state
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import Objective, RateLaw, Symbols
from skcomplex.core.system import System

HORIZON = 1.0

type Control = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


class IntegratorComponent(ElementComponent):
    """Scalar state driven by its control, with a quadratic running cost."""

    x: float = state(1.0)
    u: float = parameter(0.0, adaptive=True, bounds=(-10, 10))

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the rate of the state and the cost `(x^2 + u^2) / 2`."""
        return [RateLaw(s.x, s.u), Objective((s.x**2 + s.u**2) / 2)]


class FisheryComponent(ElementComponent):
    """Logistic resource harvested with bounded effort for a discounted profit."""

    x: float = state(1.0, domain=NumericDomain.NONNEGATIVE)
    effort: float = parameter(0.0, adaptive=True, bounds=(0, 1))
    cost: float = parameter(0.2)
    discount: float = parameter(0.1)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the logistic regrowth less the harvest, and the discounted profit to maximise."""
        profit = exp(-s.discount * s.t) * (s.x - s.cost) * s.effort
        return [RateLaw(s.x, s.x * (1 - s.x) - s.effort * s.x), Objective(profit, sense='max')]


def _riccati_control(times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the optimal control of the scalar linear-quadratic problem, `-sinh(1 - t) / cosh(1)`."""
    return -np.sinh(HORIZON - times) / np.cosh(HORIZON)


@pytest.fixture
def riccati_control() -> Control:
    """Return the analytic optimal control of the scalar linear-quadratic problem, a function of time."""
    return _riccati_control


@pytest.fixture
def linear_quadratic() -> System:
    """Return the scalar linear-quadratic problem `x' = u`, `x(0) = 1`, with cost `∫ (x^2 + u^2) / 2 dt`."""
    return System(components=[IntegratorComponent('plant')])


@pytest.fixture
def harvesting() -> System:
    """Return the discounted harvest of a logistic resource with bounded effort."""
    return System(components=[FisheryComponent('fishery')])
