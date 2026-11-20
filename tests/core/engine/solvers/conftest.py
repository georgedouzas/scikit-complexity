"""Fixtures of the solver tests: optimal control problems with known solutions."""

from collections.abc import Callable
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw, RateLaw
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.systems import System

HORIZON = 1.0

type Control = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
type Plant = Callable[..., System]


class IntegratorElement(Element):
    """Scalar value driven by its control, which lies between -10 and 10."""

    def __init__(self, label: str, *, x: float = 1.0, u: Annotated[float, RealSet.closed(-10, 10)] = 0.0) -> None:
        super().__init__(label)
        self.x = x
        self.u = u


def _riccati_control(times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the optimal control of the scalar linear-quadratic problem, `-sinh(1 - t) / cosh(1)`."""
    return -np.sinh(HORIZON - times) / np.cosh(HORIZON)


def _linear_quadratic(solver: BaseSolver | str = 'auto', *, terminal: float = 0.0, cost: bool = True) -> System:
    """Return the prepared problem `x' = u`, `x(0) = 1`, with the cost `∫ (x^2 + u^2) / 2 dt + terminal x(T)^2`."""

    def regulate(s):
        plant = s.target
        laws = [RateLaw(plant.x.diff(s.t), plant.u)]
        if cost:
            running = (plant.x**2 + plant.u**2) / 2
            laws.append(ObjectiveLaw(running, controls=[plant.u], terminal=terminal * plant.x**2))
        return laws

    regulator = Interaction('regulator', roles={'target': 'plant'}, laws=regulate)
    return System(elements=[IntegratorElement('plant')], interactions=[regulator], solver=solver).prepare()


@pytest.fixture
def riccati_control() -> Control:
    """Return the analytic optimal control of the scalar linear-quadratic problem, a function of time."""
    return _riccati_control


@pytest.fixture
def linear_quadratic() -> Plant:
    """Return a function of the solver that gives the prepared scalar linear-quadratic problem."""
    return _linear_quadratic
