"""Intermittency: bursts of a variable driven across a bifurcation by multiplicative noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import HeunSolver


class IntermittentVariableElement(Element):
    """Intermittent variable element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        epsilon: Annotated[object, ParameterSpec()] = 0.0,
        D: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.epsilon = epsilon
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the intermittent variable element."""
        return [
            RateLaw(s.x, s.epsilon * s.x - s.x**3),
            NoiseLaw(s.x, sqrt(2 * s.D) * s.x, convention='stratonovich'),
        ]


@register_model(
    field='mathematics/stochastic_processes',
    tags=('noise', 'sde', 'multiplicative noise', 'intermittency', 'bifurcation'),
    references=('Bertin 2021', 'Platt, Spiegel, and Tresser 1993', 'Heagy, Platt, and Hammel 1994'),
    defaults={'until': 500.0, 'dt': 0.1, 'solver': HeunSolver(dt=0.01)},
)
def on_off_intermittency(
    n_realizations: int = 20,
    *,
    epsilon: float = 0.0,
    D: float = 0.5,
    x: float = 1.0,
) -> System:
    """Return stochastic on-off intermittency, `dx/dt = (epsilon + xi) x - x^3` with Stratonovich white noise `xi`.

    The noise has the correlation `<xi(t) xi(t')> = 2 D delta(t - t')`. The variable `y = ln x` diffuses freely while
    `x` is small and is pushed back by `-exp(2 y)` once `x` grows, so near the bifurcation `epsilon = 0` the variable
    rests close to 0 for long laminar phases, whose durations are distributed as `tau^(-3/2)`, between bursts where
    `x` is of order 1. For `epsilon > 0` the power law is cut off beyond durations of order `D / epsilon^2`.

    Args:
        n_realizations:
            The number of independent realizations.

        epsilon:
            The control parameter, the distance from the bifurcation.

        D:
            The noise intensity.

        x:
            The initial value of every realization.

    Returns:
        system:
            The model, with the values `variables.x`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import on_off_intermittency
        >>> on_off_intermittency(5).elements[0].size
        5
    """
    variables = GroupElement(
        'variables',
        element=IntermittentVariableElement,
        size=n_realizations,
        values={'x': x, 'epsilon': epsilon, 'D': D},
    )
    return System(elements=[variables])
