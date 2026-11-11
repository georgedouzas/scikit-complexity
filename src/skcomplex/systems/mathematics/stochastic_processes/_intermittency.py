"""Intermittency.

Multiplicative noise drives a variable across a bifurcation in bursts.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import HeunSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class IntermittentVariableElement(Element):
    """Positive variable driven across the bifurcation at `epsilon = 0`.

    The drive is a multiplicative noise of intensity `D`.

    Args:
        label:
            The label of the variable or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        epsilon:
            The control parameter, the distance from the bifurcation.

        D:
            The noise intensity.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import IntermittentVariableElement
        >>> IntermittentVariableElement('variable', D=0.2).D
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        epsilon: Annotated[float, ParameterSpec()] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.epsilon = epsilon
        self.D = D


class OnOffIntermittencySystem(BaseSystem):
    """Stochastic on-off intermittency, `dx/dt = (epsilon + xi) x - x^3`.

    The noise `xi` is Stratonovich white noise.

    The noise has the correlation `<xi(t) xi(t')> = 2 D delta(t - t')`. The variable `y = ln x` diffuses freely while
    `x` is small and is pushed back by `-exp(2 y)` once `x` grows, so near the bifurcation `epsilon = 0` the variable
    rests close to 0 for long laminar phases, whose durations are distributed as `tau^(-3/2)`, between bursts where `x`
    is of order 1. For `epsilon > 0` the power law is cut off beyond durations of order `D / epsilon^2`. The system has
    the values `variables.x`.

    Args:
        n_realizations:
            The number of independent realizations.

        epsilon:
            The control parameter, the distance from the bifurcation.

        D:
            The noise intensity.

        x:
            The initial value of every realization.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import OnOffIntermittencySystem
        >>> OnOffIntermittencySystem(5).prepare().system_.elements[0].size
        5
    """

    field = 'mathematics/stochastic_processes'
    topics = ('noise', 'sde', 'multiplicative noise', 'intermittency', 'bifurcation')
    references = ('Bertin 2021', 'Platt, Spiegel, and Tresser 1993', 'Heagy, Platt, and Hammel 1994')
    until = 500.0
    dt = 0.1
    solver = HeunSolver(dt=0.01)

    def __init__(
        self,
        n_realizations: int = 20,
        *,
        epsilon: float = 0.0,
        D: float = 0.5,
        x: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_realizations = n_realizations
        self.epsilon = epsilon
        self.D = D
        self.x = x

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        variables = GroupElement(
            'variables',
            element=IntermittentVariableElement,
            size=self.n_realizations,
            values={'x': self.x, 'epsilon': self.epsilon, 'D': self.D},
        )
        drift = RateInteraction(
            'drift',
            target='variables',
            state='x',
            rate=lambda s: s.target.epsilon * s.target.x - s.target.x**3,
            noise=lambda s: sqrt(2 * s.target.D) * s.target.x,
            convention='stratonovich',
        )
        return CompositeSystem(self.label, elements=[variables], interactions=[drift])
