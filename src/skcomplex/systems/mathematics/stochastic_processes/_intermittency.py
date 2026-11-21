"""Intermittency.

Multiplicative noise drives a variable across a bifurcation in bursts.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, HeunSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class IntermittentVariableElement(Element):
    """Positive variables `x`, one per realization.

    Args:
        label:
            The label of the variables.

        x:
            The initial value, positive, one per realization or shared.

        size:
            The number of realizations, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import IntermittentVariableElement
        >>> IntermittentVariableElement('variable', x=0.2).x
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.open(0, oo)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x


class IntermittencyInteraction(Interaction):
    """The drive of a positive variable across the bifurcation at `epsilon = 0` by a multiplicative noise.

    The variable follows `dx = (epsilon x - x³) dt + sqrt(2 D) x dW` in the Stratonovich convention.

    Args:
        label:
            The label of the interaction.

        target:
            The variables the drive moves.

        epsilon:
            The control parameter, the distance from the bifurcation.

        D:
            The noise intensity, positive.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import IntermittencyInteraction
        >>> IntermittencyInteraction('drift', target='variables', D=0.2).D
        0.2
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(self, label: str, *, target: Role = 'variables', epsilon: float = 0.0, D: float = 0.5) -> None:
        super().__init__(label)
        self.target = target
        self.epsilon = epsilon
        self.D = D

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the drift and the multiplicative noise of the variable as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x = context.target.x
        self.laws_: list[BaseComponent] = [
            RateLaw(x, context.epsilon * x - x**3),
            NoiseLaw(x, sqrt(2 * context.D) * x, convention='stratonovich'),
        ]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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

    def __init__(
        self,
        n_realizations: int = 20,
        *,
        epsilon: float = 0.0,
        D: float = 0.5,
        x: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_realizations = n_realizations
        self.epsilon = epsilon
        self.D = D
        self.x = x

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the variables and their drive."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[IntermittentVariableElement('variables', x=self.x, size=self.n_realizations)],
            interactions=[IntermittencyInteraction('drift', target='variables', epsilon=self.epsilon, D=self.D)],
            solver=HeunSolver(dt=0.01),
        )
