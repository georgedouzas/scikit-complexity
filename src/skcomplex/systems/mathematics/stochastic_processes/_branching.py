"""Stochastic process systems of branching processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role


class IndividualView(AgentView):
    """View of an individual of a branching process, with its values as attributes."""

    generation: float


class IndividualElement(Element):
    """Individuals of a branching process, each with its generation `generation`.

    Args:
        label:
            The label of the individuals.

        generation:
            The initial generation, one per individual or shared.

        size:
            The number of individuals, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import IndividualElement
        >>> IndividualElement('population', size=10).size
        10
    """

    visual = VisualSpec('agents', {'color_by': 'generation'})

    def __init__(
        self,
        label: str,
        *,
        generation: Annotated[object, NN] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.generation = generation

    @classmethod
    def _get_view(cls) -> type[IndividualView]:
        """Return the view of the individuals, which types their values."""
        return IndividualView


class BranchingInteraction(Interaction):
    """Replacement of every individual by a Poisson number of offspring of mean `m` at every step.

    The offspring belong to the next generation and act from the next step on, so every step is one generation.

    Args:
        label:
            The label of the interaction.

        target:
            The individuals that reproduce and die.

        m:
            The mean number of offspring of an individual.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import BranchingInteraction
        >>> BranchingInteraction('reproduction', m=1.5).m
        1.5
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'population', m: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.m = m

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the reproduction of every individual as the rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self._replace)]

    def _replace(self, agent: IndividualView, system: SystemView) -> None:
        """Create the offspring of an individual and remove it."""
        for _ in range(int(system.rng.poisson(self.m))):
            agent.create(generation=agent.generation + 1)
        agent.remove()


class GaltonWatsonSystem(BaseSystem):
    """A Galton-Watson branching process with Poisson offspring.

    Every step is one generation.

    The mean size of generation `n` is `n_individuals m^n`. The process dies out with certainty for `m <= 1`, and
    survives with a positive probability for `m > 1`. The system has the group `population`, whose count is the size of
    the current generation.

    Args:
        n_individuals:
            The size of the first generation.

        m:
            The mean number of offspring of an individual.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import GaltonWatsonSystem
        >>> GaltonWatsonSystem(5, m=1.2).prepare().system_.elements[0].size
        5
    """

    field = 'mathematics/stochastic_processes'
    topics = ('branching process', 'agents', 'extinction', 'criticality')
    references = ('Watson and Galton 1875', 'Gros 2015', 'Jensen 2023')
    until = 20

    def __init__(
        self,
        n_individuals: int = 10,
        *,
        m: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_individuals = n_individuals
        self.m = m

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the first generation and its reproduction."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[IndividualElement('population', size=self.n_individuals)],
            interactions=[BranchingInteraction('reproduction', target='population', m=self.m)],
        )
