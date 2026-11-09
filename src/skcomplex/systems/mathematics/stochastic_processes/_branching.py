"""Stochastic process systems of branching processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LifeCycleInteraction
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class IndividualView(AgentView):
    """View of an individual of a branching process.

    The view has the states and parameters of the individual as attributes.
    """

    generation: float
    m: float


class IndividualElement(Element):
    """Individual of a Galton-Watson process.

    The life cycle of the system replaces the individual by its offspring.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        generation:
            The initial generation.

        m:
            The mean number of offspring.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import IndividualElement
        >>> group = GroupElement('population', element=IndividualElement, size=10, values={'m': 1.5})
        >>> group.label, group.element.__name__, group.size
        ('population', 'IndividualElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'generation'}),
        *,
        generation: Annotated[float, StateSpec(domain=NN)] = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.generation = generation
        self.m = m

    @classmethod
    def _get_view(cls) -> type[IndividualView]:
        """Return the view of the agents.

        The view types their states and parameters.
        """
        return IndividualView


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

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import GaltonWatsonSystem
        >>> GaltonWatsonSystem(5, m=1.2).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.n_individuals = n_individuals
        self.m = m

    def _build_system(self) -> System:
        """Return a Galton-Watson branching process with Poisson offspring.

        Every step is one generation.
        """
        return System(
            self.label,
            elements=[
                GroupElement(
                    'population',
                    element=IndividualElement,
                    size=self.n_individuals,
                    values={'m': self.m},
                    variable=True,
                ),
            ],
            interactions=[
                LifeCycleInteraction(
                    'reproduction',
                    target='population',
                    birth=1.0,
                    offspring=lambda agent, system: int(system.rng.poisson(agent.m)),
                    values=lambda agent, system: {'generation': agent.generation + 1},
                    replace=True,
                ),
            ],
        )
