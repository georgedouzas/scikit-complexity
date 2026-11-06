"""Stochastic process systems: branching processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LifeCycleInteraction
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ._individual_view import IndividualView


class IndividualElement(Element):
    """Individual of a Galton-Watson process, which the life cycle of the system replaces by its offspring.

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
        """Return the view of the agents, which types their states and parameters."""
        return IndividualView


@register_system(
    field='mathematics/stochastic_processes',
    tags=('branching process', 'agents', 'extinction', 'criticality'),
    references=('Watson and Galton 1875', 'Gros 2015', 'Jensen 2023'),
    defaults={'until': 20},
)
def galton_watson(n_individuals: int = 10, *, m: float = 1.0) -> System:
    """Return a Galton-Watson branching process with Poisson offspring, one generation per step.

    The mean size of generation `n` is `n_individuals m^n`. The process dies out with certainty for `m <= 1`, and
    survives with a positive probability for `m > 1`.

    Args:
        n_individuals:
            The size of the first generation.

        m:
            The mean number of offspring of an individual.

    Returns:
        system:
            The system, with the group `population`, whose count is the size of the current generation.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import galton_watson
        >>> galton_watson(5, m=1.2).elements[0].prepare().size_
        5
    """
    return System(
        elements=[
            GroupElement('population', element=IndividualElement, size=n_individuals, values={'m': m}, variable=True),
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
