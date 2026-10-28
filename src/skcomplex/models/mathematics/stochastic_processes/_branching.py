"""Stochastic process models: branching processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ..._catalog import register_model


class IndividualElement(Element):
    """Individual of a Galton-Watson process, replaced in the next generation by a Poisson number of offspring.

    Args:
        label:
            The label of the agent or group.

        generation:
            The initial generation.

        m:
            The mean number of offspring.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import IndividualElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reproduction rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.reproduce)]

    def reproduce(self, agent: AgentView, model: ModelView) -> None:
        """Create the offspring of the individual and remove it.

        Args:
            agent:
                The individual.

            model:
                The model.
        """
        for _ in range(int(model.rng.poisson(agent.m))):
            agent.create(generation=agent.generation + 1)
        agent.remove()


@register_model(
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
            The model, with the group `population`, whose count is the size of the current generation.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import galton_watson
        >>> galton_watson(5, m=1.2).elements[0].check().prepare().size_
        5
    """
    return System(
        elements=[
            GroupElement('population', element=IndividualElement, size=n_individuals, values={'m': m}, variable=True),
        ],
    )
