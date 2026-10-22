"""Stochastic process models: branching processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import AgentType, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView


class Individual(AgentType):
    """Individual of a Galton-Watson process, replaced in the next generation by a Poisson number of offspring.

    Args:
        label:
            The label of the agent type or group.

        location:
            Unused, since individuals have no space.

        generation:
            The initial generation.

        m:
            The mean number of offspring.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import Individual
        >>> Individual.group('population', 10, m=1.5)
        Group('population', element=Individual, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'generation'})
    generation: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    m: float = parameter(1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the reproduction rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.reproduce)]

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
        >>> galton_watson(5, m=1.2).elements[0].size
        5
    """
    return System(components=[Individual.group('population', n_individuals, m=m)])
