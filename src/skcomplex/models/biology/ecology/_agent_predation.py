"""Ecology models: agent-based predation of prey by predators on a grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.base import Domain, VisualSpec, parameter
from ....core.model import AgentType, GridSpace, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

PREY = 'prey'


def _step(agent: AgentView, model: ModelView) -> None:
    """Move an animal to a random neighbouring cell."""
    cells = agent.neighbor_cells()
    agent.move_to(cells[int(model.rng.integers(len(cells)))])


class Prey(AgentType):
    """Prey that walks at random and gives birth in cells it does not share with other prey.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        birth:
            The probability of giving birth at every step.

    Examples:
        >>> from skcomplex.models.biology.ecology import Prey
        >>> Prey.group('rabbits', 200).size
        200
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'radius': 0.3})
    birth: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.live)]

    def live(self, agent: AgentView, model: ModelView) -> None:
        """Walk, then give birth when no other prey shares the new cell.

        Args:
            agent:
                The prey.

            model:
                The model.
        """
        _step(agent, model)
        if not agent.agents_here(PREY) and model.rng.random() < agent.birth:
            agent.create()


class Predator(AgentType):
    """Predator that walks at random, eats prey of the group `prey` in its cell, breeds after a meal, and dies.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        conversion:
            The probability of breeding after eating a prey.

        death:
            The probability of dying at every step.

    Examples:
        >>> from skcomplex.models.biology.ecology import Predator
        >>> Predator.group('foxes', 50).size
        50
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'radius': 0.4})
    conversion: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    death: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the hunting rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.hunt)]

    def hunt(self, agent: AgentView, model: ModelView) -> None:
        """Walk, eat one prey of the new cell and breed after it, then die at random.

        Args:
            agent:
                The predator.

            model:
                The model.
        """
        _step(agent, model)
        meals = agent.agents_here(PREY)
        if meals:
            meals[int(model.rng.integers(len(meals)))].remove()
            if model.rng.random() < agent.conversion:
                agent.create()
        if model.rng.random() < agent.death:
            agent.remove()


@register_model(
    field='biology/ecology',
    tags=('predator-prey', 'agents', 'grid', 'oscillation'),
    references=('Argun et al. 2021', 'Lotka 1925', 'Volterra 1926'),
    defaults={'until': 200},
)
def agent_based_lotka_volterra(
    width: int = 30,
    height: int = 30,
    *,
    prey: int = 300,
    predators: int = 60,
    birth: float = 0.2,
    conversion: float = 0.5,
    death: float = 0.1,
    random_state: int | np.random.Generator | None = None,
) -> System:
    """Return prey and predators that walk on a grid, whose numbers cycle like the Lotka-Volterra equations.

    Prey give birth only in cells they do not share with other prey, which limits their density, and predators eat
    prey they meet and breed after a meal. Demographic noise makes the cycles irregular and can end them by extinction.

    Args:
        width:
            The number of columns of the torus.

        height:
            The number of rows of the torus.

        prey:
            The initial number of prey.

        predators:
            The initial number of predators.

        birth:
            The probability that a prey gives birth at every step.

        conversion:
            The probability that a predator breeds after eating a prey.

        death:
            The probability that a predator dies at every step.

        random_state:
            The seed or generator of the initial cells.

    Returns:
        system:
            The model, with the groups `prey` and `predators`.

    Examples:
        >>> from skcomplex.models.biology.ecology import agent_based_lotka_volterra
        >>> [group.size for group in agent_based_lotka_volterra(10, 10, prey=50, predators=5).elements]
        [50, 5]
    """
    rng = np.random.default_rng(random_state)
    cells = rng.integers(0, [width, height], size=(prey + predators, 2))
    locations = [tuple(int(item) for item in cell) for cell in cells]
    return System(
        elements=[
            Prey.group('prey', prey, location=locations[:prey], birth=birth),
            Predator.group(
                'predators',
                predators,
                location=locations[prey:],
                conversion=conversion,
                death=death,
            ),
        ],
        space=GridSpace(width, height, torus=True),
    )
