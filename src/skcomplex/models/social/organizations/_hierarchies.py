"""Hierarchy models: dominance that emerges from fights between wandering agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.space import LatticeSpace
from ....core.system import System


class Contender(Element):
    """Agent of the Bonabeau model, who wanders on a grid and fights the agents it runs into.

    Every step an agent's strength relaxes by the factor `1 - relaxation`, and the agent tries to step into a random
    neighbouring cell. An attacker beats the occupant with probability `1 / (1 + exp(eta * (h_occupant - h_attacker)))`,
    the winner gains 1 and the loser loses 1, and a winning attacker swaps places with the occupant. Agents differ only
    by their strength, so the swap exchanges their strengths.

    Args:
        label:
            The label of the agent or group.

        strength:
            The initial strength `h`.

        eta:
            The sensitivity of the outcome of a fight to the strength difference.

        relaxation:
            The fraction of its strength an agent loses every step.

    Examples:
        >>> from skcomplex.models.social.organizations import Contender
        >>> Contender.group('contenders', 10)
        Group('contenders', element=Contender, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'strength'})
    strength: float = state(default=0.0)
    eta: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    relaxation: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the wander and fight rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.wander)]

    def _fight(self, agent: AgentView, occupant: AgentView, model: ModelView) -> None:
        """Fight the occupant of a cell and take its place on a win."""
        wins = model.rng.random() < 1 / (1 + np.exp(np.clip(agent.eta * (occupant.strength - agent.strength), -50, 50)))
        if wins:
            agent.strength, occupant.strength = occupant.strength - 1, agent.strength + 1
        else:
            agent.strength, occupant.strength = agent.strength - 1, occupant.strength + 1

    def wander(self, agent: AgentView, model: ModelView) -> None:
        """Relax the strength, then step into a random neighbouring cell or fight its occupant.

        Args:
            agent:
                The contender.

            model:
                The model.
        """
        agent.strength *= 1 - agent.relaxation
        cells = agent.neighbor_cells()
        cell = cells[int(model.rng.integers(len(cells)))]
        occupants = [
            other
            for other in agent.neighbors()
            if tuple(np.ravel(np.asarray(other.location))) == tuple(np.ravel(np.asarray(cell)))
        ]
        if occupants:
            self._fight(agent, occupants[0], model)
            return
        agent.move_to(cell)


@register_model(
    field='social/organizations',
    tags=('hierarchy', 'agents', 'grid', 'phase transition'),
    references=('Bonabeau, Theraulaz, and Deneubourg 1995', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 50},
)
def bonabeau_hierarchies(
    width: int = 20,
    height: int = 20,
    *,
    density: float = 0.5,
    eta: float = 1.0,
    relaxation: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Bonabeau model, where a hierarchy of strengths emerges above a critical density of agents.

    Args:
        width:
            The number of columns of the torus.

        height:
            The number of rows of the torus.

        density:
            The share of cells with an agent.

        eta:
            The sensitivity of the outcome of a fight to the strength difference.

        relaxation:
            The fraction of its strength an agent loses every step.

        random_state:
            The seed or generator of the initial cells.

    Returns:
        system:
            The model, with the group `contenders`.

    Examples:
        >>> from skcomplex.models.social.organizations import bonabeau_hierarchies
        >>> bonabeau_hierarchies(10, 10, density=0.2).elements[0].size
        20
    """
    size = int(width * height * density)
    cells = np.random.default_rng(random_state).permutation(width * height)[:size]
    contenders = Contender.group(
        'contenders',
        size,
        location=[(int(cell % width), int(cell // width)) for cell in cells],
        eta=eta,
        relaxation=relaxation,
    )
    return System(components=[contenders], space=LatticeSpace((width, height), boundary='periodic', capacity=1))
