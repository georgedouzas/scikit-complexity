"""Hierarchy systems: dominance that emerges from fights between wandering agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, runtime_checkable

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role


@runtime_checkable
class ContenderView(AgentViewProtocol, Protocol):
    """View of an agent of the Bonabeau hierarchy system.

    Its states and parameters are attributes.
    """

    strength: float
    eta: float
    relaxation: float


class ContenderElement(Element):
    """Agent of the Bonabeau system.

    It wanders on a grid and fights the agents it runs into. Every step an agent's strength relaxes by the factor
    `1 - relaxation`, and the agent tries to step into a random neighbouring cell. An attacker beats the occupant with
    probability `1 / (1 + exp(eta * (h_occupant - h_attacker)))`, the winner gains 1 and the loser loses 1, and a
    winning attacker swaps places with the occupant. Agents differ only by their strength, so the swap exchanges their
    strengths.

    Args:
        label:
            The label of the agent or group.

        strength:
            The initial strength `h`.

        eta:
            The sensitivity of the outcome of a fight to the strength difference.

        relaxation:
            The fraction of its strength an agent loses every step.

        x:
            The column of the cell of the contender on a lattice, one per contender or shared.

        y:
            The row of the cell of the contender on a lattice, one per contender or shared.

        size:
            The number of members, or `None` for one.

        space:
            The lattice the contenders live on, whose axes `x` and `y` make `x` and `y` their cell, or `None`.

    Examples:
        >>> from skcomplex.systems.social.organizations import ContenderElement
        >>> group = ContenderElement('contenders', size=10)
        >>> group.label, type(group).__name__, group.size
        ('contenders', 'ContenderElement', 10)
    """

    visual = VisualSpec('agents', {'color_by': 'strength'})

    def __init__(
        self,
        label: str,
        *,
        strength: object = 0.0,
        eta: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        relaxation: Annotated[object, RealSet.closed_open(0, oo)] = 0.1,
        x: object = 0,
        y: object = 0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.strength = strength
        self.eta = eta
        self.relaxation = relaxation
        self.x = x
        self.y = y


class WanderInteraction(Interaction):
    """Wander and fight of the contenders of the Bonabeau system.

    Every step every contender relaxes its strength by the fraction `relaxation`, then steps into a random neighbouring
    cell, and when the cell is occupied it fights the occupant, winning with the probability
    `1 / (1 + exp(eta (s_occupant - s_agent)))`. The winner gains one unit of strength and takes the cell, and the loser
    loses one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of contenders.

    Examples:
        >>> from skcomplex.systems.social.organizations import WanderInteraction
        >>> WanderInteraction('wander', target='contenders').target
        'contenders'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the wander and fight of every contender as the rule.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self.wander)]

    def _fight(self, agent: ContenderView, occupant: ContenderView, system: SystemView) -> None:
        """Fight the occupant of a cell and take its place on a win."""
        wins = system.rng.random() < 1 / (
            1 + np.exp(np.clip(agent.eta * (occupant.strength - agent.strength), -50, 50))
        )
        if wins:
            agent.strength, occupant.strength = occupant.strength - 1, agent.strength + 1
        else:
            agent.strength, occupant.strength = agent.strength - 1, occupant.strength + 1

    def wander(self, agent: ContenderView, system: SystemView) -> None:
        """Relax the strength, then step into a random neighbouring cell.

        An occupied cell makes the agent fight its occupant.

        Args:
            agent:
                The contender.

            system:
                The system.
        """
        agent.strength *= 1 - agent.relaxation
        cells = agent.get_neighbor_cells()
        cell = cells[int(system.rng.integers(len(cells)))]
        occupants = [
            other
            for other in agent.get_neighbors()
            if isinstance(other, ContenderView)
            and tuple(np.ravel(np.asarray(other.location))) == tuple(np.ravel(np.asarray(cell)))
        ]
        if occupants:
            self._fight(agent, occupants[0], system)
            return
        agent.move_to(cell)


class BonabeauHierarchiesSystem(BaseSystem):
    """The Bonabeau system.

    A hierarchy of strengths emerges above a critical density of agents. The system has the group `contenders`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.organizations import BonabeauHierarchiesSystem
        >>> BonabeauHierarchiesSystem(10, 10, density=0.2).prepare().system_.elements[0].size
        20
    """

    field = 'social/organizations'
    topics = ('hierarchy', 'agents', 'grid', 'phase transition')
    references = ('Bonabeau, Theraulaz, and Deneubourg 1995', 'Castellano, Fortunato, and Loreto 2009')
    until = 50

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        density: float = 0.5,
        eta: float = 1.0,
        relaxation: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.width = width
        self.height = height
        self.density = density
        self.eta = eta
        self.relaxation = relaxation
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        size = int(self.width * self.height * self.density)
        cells = np.random.default_rng(self.random_state).permutation(self.width * self.height)[:size]
        contenders = ContenderElement(
            'contenders',
            eta=self.eta,
            relaxation=self.relaxation,
            x=np.asarray(cells) % self.width,
            y=np.asarray(cells) // self.width,
            size=size,
            space=LatticeSpace((self.width, self.height), boundary='periodic', capacity=1),
        )
        self.system_ = System(
            self.label,
            elements=[contenders],
            interactions=[WanderInteraction('wander', target='contenders')],
        )
