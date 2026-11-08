"""Hierarchy systems: dominance that emerges from fights between wandering agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ContenderView(AgentView):
    """View of an agent of the Bonabeau hierarchy system, with its states and parameters as attributes."""

    strength: float
    eta: float
    relaxation: float


class ContenderElement(Element):
    """Agent of the Bonabeau system, who wanders on a grid and fights the agents it runs into.

    Every step an agent's strength relaxes by the factor `1 - relaxation`, and the agent tries to step into a random
    neighbouring cell. An attacker beats the occupant with probability `1 / (1 + exp(eta * (h_occupant - h_attacker)))`,
    the winner gains 1 and the loser loses 1, and a winning attacker swaps places with the occupant. Agents differ only
    by their strength, so the swap exchanges their strengths.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        strength:
            The initial strength `h`.

        eta:
            The sensitivity of the outcome of a fight to the strength difference.

        relaxation:
            The fraction of its strength an agent loses every step.

    Examples:
        >>> from skcomplex.systems.social.organizations import ContenderElement
        >>> group = GroupElement('contenders', element=ContenderElement, size=10)
        >>> group.label, group.element.__name__, group.size
        ('contenders', 'ContenderElement', 10)
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'strength'}),
        *,
        strength: Annotated[float, StateSpec()] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        relaxation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strength = strength
        self.eta = eta
        self.relaxation = relaxation

    @classmethod
    def _get_view(cls) -> type[ContenderView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return ContenderView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the wander and fight rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.wander)]

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
        """Relax the strength, then step into a random neighbouring cell or fight its occupant.

        Args:
            agent:
                The contender.

            system:
                The system.
        """
        agent.strength *= 1 - agent.relaxation
        cells = agent.neighbor_cells()
        cell = cells[int(system.rng.integers(len(cells)))]
        occupants = [
            other
            for other in agent.neighbors()
            if isinstance(other, ContenderView)
            and tuple(np.ravel(np.asarray(other.location))) == tuple(np.ravel(np.asarray(cell)))
        ]
        if occupants:
            self._fight(agent, occupants[0], system)
            return
        agent.move_to(cell)


class BonabeauHierarchiesSystem(BaseSystem):
    """The Bonabeau system, where a hierarchy of strengths emerges above a critical density of agents.

    The system has the group `contenders`.

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

    Examples:
        >>> from skcomplex.systems.social.organizations import BonabeauHierarchiesSystem
        >>> BonabeauHierarchiesSystem(10, 10, density=0.2).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.density = density
        self.eta = eta
        self.relaxation = relaxation
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        size = int(self.width * self.height * self.density)
        cells = np.random.default_rng(self.random_state).permutation(self.width * self.height)[:size]
        contenders = GroupElement(
            'contenders',
            element=ContenderElement,
            size=size,
            values={'eta': self.eta, 'relaxation': self.relaxation},
            location=[(int(cell % self.width), int(cell // self.width)) for cell in cells],
        )
        return System(
            self.label,
            elements=[contenders],
            space=LatticeSpace((self.width, self.height), boundary='periodic', capacity=1),
        )
